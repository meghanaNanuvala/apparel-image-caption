"""Fine-tune Florence-2-base on (image, alt-text) pairs.

    python scripts/04_train.py                                      # full training run
    python scripts/04_train.py --overfit 32 --epochs 60 --lr 5e-5   # sanity check on 32 examples

The model gets the image and the <CAPTION> prompt and has to write the alt-text.
After every epoch we check validation loss and CIDEr and keep the best model in
outputs/checkpoints/best.
"""

import argparse
import json
import random
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Florence2ForConditionalGeneration, get_linear_schedule_with_warmup

from utils import DATA_DIR, FLORENCE_REPO, OUTPUT_DIR
from utils.data import CaptionDataset, FlorenceCollator, load_split, to_device
from utils.metrics import cider
from utils.models import BEST_CHECKPOINT, florence_generate, load_image

SEED = 42
DEVICE = "cuda"
WEIGHT_DECAY = 0.01
WARMUP = 0.05        # share of the steps used to warm the learning rate up
CIDER_IMAGES = 300   # generating captions is slow, so CIDEr is measured on part of val


def compute_loss(model, batch):
    inputs = to_device(batch["model_inputs"], DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):   # mixed precision, much faster on the H100
        return model(**inputs).loss


@torch.no_grad()
def validate(model, processor, val_loader, cider_rows):
    model.eval()

    # loss on the whole validation split
    total, count = 0.0, 0
    for batch in val_loader:
        n = len(batch["ids"])
        total += compute_loss(model, batch).item() * n
        count += n

    # CIDEr: generate captions and compare them with the references
    predictions = {}
    for i in range(0, len(cider_rows), 32):
        rows = cider_rows[i:i + 32]
        images = [load_image(DATA_DIR / r["image"]) for r in rows]
        for r, caption in zip(rows, florence_generate(model, processor, images)):
            predictions[r["id"]] = caption
    references = {r["id"]: r["alt_text"] for r in cider_rows}

    return total / count, cider(predictions, references)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--overfit", type=int, default=0,
                        help="train and validate on the first N examples only (nothing is saved)")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split("train")
    val_rows = load_split("val")
    if args.overfit:
        train_rows = val_rows = train_rows[:args.overfit]
    cider_rows = random.sample(val_rows, min(CIDER_IMAGES, len(val_rows)))
    print(f"train {len(train_rows)} | val {len(val_rows)} | CIDEr on {len(cider_rows)} val images")

    # float32 is needed here: the checkpoint is stored in float16, and training in
    # float16 turned every weight into NaN on the first step
    processor = AutoProcessor.from_pretrained(FLORENCE_REPO)
    model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_REPO, dtype=torch.float32)
    model.to(DEVICE)

    collate = FlorenceCollator(processor)
    train_loader = DataLoader(CaptionDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate, num_workers=8, pin_memory=True,
                              persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(CaptionDataset(val_rows), batch_size=args.batch_size, collate_fn=collate,
                            num_workers=4)

    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total_steps), total_steps)
    print(f"{len(train_loader)} steps per epoch, {total_steps} steps in total")

    started = time.time()
    step_log, epoch_log = [], []

    # epoch 0 is the pretrained model before any training, so we know where we start from
    val_loss, val_cider = validate(model, processor, val_loader, cider_rows)
    print(f"epoch 0: val loss {val_loss:.4f} | val CIDEr {val_cider:.1f}")
    epoch_log.append({"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_cider": val_cider,
                      "minutes": (time.time() - started) / 60})
    best_cider, best_epoch = val_cider, 0

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for i, batch in enumerate(train_loader, start=1):
            loss = compute_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            loss_sum += loss.item()
            if step % 20 == 0:
                step_log.append({"step": step, "epoch": epoch, "loss": loss_sum / i,
                                 "lr": scheduler.get_last_lr()[0]})
            if step % 100 == 0:
                print(f"  step {step}/{total_steps} loss {loss_sum / i:.4f}")
        train_loss = loss_sum / len(train_loader)

        val_loss, val_cider = validate(model, processor, val_loader, cider_rows)
        minutes = (time.time() - started) / 60
        print(f"epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val CIDEr {val_cider:.1f} | {minutes:.0f} min")
        epoch_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                          "val_cider": val_cider, "minutes": minutes})

        if args.overfit:
            continue   # sanity check only, don't touch the real checkpoint or logs

        if val_cider > best_cider:
            best_cider, best_epoch = val_cider, epoch
            model.save_pretrained(BEST_CHECKPOINT)
            processor.save_pretrained(BEST_CHECKPOINT)
        pd.DataFrame(step_log).to_csv(OUTPUT_DIR / "train_log.csv", index=False)
        pd.DataFrame(epoch_log).to_csv(OUTPUT_DIR / "val_log.csv", index=False)

    if args.overfit:
        print("overfit check done: the loss should be close to 0 and CIDEr far above epoch 0")
        return

    summary = {"base_model": FLORENCE_REPO, "epochs": args.epochs, "batch_size": args.batch_size,
               "lr": args.lr, "train_examples": len(train_rows), "minutes": minutes,
               "best_epoch": best_epoch, "best_val_cider": best_cider,
               "pretrained_val_cider": epoch_log[0]["val_cider"]}
    (OUTPUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done. best epoch {best_epoch}, val CIDEr {best_cider:.1f} "
          f"(pretrained: {epoch_log[0]['val_cider']:.1f})")


if __name__ == "__main__":
    main()
