"""Caption the test split with one or more systems and save the predictions.

    python scripts/03_predict.py                                  # the three zero-shot baselines
    python scripts/03_predict.py --systems florence_finetuned     # after training

Writes outputs/preds/{system}.jsonl with rows: id, prediction.
"""

import argparse
import time

from utils import DATA_DIR, OUTPUT_DIR
from utils.data import load_split
from utils.io import write_jsonl
from utils.models import SYSTEMS, load_image, load_system

ZERO_SHOT = ["blip_zero_shot", "florence_zero_shot", "florence_zero_shot_detailed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=ZERO_SHOT, choices=sorted(SYSTEMS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    args = parser.parse_args()

    rows = load_split(args.split)[:args.limit]
    print(f"{len(rows)} {args.split} images", flush=True)

    for name in args.systems:
        started = time.time()
        system = load_system(name)
        predictions = []
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            texts = system.caption([load_image(DATA_DIR / r["image"]) for r in chunk])
            predictions += [{"id": r["id"], "prediction": t} for r, t in zip(chunk, texts)]
        # a quick test (--split or --limit) gets its own file, so it cannot overwrite the real predictions
        suffix = "" if args.split == "test" and args.limit is None else f".{args.split}{args.limit or ''}"
        write_jsonl(OUTPUT_DIR / "preds" / f"{name}{suffix}.jsonl", predictions)
        print(f"{name}: {len(predictions)} captions in {time.time() - started:.0f}s", flush=True)
        for p in predictions[:3]:
            print(f"    {p['prediction']}", flush=True)
        del system  # free the GPU before the next model is loaded


if __name__ == "__main__":
    main()
