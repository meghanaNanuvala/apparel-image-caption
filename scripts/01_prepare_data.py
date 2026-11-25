"""Turn the downloaded dataset into resized JPEGs and a train/val/test split.

    python scripts/01_prepare_data.py --workers 16

Writes data/images/*.jpg, data/meta.jsonl and data/splits/{train,val,test}.jsonl.

Images with the same description (colour variants of one product) always go to the same
split, otherwise the test set would contain captions the model has already seen.
"""

import argparse
import io
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from utils import DATA_DIR, DATASET_REPO
from utils.io import write_jsonl
from utils.textnorm import split_caption

SHORT_SIDE = 768     # Florence-2 works at 768 px, so larger images only slow training down
JPEG_QUALITY = 92
VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEED = 42


def process_shard(shard_path: str, first_id: int, image_dir: str) -> list[dict]:
    rows = []
    row_id = first_id
    shard = pq.ParquetFile(shard_path)
    for batch in shard.iter_batches(batch_size=32, columns=["text", "image"]):
        texts = batch.column("text").to_pylist()
        images = batch.column("image").to_pylist()
        for text, image in zip(texts, images):
            name = f"{row_id:05d}.jpg"
            target = Path(image_dir) / name
            with Image.open(io.BytesIO(image["bytes"])) as img:
                width, height = img.size
                if not target.exists():
                    img = img.convert("RGB")
                    scale = SHORT_SIDE / min(width, height)
                    if scale < 1:
                        new_size = (round(width * scale), round(height * scale))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    tmp = target.with_suffix(".tmp")
                    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
                    tmp.rename(target)  # so a killed job never leaves a half-written image
            rows.append(
                {
                    "id": f"{row_id:05d}",
                    "image": f"images/{name}",
                    "raw_caption": text,
                    "orig_width": width,
                    "orig_height": height,
                    **split_caption(text),
                }
            )
            row_id += 1
    return rows


def grouped_split(rows):
    # group the images by description, then hand out whole groups to test, val and train
    groups = defaultdict(list)
    for row in rows:
        groups[row["body"]].append(row)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    n_test = round(len(rows) * TEST_FRAC)
    n_val = round(len(rows) * VAL_FRAC)
    splits = {"test": [], "val": [], "train": []}
    for key in keys:
        if len(splits["test"]) < n_test:
            splits["test"].extend(groups[key])
        elif len(splits["val"]) < n_val:
            splits["val"].extend(groups[key])
        else:
            splits["train"].extend(groups[key])
    for part in splits.values():
        part.sort(key=lambda r: r["id"])

    multi = [g for g in groups.values() if len(g) > 1]
    stats = {
        "groups": len(groups),
        "groups_with_several_images": len(multi),
        "images_in_those_groups": sum(len(g) for g in multi),
    }
    return splits, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    snapshot = Path(snapshot_download(DATASET_REPO, repo_type="dataset", local_files_only=True))
    shards = sorted((snapshot / "data").glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards under {snapshot}. Run scripts/00_download.py first.")

    image_dir = DATA_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # an image's id is its row number in the dataset, so ids are the same on every run
    jobs, first_id = [], 0
    for shard in shards:
        jobs.append((str(shard), first_id, str(image_dir)))
        first_id += pq.ParquetFile(shard).metadata.num_rows
    total = first_id
    print(f"{len(shards)} shards, {total} rows, {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_shard, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"  shard {done}/{len(shards)} done ({len(rows)} rows)", flush=True)
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == total, f"expected {total} rows, got {len(rows)}"

    write_jsonl(DATA_DIR / "meta.jsonl", rows)
    splits, stats = grouped_split(rows)
    for name, part in splits.items():
        write_jsonl(DATA_DIR / "splits" / f"{name}.jsonl", part)

    # no description may appear in two splits
    bodies = {name: {r["body"] for r in part} for name, part in splits.items()}
    overlap = (
        len(bodies["train"] & bodies["val"])
        + len(bodies["train"] & bodies["test"])
        + len(bodies["val"] & bodies["test"])
    )
    n_images = sum(1 for _ in image_dir.glob("*.jpg"))
    unparsed = sum(r["pattern"] is None for r in rows)

    print("\nSummary")
    print(f"  rows                      {total}")
    print(f"  images on disk            {n_images}")
    for name in ("train", "val", "test"):
        print(f"  {name:<25} {len(splits[name])}")
    print(f"  split sizes sum to        {sum(len(p) for p in splits.values())}")
    print(f"  description groups        {stats['groups']}")
    print(f"  groups with >1 image      {stats['groups_with_several_images']}"
          f" (covering {stats['images_in_those_groups']} images)")
    print(f"  group overlap across splits  {overlap}")
    print(f"  captions with unparsed prefix  {unparsed}")
    assert overlap == 0 and n_images == total


if __name__ == "__main__":
    main()
