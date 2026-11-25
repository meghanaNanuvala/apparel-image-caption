"""Loading a data split, plus the Dataset and batch collator used for training."""

import torch
from torch.utils.data import Dataset

from utils import DATA_DIR
from utils.io import read_jsonl
from utils.models import FLORENCE_TASK, load_image


def load_split(name):
    """Rows of one split with their alt-text. Rewrites that failed the checks are left out."""
    alt = {r["id"]: r["alt_text"] for r in read_jsonl(DATA_DIR / "alt_text.jsonl") if r["ok"]}
    rows = read_jsonl(DATA_DIR / "splits" / f"{name}.jsonl")
    return [{**row, "alt_text": alt[row["id"]]} for row in rows if row["id"] in alt]


class CaptionDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {"id": row["id"], "image": load_image(DATA_DIR / row["image"]),
                "alt_text": row["alt_text"]}


class FlorenceCollator:
    """Turns a list of examples into a batch: inputs = prompt + image, labels = alt-text."""

    def __init__(self, processor, task: str = FLORENCE_TASK, max_target_tokens: int = 64):
        self.processor = processor
        self.task = task
        self.max_target_tokens = max_target_tokens

    def __call__(self, batch: list[dict]) -> dict:
        inputs = self.processor(text=[self.task] * len(batch), images=[b["image"] for b in batch],
                                return_tensors="pt")
        labels = self.processor.tokenizer(
            [b["alt_text"] for b in batch], padding=True, truncation=True,
            max_length=self.max_target_tokens, return_tensors="pt",
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # -100 = ignored by the loss
        return {
            "ids": [b["id"] for b in batch],
            "references": [b["alt_text"] for b in batch],
            "model_inputs": {**inputs, "labels": labels},
        }


def to_device(model_inputs: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in model_inputs.items()}
