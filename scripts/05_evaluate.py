"""Score every system that has predictions for the test split.

    python scripts/05_evaluate.py

Reads outputs/preds/{system}.jsonl and writes:
    outputs/results.csv, outputs/results.md     one row per system
    outputs/test_predictions.csv                every test image with each system's caption

The table also has two rows that are not model predictions, for comparison: the reference
alt-text and the raw H&M caption. They only get CLIPScore and the alt-text checks.
"""

import pandas as pd

from utils import DATA_DIR, OUTPUT_DIR
from utils.data import load_split
from utils.io import read_jsonl
from utils.metrics import ClipScorer, alt_text_checks, reference_metrics
from utils.models import SYSTEMS


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| system | " + " | ".join(frame.columns) + " |"
    rule = "|---|" + "---:|" * len(frame.columns)
    lines = [header, rule]
    for name, row in frame.iterrows():
        cells = ["" if pd.isna(v) else f"{v:.1f}" for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_split("test")
    ids = [r["id"] for r in rows]
    references = {r["id"]: r["alt_text"] for r in rows}
    image_paths = [DATA_DIR / r["image"] for r in rows]
    print(f"{len(rows)} test images with a reference", flush=True)

    scorer = ClipScorer()
    table = {}
    per_image = pd.DataFrame(
        {"id": ids, "image": [r["image"] for r in rows],
         "raw_caption": [r["raw_caption"] for r in rows], "reference": [references[i] for i in ids]}
    )

    def add_row(label: str, texts: list[str], with_reference_metrics: bool) -> list[float]:
        clip = scorer.score(image_paths, texts)
        row = {}
        if with_reference_metrics:
            row.update(reference_metrics(dict(zip(ids, texts)), references))
        row["CLIPScore"] = sum(clip) / len(clip)
        row.update(alt_text_checks(texts))
        table[label] = row
        print(f"{label}: " + ", ".join(f"{k} {v:.1f}" for k, v in row.items()), flush=True)
        return clip

    for name, (label, _) in SYSTEMS.items():
        path = OUTPUT_DIR / "preds" / f"{name}.jsonl"
        if not path.exists():
            print(f"skipping {name}: {path.name} not found", flush=True)
            continue
        predictions = {r["id"]: r["prediction"] for r in read_jsonl(path)}
        texts = [predictions[i] for i in ids]
        per_image[name] = texts
        per_image[f"{name}_clipscore"] = add_row(label, texts, with_reference_metrics=True)

    add_row("Reference alt-text (LLM rewrite of the H&M caption)", [references[i] for i in ids], False)
    add_row("Raw H&M caption", [r["raw_caption"] for r in rows], False)

    results = pd.DataFrame(table).T
    metric_columns = ["BLEU-4", "ROUGE-L", "METEOR", "CIDEr"]
    results = results[[c for c in metric_columns if c in results] +
                      [c for c in results.columns if c not in metric_columns]]
    results.index.name = "system"
    results.to_csv(OUTPUT_DIR / "results.csv")
    (OUTPUT_DIR / "results.md").write_text(markdown_table(results))
    per_image.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print("\n" + markdown_table(results))


if __name__ == "__main__":
    main()
