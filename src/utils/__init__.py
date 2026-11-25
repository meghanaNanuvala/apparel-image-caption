"""Paths and model names shared by the apparel image caption project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATASET_REPO = "tomytjandra/h-and-m-fashion-caption"
FLORENCE_REPO = "florence-community/Florence-2-base-ft"
BLIP_REPO = "Salesforce/blip-image-captioning-base"
REWRITER_REPO = "Qwen/Qwen2.5-7B-Instruct"
CLIP_REPO = "openai/clip-vit-base-patch32"

# Common guidance for alt-text length; used as the target, not a hard cap.
ALT_TEXT_TARGET_CHARS = 125
