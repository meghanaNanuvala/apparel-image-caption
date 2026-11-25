"""The captioning models. Each one has caption(images) which returns a list of strings."""

from pathlib import Path

import torch
from PIL import Image

from utils import BLIP_REPO, FLORENCE_REPO, OUTPUT_DIR
from utils.textnorm import tidy_output

FLORENCE_TASK = "<CAPTION>"  # Florence-2 task prompt, used for training and for captioning
BEST_CHECKPOINT = OUTPUT_DIR / "checkpoints" / "best"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


@torch.inference_mode()
def florence_generate(model, processor, images: list[Image.Image], task: str = FLORENCE_TASK,
                      max_new_tokens: int = 96, num_beams: int = 3) -> list[str]:
    """Caption a batch of images with a Florence-2 model."""
    inputs = processor(text=[task] * len(images), images=images, return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams,
                         do_sample=False)
    return [tidy_output(t) for t in processor.batch_decode(ids, skip_special_tokens=True)]


class FlorenceCaptioner:
    """Florence-2, either the pretrained checkpoint (zero-shot) or our fine-tuned folder."""

    def __init__(self, checkpoint=FLORENCE_REPO, task=FLORENCE_TASK, device=None,
                 max_new_tokens=96, num_beams=3):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.task = task
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        # the pretrained checkpoint is stored in float16 and ours in float32;
        # load both as float32 so they are compared at the same precision
        self.model = Florence2ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32)
        self.model.to(device or pick_device()).eval()

    def caption(self, images: list[Image.Image]) -> list[str]:
        return florence_generate(self.model, self.processor, images, self.task,
                                 self.max_new_tokens, self.num_beams)


class BlipCaptioner:
    """BLIP-base, the zero-shot baseline."""

    def __init__(self, checkpoint=BLIP_REPO, device=None, max_new_tokens=60, num_beams=3):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = device or pick_device()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.processor = BlipProcessor.from_pretrained(checkpoint)
        self.model = BlipForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def caption(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  num_beams=self.num_beams, do_sample=False)
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [tidy_output(t) for t in texts]


# name -> (label shown in tables, function that loads the model).
# Keep <angle brackets> out of the labels, they disappear in HTML and Markdown tables.
SYSTEMS = {
    "blip_zero_shot": ("BLIP-base, zero-shot", lambda device: BlipCaptioner(device=device)),
    "florence_zero_shot": (
        "Florence-2-base, zero-shot (caption prompt)",
        lambda device: FlorenceCaptioner(task="<CAPTION>", device=device),
    ),
    "florence_zero_shot_detailed": (
        "Florence-2-base, zero-shot (detailed-caption prompt)",
        lambda device: FlorenceCaptioner(task="<DETAILED_CAPTION>", device=device),
    ),
    "florence_finetuned": (
        "Florence-2-base, fine-tuned (ours)",
        lambda device: FlorenceCaptioner(checkpoint=str(BEST_CHECKPOINT), device=device),
    ),
}


def load_system(name: str, device=None):
    if name not in SYSTEMS:
        raise KeyError(f"unknown system '{name}'; choose from {sorted(SYSTEMS)}")
    if name == "florence_finetuned" and not Path(BEST_CHECKPOINT).exists():
        raise FileNotFoundError(f"{BEST_CHECKPOINT} not found; run scripts/04_train.py first")
    return SYSTEMS[name][1](device)
