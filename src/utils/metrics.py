"""Caption metrics: BLEU, ROUGE-L, METEOR and CIDEr (against a reference), CLIPScore (against
the image), and simple alt-text checks.

All scores are multiplied by 100. CIDEr's raw range is 0 to 10, so it can reach 1000.
The reference metrics compare lowercase words, so capitals and punctuation do not count.
"""

import torch

from utils import ALT_TEXT_TARGET_CHARS, CLIP_REPO
from utils.models import load_image, pick_device
from utils.textnorm import mentions_image, mentions_person, non_visible_terms, tokenize


def _as_coco(texts: dict[str, str], ids: list[str]) -> dict[str, list[str]]:
    # pycocoevalcap wants {id: ["words joined by spaces"]}; an empty string breaks the scoring
    return {i: [" ".join(tokenize(texts[i])) or "empty"] for i in ids}


def cider(predictions: dict[str, str], references: dict[str, str]) -> float:
    from pycocoevalcap.cider.cider import Cider

    ids = sorted(references)
    score, _ = Cider().compute_score(_as_coco(references, ids), _as_coco(predictions, ids))
    return float(score) * 100


def reference_metrics(predictions: dict[str, str], references: dict[str, str]) -> dict[str, float]:
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge

    ids = sorted(references)
    missing = [i for i in ids if i not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} test images have no prediction, e.g. {missing[:3]}")
    gts, res = _as_coco(references, ids), _as_coco(predictions, ids)

    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    rouge, _ = Rouge().compute_score(gts, res)
    meteor = sum(
        meteor_score([tokenize(references[i])], tokenize(predictions[i])) for i in ids
    ) / len(ids)
    return {
        "BLEU-4": bleu[3] * 100,
        "ROUGE-L": float(rouge) * 100,
        "METEOR": meteor * 100,
        "CIDEr": cider(predictions, references),
    }


def alt_text_checks(texts: list[str]) -> dict[str, float]:
    """Length and style checks over a list of captions. Percentages are 0-100."""
    n = len(texts)
    stripped = [t.strip() for t in texts]

    def share(flags) -> float:
        return 100 * sum(flags) / n

    return {
        "mean chars": sum(len(t) for t in stripped) / n,
        f"<= {ALT_TEXT_TARGET_CHARS} chars %": share(len(t) <= ALT_TEXT_TARGET_CHARS for t in stripped),
        "sentence case + period %": share(bool(t) and t[0].isupper() and t.endswith(".") for t in stripped),
        "non-visible term %": share(bool(non_visible_terms(t)) for t in stripped),
        "mentions a person %": share(mentions_person(t) for t in stripped),
        "mentions the image %": share(mentions_image(t) for t in stripped),
    }


class ClipScorer:
    """CLIPScore (Hessel et al., 2021): 2.5 * max(cos(image, text), 0), reported x100.
    It needs no reference caption, so it does not favour one writing style."""

    def __init__(self, device=None, batch_size: int = 64):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or pick_device()
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(CLIP_REPO, dtype=torch.float32).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_REPO)
        self._image_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _unit(features) -> torch.Tensor:
        # transformers 5 returns an output object here, older versions the tensor itself
        if not torch.is_tensor(features):
            features = features.pooler_output
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()

    @torch.inference_mode()
    def _embed_images(self, paths: list[str]) -> torch.Tensor:
        # every system is scored on the same images, so embed each image only once
        todo = [p for p in dict.fromkeys(paths) if p not in self._image_cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            inputs = self.processor(images=[load_image(p) for p in chunk],
                                    return_tensors="pt").to(self.device)
            embeds = self._unit(self.model.get_image_features(**inputs))
            for path, embed in zip(chunk, embeds):
                self._image_cache[path] = embed
        return torch.stack([self._image_cache[p] for p in paths])

    @torch.inference_mode()
    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [t if t.strip() else "empty" for t in texts[start:start + self.batch_size]]
            # CLIP reads at most 77 tokens, longer captions are cut off
            inputs = self.processor(text=chunk, return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.device)
            out.append(self._unit(self.model.get_text_features(**inputs)))
        return torch.cat(out)

    def score(self, image_paths: list[str], texts: list[str]) -> list[float]:
        images = self._embed_images([str(p) for p in image_paths])
        captions = self._embed_texts(texts)
        cosine = (images * captions).sum(dim=-1)
        return (250 * cosine.clamp(min=0)).tolist()
