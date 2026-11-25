"""Download the dataset and all models into ./.cache. Run once on the login node.

    python scripts/00_download.py                 # everything
    python scripts/00_download.py --only dataset  # just the H&M data

After this the GPU jobs run offline (HF_HUB_OFFLINE=1) and load everything from ./.cache.
"""

import argparse
import os
import sys

from huggingface_hub import list_repo_files, snapshot_download

from utils import BLIP_REPO, CLIP_REPO, DATASET_REPO, FLORENCE_REPO, PROJECT_ROOT, REWRITER_REPO

# weight formats we never load (TensorFlow, Flax, ONNX, ...)
UNUSED_FORMATS = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "openvino/*", "*.tflite"]


def check_cache_is_local() -> None:
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home.startswith(str(PROJECT_ROOT)):
        sys.exit(f"HF_HOME is '{hf_home}', not inside the project. Run `source env.sh` first.")


def download_model(repo: str) -> None:
    files = list_repo_files(repo)
    ignore = list(UNUSED_FORMATS)
    # many repos ship the same weights twice (.bin and .safetensors), one copy is enough
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")
    path = snapshot_download(repo, ignore_patterns=ignore)
    print(f"model   {repo} -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["dataset", "models", "nltk"], default=None)
    args = parser.parse_args()
    check_cache_is_local()

    if args.only in (None, "dataset"):
        path = snapshot_download(DATASET_REPO, repo_type="dataset")
        print(f"dataset {DATASET_REPO} -> {path}", flush=True)

    if args.only in (None, "models"):
        for repo in (FLORENCE_REPO, BLIP_REPO, CLIP_REPO, REWRITER_REPO):
            download_model(repo)

    if args.only in (None, "nltk"):
        import nltk

        target = os.environ["NLTK_DATA"]
        os.makedirs(target, exist_ok=True)
        os.chmod(target, 0o755)  # NLTK refuses to install into a group-writable folder
        for package in ("wordnet", "omw-1.4"):  # METEOR needs WordNet
            if not nltk.download(package, download_dir=target, quiet=True):
                sys.exit(f"NLTK could not download '{package}' into {target}")
        print(f"nltk    wordnet -> {target}", flush=True)


if __name__ == "__main__":
    main()
