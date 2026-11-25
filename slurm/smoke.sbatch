#!/bin/bash
#SBATCH -J caption-smoke
#SBATCH -p h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out

# Checks that the GPU node can run the project: CUDA visible, models load offline,
# Florence-2 captions an image on the GPU.
# Submit from the project root:  sbatch slurm/smoke.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1
bash slurm/warm_cache.sh

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("device:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())

from utils import DATA_DIR
from utils.io import read_jsonl
from utils.models import FlorenceCaptioner, load_image

rows = read_jsonl(DATA_DIR / "splits" / "test.jsonl")[:4]
captioner = FlorenceCaptioner()
for row, text in zip(rows, captioner.caption([load_image(DATA_DIR / r["image"]) for r in rows])):
    print(f"{row['id']} | zero-shot: {text}\n      | raw:       {row['raw_caption']}")
print("SMOKE TEST PASSED")
EOF
