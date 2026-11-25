#!/bin/bash
#SBATCH -J caption-rewrite
#SBATCH -p h100-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out

# Step 2: rewrite raw captions into alt-text with Qwen2.5-7B-Instruct.
# Submit from the project root. Extra arguments go to the script, for example a trial run:
#   sbatch -p h100-debug --time=00:30:00 slurm/rewrite.sbatch --limit 200 --out data/alt_text_trial.jsonl
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1
bash slurm/warm_cache.sh

python scripts/02_rewrite_captions.py "$@"
