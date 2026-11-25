#!/bin/bash
#SBATCH -J caption-trial
#SBATCH -p h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH -o logs/%x-%j.out

# Cheap check before the full rewrite. Submit from the project root.
# Rewrites 300 random captions so the rewrite prompt can be read and judged by hand.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1
bash slurm/warm_cache.sh
bash slurm/warm_cache.sh images &

python scripts/02_rewrite_captions.py --limit 300 --out data/alt_text_trial.jsonl
