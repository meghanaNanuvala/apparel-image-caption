#!/bin/bash
#SBATCH -J caption-train
#SBATCH -p h100-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH -o logs/%x-%j.out

# Step 4: fine-tune Florence-2-base (about 35 minutes). Submit from the project root.
# Extra arguments go to the script, for example the sanity check on 32 examples:
#   sbatch slurm/train.sbatch --overfit 32 --epochs 60 --lr 5e-5
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1
bash slurm/warm_cache.sh
bash slurm/warm_cache.sh images &

python scripts/04_train.py "$@"
