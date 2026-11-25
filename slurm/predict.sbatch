#!/bin/bash
#SBATCH -J caption-predict
#SBATCH -p h100-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x-%j.out

# Steps 3 and 5: caption the test split, then score everything that has predictions.
# Submit from the project root.
#   sbatch slurm/predict.sbatch                                   # zero-shot baselines
#   sbatch slurm/predict.sbatch --systems florence_finetuned      # after training
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1
bash slurm/warm_cache.sh
bash slurm/warm_cache.sh images &

python scripts/03_predict.py "$@"
python scripts/05_evaluate.py
