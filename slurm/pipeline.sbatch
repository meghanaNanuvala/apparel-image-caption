#!/bin/bash
#SBATCH -J caption-pipeline
#SBATCH -p h100-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out

# Steps 2 to 5 in one job: rewrite captions, zero-shot baselines, fine-tune, evaluate.
# Running them together pays the slow first import from the shared filesystem only once.
# Submit from the project root:  sbatch slurm/pipeline.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1
bash slurm/warm_cache.sh
bash slurm/warm_cache.sh images &

echo "=== step 2: rewrite captions ===";      python scripts/02_rewrite_captions.py --batch-size 128
echo "=== step 3: zero-shot baselines ===";   python scripts/03_predict.py
python scripts/05_evaluate.py   # baseline numbers early; also proves the evaluation runs before training
echo "=== step 4: fine-tune ===";             python scripts/04_train.py
echo "=== step 5: predict and evaluate ===";  python scripts/03_predict.py --systems florence_finetuned
python scripts/05_evaluate.py
