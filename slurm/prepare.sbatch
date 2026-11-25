#!/bin/bash
#SBATCH -J caption-prepare
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH -o logs/%x-%j.out

# Submit from the project root:  sbatch slurm/prepare.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the project root}"
source env.sh
set -u
export HF_HUB_OFFLINE=1

python scripts/01_prepare_data.py --workers "${SLURM_CPUS_PER_TASK:-4}"
