#!/bin/bash
#SBATCH --job-name=digitaltwin
#SBATCH --output=logs/%x-%j.out     # submit from the repo root: sbatch slurm/train.sh --model gemma-4-e4b
#SBATCH --partition=AMD7763-A100-CCP    # A100 80GB, node208-24; Ada: sbatch -p AMD7V12-RTX4500-CCP slurm/train.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --mem=64G

module load cuda/13.1

# Activate your environment (built with `uv sync`)
source .venv/bin/activate

export WANDB_PROJECT="digitaltwin"
export WANDB_MODE=offline

# Arguments are passed through: see python -m digital_twin.train --help
srun python -m digital_twin.train "$@"
