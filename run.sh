#!/bin/bash
#SBATCH --job-name=python
#SBATCH -w node4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # Number of CPU cores per task

# Activate your environment
source ~/DigitalTwin/.venv/bin/activate

# Set WandB project name and mode (can be overridden in train.py)
export WANDB_PROJECT="digitaltwin-llama3-finetune" # Match the one in train.py or manage centrally here
export WANDB_MODE=offline

# Ensure output directories exist
mkdir -p logs outputs # Changed to base output dir

# Run the training using srun
srun python train.py

