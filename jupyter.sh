#!/bin/bash
#SBATCH --job-name="jupyter"
#SBATCH --partition=AMD7V12-RTX4500-CCP # RTX 4500 Ada 24GB, node211-22; A100: sbatch -p AMD7763-A100-CCP jupyter.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/13.1
source .venv/bin/activate

srun jupyter lab --no-browser --ip='*' --port 7777
