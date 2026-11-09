#!/bin/bash
#SBATCH --job-name=dt-eval
#SBATCH --output=logs/%x-%j.out     # submit from the repo root: sbatch slurm/eval.sh generate outputs/<run>/final --name x
#SBATCH --partition=AMD7V12-RTX4500-CCP # RTX 4500 Ada 24GB; the judge also fits here in 4 bit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

module load cuda/13.1
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Arguments are passed through: see python -m digital_twin.evaluate --help
srun python -m digital_twin.evaluate "$@"
