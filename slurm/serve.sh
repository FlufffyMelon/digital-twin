#!/bin/bash
#SBATCH --job-name=dt-serve
#SBATCH --output=logs/%x-%j.out     # submit from the repo root: sbatch slurm/serve.sh outputs/<run>/final
#SBATCH --partition=AMD7V12-RTX4500-CCP # RTX 4500 Ada 24GB: a 12B model needs --load-in-4bit here
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

module load cuda/13.1
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# The model server for the Telegram bot; stop it with scancel when the bot is not needed,
# it holds a GPU the whole time. Arguments: see python -m digital_twin.server --help
srun python -m digital_twin.server "$@"
