#!/bin/bash
#SBATCH --job-name=dt-serve
#SBATCH --output=logs/%x-%j.out         # submit from the repository root: sbatch slurm/serve.sh outputs/<run>/final
#SBATCH --partition=AMD7V12-RTX4500-CCP # a 24GB GPU (RTX 4500 Ada): a 12B model needs --load-in-4bit here
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

module load cuda/13.1 2>/dev/null || true
source .venv/bin/activate
export HF_HUB_OFFLINE=1

# The model server for the Telegram bot. It holds a GPU while it runs: stop it with scancel.
# Arguments: see python -m digital_twin.chat.server --help
srun python -m digital_twin.chat.server "$@"
