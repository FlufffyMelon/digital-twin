#!/bin/bash
#SBATCH --job-name=dt-train
#SBATCH --output=logs/%x-%j.out         # submit from the repository root: sbatch slurm/train.sh --model gemma-4-12b
#SBATCH --partition=AMD7763-A100-CCP    # an A100 80GB partition: change to yours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load cuda/13.1 2>/dev/null || true

# Checkpoints of a 12B model take a few GB each: on a CephFS home with a quota,
# refuse to start when less than 15 GB is left.
if max=$(getfattr --only-values -n ceph.quota.max_bytes ~ 2>/dev/null) && [ "$max" -gt 0 ]; then
    used=$(getfattr --only-values -n ceph.dir.rbytes ~)
    free_gb=$(( (max - used) / 1024**3 ))
    if [ "$free_gb" -lt 15 ]; then
        echo "only ${free_gb}G left in the home quota, not starting" >&2
        exit 1
    fi
fi

source .venv/bin/activate

# Arguments are passed through: see python -m digital_twin.model.train --help
srun python -m digital_twin.model.train "$@"
