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

# The home quota (1 TB) is shared with other projects: refuse to start when it is nearly full.
free_gb=$(( ($(getfattr --only-values -n ceph.quota.max_bytes ~ 2>/dev/null) - $(getfattr --only-values -n ceph.dir.rbytes ~ 2>/dev/null)) / 1024**3 ))
if [ "$free_gb" -lt 15 ]; then
    echo "only ${free_gb}G left in the home quota, not starting" >&2
    exit 1
fi

source .venv/bin/activate
export HF_HUB_OFFLINE=1

# Arguments are passed through: see python -m digital_twin.evaluate --help
srun python -m digital_twin.evaluate "$@"
