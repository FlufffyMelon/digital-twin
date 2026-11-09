#!/bin/bash
#SBATCH --job-name="jupyter"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G

srun jupyter lab --no-browser --ip='*' --port 7777

