#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB

module load uv
module load CUDA/12.1.1
srun uv run python -m AML_BRATS.models.train_FNC2 "$@"
