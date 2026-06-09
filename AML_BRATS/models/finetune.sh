#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB

module load uv
module load CUDA/12.1.1
srun uv run python -m AML_BRATS.models.train_FNC2 resume_from=model_pruned/UNET_HYD_25EPOCHS_adam_BNORM_LR0.0001_WD0.01_bce1_NOAUG_BS64_FOLD1_final_pruned_random_0.2.pkl "$@"