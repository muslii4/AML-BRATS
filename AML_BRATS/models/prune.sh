uv run python -m AML_BRATS.models.prune \
 model_name=UNet \
 model_state=models/UNET_HYD_25EPOCHS_adam_BNORM_LR0.0001_WD0.01_bce1_NOAUG_BS64_FOLD1_final.pkl \
 pruning_type=random \
 amount=0.2 \
 hydra.job.chdir=false