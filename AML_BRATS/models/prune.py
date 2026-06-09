import hydra
import torch
from omegaconf import DictConfig
from AML_BRATS.models.FNC2 import SegNet
from AML_BRATS.models.train_model import pruner_random, pruner_structured
from AML_BRATS.models.unet import UNet
from pathlib import Path
from AML_BRATS.models.train_FNC2 import train as train_segnet
from AML_BRATS.models.train_unet import train as train_unet

# project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_model(model_name: str, model_state: str, device: torch.device) -> torch.nn.Module:
    if model_name == "SegNet":
        model= SegNet(num_classes=3, MC_Dropout=False, dropout_p=0.0, num_passes=1)
    elif model_name == "UNet":
        model= UNet(num_classes=3, batch_norm=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    candidate = Path(model_state)
    if candidate.exists():
        state = candidate
    else:
        state = PROJECT_ROOT / "models" / f"{model_state}.pkl"
        if not state.exists():
            raise FileNotFoundError(f"Model state file not found: {state}")
    
    model = model.to(device)
    model.to(device)
    model.load_state_dict(
        torch.load(state, weights_only=True, map_location=device),
    )
    return model

@hydra.main(
    config_path="../config/models", config_name="prune", version_base=None
)
def train(cfg: DictConfig):
    amount = cfg.amount
    model_name = cfg.model_name
    method = cfg.pruning_type
    model_state = cfg.model_state
    model_name_for_file = Path(model_state).stem
    save_dir = PROJECT_ROOT / "model_pruned"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, model_state, device=device)
    MODEL_NAME = model_name_for_file
    
    if method == "random":
        print("prunting with random")
        model = pruner_random(model, amount=amount)
        torch.save(model.state_dict(), save_dir / f"{MODEL_NAME}_pruned_{method}_{amount}.pkl")
        print("completed random pruning")

    elif method == "structured":
        print("pruning with structured")
        model, before, after = pruner_structured(model, input_shape=(1,4,240,240), amount=amount)
        torch.save(model, save_dir / f"{MODEL_NAME}_pruned_{method}_{amount}.pth")
        print("Params before:", before, "after:", after)
        print("completed structured pruning")
    else:
        raise ValueError(f"Unsupported pruning method: {method} choose 'random' or 'structured'")
    
    return model

#cd /home/joanl/AML/AML-BRATS
#source .venv/bin/activate
#python -m AML_BRATS.models.prune \
#  model_name=UNet \
#  model_state=models/UNET_HYD_25EPOCHS_adam_BNORM_LR0.0001_WD0.01_bce1_NOAUG_BS64_FOLD1_final.pkl \
#  pruning_type=random \
#  amount=0.2 \
#  hydra.job.chdir=false
if __name__ == "__main__":
    train()