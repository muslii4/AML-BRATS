import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
from .FNC2 import SegNet, dice_score, iou_score
from .metrics import calculate_dice_all_channels
from .train_model import train_k_fold

LR = 1e-2
NUM_EPOCHS = 100
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return 1 - calculate_dice_all_channels(
            logits.sigmoid(), targets, self.smooth
        )


class DiceBCELoss(torch.nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.bce_weight = bce_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice(
            logits, targets
        )


@hydra.main(
    config_path="../config/models", config_name="SegNet", version_base=None
)
def train(cfg: DictConfig):
    bce_weight = cfg.training.bce_weight
    num_epochs = cfg.training.num_epochs
    mc_dropout = cfg.training.MC_Dropout
    num_passes = cfg.training.num_passes
    probability_dropout = cfg.training.dropout_p
    resume_from = cfg.resume_from

    loss_fn = DiceBCELoss(bce_weight=bce_weight)

    def model_fn():

        if resume_from:
            checkpoint = Path(resume_from)
            if not checkpoint.exists():
                checkpoint = PROJECT_ROOT / resume_from
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_from}"
                )
            
            model = SegNet(
                3,
                MC_Dropout=mc_dropout,
                num_passes=num_passes,
                dropout_p=probability_dropout,
            )
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            return model
        
        model = SegNet(
            3,
            MC_Dropout=mc_dropout,
            num_passes=num_passes,
            dropout_p=probability_dropout,
        )
        if cfg.initial_bias:
            if model.out.bias is None:
                raise RuntimeError
            torch.nn.init.constant_(model.out.bias, -2.0)
        return model

    def optimizer_fn(params):
        optimizer = cfg.training.optimizer
        if optimizer.type == "sgd":
            return torch.optim.SGD(
                params, lr=optimizer.sgd.lr, momentum=optimizer.sgd.momentum
            )
        elif optimizer.type == "adam":
            return torch.optim.Adam(
                params,
                lr=optimizer.adam.lr,
                weight_decay=optimizer.adam.weight_decay,
            )
        else:
            raise ValueError

    opt = cfg.training.optimizer
    opt_type = opt.type
    parts = [f"SegNetv2_HYD_{num_epochs}EPOCHS", opt_type]
    if cfg.initial_bias:
        parts.append("INBIAS")
    parts.append(f"MCDropout{cfg.training.MC_Dropout}")
    parts.append(f"passes{cfg.training.num_passes}")
    parts.append(f"drop{cfg.training.dropout_p}")
    if opt_type == "sgd":
        parts.append(f"LR{opt.sgd.lr}")
        parts.append(f"MOM{opt.sgd.momentum}")
    elif opt_type == "adam":
        parts.append(f"LR{opt.adam.lr}")
        parts.append(f"WD{opt.adam.weight_decay}")

    parts.append(f"bce{(bce_weight)}")
    if not cfg.training.augmentation:
        parts.append("NOAUG")

    run_name = "_".join(parts)

    train_k_fold(
        model_fn,
        optimizer_fn,
        loss_fn,
        metrics={"dice": dice_score, "iou": iou_score},
        epochs=cfg.training.num_epochs,
        run_name=run_name,
        augment_train=cfg.training.augmentation,
        batch_size=cfg.training.batch_size,
    )


if __name__ == "__main__":
    train()
