import hydra
import torch
from omegaconf import DictConfig

from .metrics import calculate_dice
from .train_model import train_k_fold
from .unet import UNet


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return 1 - calculate_dice(logits.sigmoid(), targets, self.smooth)


class DiceBCELoss(torch.nn.Module):
    def __init__(self, bce_weight: float = 1.0, pos_weight: torch.Tensor = None, smooth: float = 1e-5) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth)
        self.bce_weight = bce_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice(
            logits, targets
        )


@hydra.main(
    config_path="../config/models", config_name="unet", version_base=None
)
def train(cfg: DictConfig):
    bce_weight = cfg.training.bce_weight
    num_epochs = cfg.training.num_epochs

    loss_fn = DiceBCELoss(bce_weight=bce_weight)

    def model_fn():
        model = UNet(3, cfg.batch_norm)
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
    parts = [f"UNET_HYD_{num_epochs}EPOCHS", opt_type]
    if cfg.initial_bias:
        parts.append("INBIAS")
    if cfg.batch_norm:
        parts.append("BNORM")
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
        metrics={"dice": calculate_dice},
        epochs=cfg.training.num_epochs,
        run_name=run_name,
        augment_train=cfg.training.augmentation,
        batch_size=cfg.training.batch_size,
    )


if __name__ == "__main__":
    train()
