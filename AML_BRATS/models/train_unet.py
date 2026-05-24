import torch

from .train_model import train_k_fold
from .unet import UNet
from .data.data_loading import calculate_pos_weight

LR = 1e-2
NUM_EPOCHS = 100


def calculate_dice(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5
) -> torch.Tensor:
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (num + smooth) / (den + smooth)
    valid_channels = targets.sum(dim=(2, 3)) > 0

    if valid_channels.any():
        return dice.masked_select(valid_channels).mean()

    return dice.new_tensor(0.0)


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


if __name__ == "__main__":
    pos_weights = calculate_pos_weight(
        torch.load("data/metadata.pt")
    )  # Calculate pos weights for the dataset
    loss_fn = DiceBCELoss(bce_weight=3.0, pos_weight=pos_weights)

    def model_fn():
        return UNet(3)

    def optimizer_fn(params):
        return torch.optim.SGD(params, lr=LR, momentum=0.9)

    train_k_fold(
        model_fn,
        optimizer_fn,
        loss_fn,
        metrics={"dice": calculate_dice},
        epochs=NUM_EPOCHS,
        run_name=f"UNET_MDICEBCE3_SGD_{NUM_EPOCHS}EPOCHS_{LR}LR",
    )
