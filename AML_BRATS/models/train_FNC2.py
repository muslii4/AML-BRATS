import torch
from .FNC2 import SegNet, dice_score, iou_score
from .train_model import train_k_fold


LR = 1e-2
NUM_EPOCHS = 100


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (num + self.smooth) / (den + self.smooth)
        valid_channels = targets.sum(dim=(2, 3)) > 0

        if valid_channels.any():
            return 1 - dice.masked_select(valid_channels).mean()

        return dice.new_tensor(1.0)


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


if __name__ == "__main__":

    def model_fn() -> torch.nn.Module:
        return SegNet(num_classes=4, MC_Dropout=False)


    loss_fn = DiceBCELoss(bce_weight=3.0)

    def optimizer_fn(params):
        return torch.optim.SGD(params, lr=LR, momentum=0.9)


    train_k_fold(
        model_fn,
        optimizer_fn,
        loss_fn,
        epochs=NUM_EPOCHS,
        run_name=f"SegNet_{NUM_EPOCHS}EPOCHS_{LR}LR",
        batch_size=64,
        metrics={"dice": dice_score, "iou": iou_score},
    )
