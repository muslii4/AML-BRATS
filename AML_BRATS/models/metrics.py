import torch


def calculate_dice(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5
) -> torch.Tensor:
    """Calculate Dice score for probability or binary masks."""
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (num + smooth) / (den + smooth)
    valid_channels = targets.sum(dim=(2, 3)) > 0

    if valid_channels.any():
        return dice.masked_select(valid_channels).mean()

    return dice.new_tensor(0.0)


def calculate_precision(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5
) -> torch.Tensor:
    """Calculate precision (TP / (TP + FP)) for binary masks."""
    tp = (probs * targets).sum(dim=(2, 3))
    fp = (probs * (1 - targets)).sum(dim=(2, 3))
    
    precision = (tp + smooth) / (tp + fp + smooth)
    valid_channels = targets.sum(dim=(2, 3)) > 0
    
    if valid_channels.any():
        return precision.masked_select(valid_channels).mean()
    
    return precision.new_tensor(0.0)


def calculate_recall(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5
) -> torch.Tensor:
    """Calculate recall (TP / (TP + FN)) for binary masks."""
    tp = (probs * targets).sum(dim=(2, 3))
    fn = ((1 - probs) * targets).sum(dim=(2, 3))
    
    recall = (tp + smooth) / (tp + fn + smooth)
    valid_channels = targets.sum(dim=(2, 3)) > 0
    
    if valid_channels.any():
        return recall.masked_select(valid_channels).mean()
    
    return recall.new_tensor(0.0)