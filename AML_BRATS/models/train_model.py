from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.data_loading import BRATSDataset, get_dataset_folds
from .metrics import (
    calculate_dice,
    calculate_precision,
    calculate_recall,
)


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.to(device)
    model.train()

    train_loss = 0.0
    total_samples = 0

    for datapoint in tqdm(dataloader, desc="Training"):
        X = datapoint["image"]
        y_true = datapoint["mask"]
        X = X.to(device)
        y_true = y_true.to(device)

        batch_size = X.size(0)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y_true)

        train_loss += loss.item() * batch_size
        total_samples += batch_size

        loss.backward()
        optimizer.step()

    return train_loss / total_samples if total_samples else 0.0


def validation_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device,
) -> float:
    """Run one validation epoch and return the average loss."""
    model.to(device)
    model.eval()
    val_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for datapoint in tqdm(dataloader, desc="Validation"):
            X = datapoint["image"]
            y_true = datapoint["mask"]
            X = X.to(device)
            y_true = y_true.to(device)

            batch_size = X.size(0)
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)

            val_loss += loss.item() * batch_size
            total_samples += batch_size

    return val_loss / total_samples if total_samples else 0.0


def compute_metrics_from_outputs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    metrics: dict[str, Callable[..., torch.Tensor]],
) -> dict[str, float]:
    """Compute all metrics from collected probabilities and targets."""
    metric_avgs: dict[str, float] = {}

    for metric_name, metric_fn in metrics.items():
        metric_val = metric_fn(probs, targets)

        if metric_val.numel() == 1:
            metric_val = float(metric_val.item())
        else:
            metric_val = float(metric_val.mean().item())

        metric_avgs[metric_name] = metric_val

    return metric_avgs


@torch.no_grad()
def collect_validation_outputs(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all validation probabilities and targets for epoch-level metrics."""
    model.to(device)
    model.eval()

    probs_batches: list[torch.Tensor] = []
    targets_batches: list[torch.Tensor] = []

    for datapoint in tqdm(dataloader, desc="Threshold search"):
        X = datapoint["image"].to(device)
        y_true = datapoint["mask"].to(device)

        probs_batches.append(model(X).sigmoid().cpu())
        targets_batches.append(y_true.cpu())

    if not probs_batches:
        empty = torch.empty(0)
        return empty, empty

    return torch.cat(probs_batches, dim=0), torch.cat(targets_batches, dim=0)


def calculate_thresholded_dice(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Calculate Dice after converting probabilities into binary masks."""
    binary_probs = (probs >= threshold).to(targets.dtype)
    return calculate_dice(binary_probs, targets, smooth)


def find_best_thresholded_dice(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 101,
) -> tuple[float, float]:
    """Find the threshold that maximizes Dice over the full validation set."""
    if probs.numel() == 0 or targets.numel() == 0:
        return 0.0, 0.0

    thresholds = torch.linspace(0.0, 1.0, steps=num_thresholds)
    best_threshold = 0.0
    best_score = float("-inf")

    for threshold in thresholds:
        score = float(
            calculate_thresholded_dice(
                probs, targets, float(threshold.item())
            ).item()
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold.item())

    return best_threshold, best_score


def _get_device() -> torch.device:
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def train_model(
    model: torch.nn.Module,
    train_dl: DataLoader,
    validation_dl: DataLoader,
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    run_name: str,
    metrics: dict[str, Callable[..., torch.Tensor]] = {},
    threshold_search_points: int = 101,
    device: torch.device = _get_device(),
) -> tuple[float, float, float]:
    """
    Train any model with the specified loss function and optimizer.
    Train and validation losses are saved using tensorboard.
    Final loss values are returned, and the final model state is saved.
    """
    writer = SummaryWriter(f"runs/{run_name}")

    train_loss = 0.0
    val_loss = 0.0
    best_threshold = 0.5  # Default threshold

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        val_loss = validation_epoch(validation_dl, model, loss_fn, device)
        writer.add_scalar("Loss/val", val_loss, epoch)

        val_probs, val_targets = collect_validation_outputs(
            validation_dl, model, device
        )

        if metrics:
            metric_avgs = compute_metrics_from_outputs(
                val_probs, val_targets, metrics
            )
            for metric_name, metric_avg in metric_avgs.items():
                writer.add_scalar(
                    f"Metrics/{metric_name}/val", metric_avg, epoch
                )
                print(f"Validation {metric_name}: {metric_avg}")

        best_threshold, best_thresholded_dice = find_best_thresholded_dice(
            val_probs, val_targets, threshold_search_points
        )
        writer.add_scalar(
            "Metrics/thresholded_dice/val", best_thresholded_dice, epoch
        )
        writer.add_scalar(
            "Metrics/thresholded_dice_threshold/val", best_threshold, epoch
        )
        print(
            f"Validation thresholded dice: {best_thresholded_dice} "
            f"at threshold {best_threshold}"
        )
        print(f"Train loss: {train_loss}, validation loss: {val_loss}")

    torch.save(model.state_dict(), f"models/{run_name}_final.pkl")

    return train_loss, val_loss, best_threshold


def train_k_fold(
    model_fn: Callable[[], torch.nn.Module],
    optimizer_fn: Callable[[object], torch.optim.Optimizer],
    loss_fn: Callable[..., torch.Tensor],
    epochs: int,
    run_name: str,
    metrics: dict[str, Callable[..., torch.Tensor]] = {},
    batch_size: int = 64,
    augment_train: bool = True,
    threshold_search_points: int = 101,
) -> tuple[float, float]:
    """
    Train a given model for all k folds.
    Returns average train and validation loss across folds.
    """
    print(f"Training {run_name}_BS{batch_size}...")
    folds, _ = get_dataset_folds()
    total_train_loss = 0.0
    total_val_loss = 0.0
    for i, fold in enumerate(folds):
        train_ds = BRATSDataset(fold[0], augmented=augment_train)
        val_ds = BRATSDataset(fold[1])

        train_dl = DataLoader(
            train_ds, batch_size=batch_size, num_workers=8, shuffle=True
        )
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=8)

        print(f"Training fold {i + 1}/{len(folds)}")
        model = model_fn()
        optimizer = optimizer_fn(model.parameters())

        train_loss, val_loss, best_threshold = train_model(
            model,
            train_dl,
            val_dl,
            loss_fn,
            optimizer,
            epochs,
            metrics=metrics,
            run_name=f"{run_name}_BS{batch_size}_FOLD{i + 1}",
            threshold_search_points=threshold_search_points,
        )
        total_train_loss += train_loss
        total_val_loss += val_loss

        with torch.no_grad():
            val_probs, val_targets = collect_validation_outputs(
                val_dl, model, _get_device()
            )
            binary_probs = (val_probs >= best_threshold).to(val_targets.dtype)
            fold_precision = float(
                calculate_precision(binary_probs, val_targets).item()
            )
            fold_recall = float(
                calculate_recall(binary_probs, val_targets).item()
            )

        print(
            f"Fold {i + 1} - Precision: {fold_precision:.4f}, "
            f"Recall: {fold_recall:.4f}, Threshold: {best_threshold:.2f}"
        )

    n = len(folds) if len(folds) > 0 else 1
    return total_train_loss / n, total_val_loss / n
