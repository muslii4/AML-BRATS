from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.prune as prune
from tqdm import tqdm
import torch_pruning as tp
from ..data.data_loading import BRATSDataset, get_dataset_folds
import time

def pruner_random(model: torch.nn.Module, amount: float = 0.2) -> torch.nn.Module:
    """ making it smaller but not necessarily faster at inference
    Apply global unstructured pruning to the model by randomly pruning weights.
    this is an iterative process that needs fine tuning after each pruning step 
    """
    model.eval()

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=amount,  
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")  

    zero = sum((p == 0).sum().item() for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Sparsity: {zero/total:.1%}")

    return model 

def pruner_structured(model: torch.nn.Module, input_shape: tuple = (1, 4, 240, 240), amount: float = 0.2) -> tuple[torch.nn.Module, int, int]:
    """ for making it faster at inference 
    Apply structured pruning to the model based on L1 norm.
    this is an iterative process that needs fine tuning after each pruning step
    """
    device = next(model.parameters()).device
    shape = torch.randn(*input_shape).to(device)
    model = model.eval()

    params1 = sum(p.numel() for p in model.parameters())
    print(f"Parameters before pruning: {params1:,}")

    ignored_layers = []
    for module in model.modules():
        for name, module in model.named_modules():
            if name in ["out", "final"]:   
                ignored_layers.append(module)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=shape,
        importance=tp.importance.MagnitudeImportance(p=1),  # L1
        pruning_ratio=amount,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    params2 = sum(p.numel() for p in model.parameters())
    print(f"Parameters after pruning: {params2:,}")
    return model, params1, params2

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
    metrics: dict[str, Callable[..., torch.Tensor]] = None,
) -> tuple[float, dict[str, float]]:
    """Run one validation epoch and return the average loss and computed metrics."""
    model.to(device)
    model.eval()
    val_loss = 0.0
    total_samples = 0

    probs_batches: list[torch.Tensor] = []
    targets_batches: list[torch.Tensor] = []
    volumes: list[str] = []
    collect_outputs = metrics is not None and len(metrics) > 0

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

            if collect_outputs:
                probs_batches.append(y_pred.sigmoid().cpu())
                targets_batches.append(y_true.cpu())
                volumes.extend(datapoint["volume"])

    avg_loss = val_loss / total_samples if total_samples else 0.0

    metric_avgs: dict[str, float] = {}
    if collect_outputs and probs_batches:
        val_probs = torch.cat(probs_batches, dim=0)
        val_targets = torch.cat(targets_batches, dim=0)
        metric_avgs = compute_metrics_from_outputs(
            val_probs, val_targets, metrics, volumes
        )

    return avg_loss, metric_avgs


def compute_metrics_from_outputs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    metrics: dict[str, Callable[..., torch.Tensor]],
    volumes: list[str],
) -> dict[str, float]:
    """Compute all metrics (mean and standard error) from collected probabilities and targets."""
    results: dict[str, float] = {}

    patient_indices: dict[str, list[int]] = {}
    for idx, vol in enumerate(volumes):
        if vol not in patient_indices:
            patient_indices[vol] = [idx]
        else:
            patient_indices[vol].append(idx)

    for metric_name, metric_fn in metrics.items():
        patient_scores = []
        for vol, indices in patient_indices.items():
            indices_t = torch.tensor(indices, dtype=torch.long)
            p_probs = probs[indices_t]
            p_targets = targets[indices_t]

            val = metric_fn(p_probs, p_targets)
            patient_scores.append(val.item())

        scores_tensor = torch.tensor(patient_scores, dtype=torch.float32)
        num_patients = len(patient_scores)
        results[metric_name] = float(scores_tensor.mean().item())
        std_val = (
            float(scores_tensor.std().item()) if num_patients > 1 else 0.0
        )
        results[f"{metric_name}_se"] = std_val / (num_patients**0.5)

    return results


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
    device: torch.device = _get_device(),
) -> tuple[float, float]:
    """
    Train any model with the specified loss function and optimizer.
    Train and validation losses are saved using tensorboard.
    Final loss values are returned, and the final model state is saved.
    """
    writer = SummaryWriter(f"runs/{run_name}")

    train_loss = 0.0
    val_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        val_loss, metric_avgs = validation_epoch(
            validation_dl, model, loss_fn, device, metrics=metrics
        )
        writer.add_scalar("Loss/val", val_loss, epoch)

        if metrics:
            for metric_name, metric_avg in metric_avgs.items():
                writer.add_scalar(
                    f"Metrics/{metric_name}/val", metric_avg, epoch
                )
                print(f"Validation {metric_name}: {metric_avg}")
        print(f"Train loss: {train_loss}, validation loss: {val_loss}")

    writer.close()
    torch.save(model.state_dict(), f"models/{run_name}_final.pkl")

    return train_loss, val_loss


def train_k_fold(
    model_fn: Callable[[], torch.nn.Module],
    optimizer_fn: Callable[[object], torch.optim.Optimizer],
    loss_fn: Callable[..., torch.Tensor],
    epochs: int,
    run_name: str,
    metrics: dict[str, Callable[..., torch.Tensor]] = {},
    batch_size: int = 64,
    augment_train: bool = True,
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

        train_loss, val_loss = train_model(
            model,
            train_dl,
            val_dl,
            loss_fn,
            optimizer,
            epochs,
            metrics=metrics,
            run_name=f"{run_name}_BS{batch_size}_FOLD{i + 1}",
        )
        total_train_loss += train_loss
        total_val_loss += val_loss
    n = len(folds) if len(folds) > 0 else 1
    return total_train_loss / n, total_val_loss / n
