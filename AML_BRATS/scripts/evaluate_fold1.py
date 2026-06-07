from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from AML_BRATS.data.data_loading import BRATSDataset, get_dataset_folds
from AML_BRATS.models.FNC2 import SegNet
from AML_BRATS.models.unet import UNet

THRESHOLD = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the stored model checkpoint.",
    )
    return parser.parse_args()


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0

    mean = float(np.mean(values))

    if len(values) == 1:
        return mean, 0.0

    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, sem


@torch.no_grad()
def evaluate_fold1(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = THRESHOLD,
):
    model.to(device)
    model.eval()

    volumes = defaultdict(int)

    smooth = 1e-5
    precision_values: dict[int, list[float]] = defaultdict(list)
    recall_values: dict[int, list[float]] = defaultdict(list)
    f1_values: dict[int, list[float]] = defaultdict(list)
    dice_values: dict[int, list[float]] = defaultdict(list)

    for datapoint in tqdm(dataloader):
        inputs = datapoint["image"].to(device)
        targets = datapoint["mask"].to(device)
        volume = datapoint["volume"].item()
        volumes[volume] += 1

        probs = model(inputs).sigmoid()
        binary_probs = (probs >= threshold).to(targets.dtype)

        tp = (binary_probs * targets).sum(dim=(2, 3))
        fp = (binary_probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - binary_probs) * targets).sum(dim=(2, 3))

        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        f1 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

        valid_channels = targets.sum(dim=(2, 3)) > 0
        if valid_channels.any():
            valid_precision = precision.masked_select(valid_channels)
            valid_recall = recall.masked_select(valid_channels)
            valid_f1 = f1.masked_select(valid_channels)

            precision_values[volume].extend(valid_precision.cpu().tolist())
            recall_values[volume].extend(valid_recall.cpu().tolist())
            f1_values[volume].extend(valid_f1.cpu().tolist())

            channel_dice = (2 * (probs * targets).sum(dim=(2, 3)) + smooth) / (
                probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
            )

            dice_values[volume].extend(
                channel_dice.masked_select(valid_channels).cpu().tolist()
            )

    patient_precisions = [
        cast(float, np.mean(values)) for values in precision_values.values()
    ]

    patient_recalls = [
        cast(float, np.mean(values)) for values in recall_values.values()
    ]

    patient_f1s = [
        cast(float, np.mean(values)) for values in f1_values.values()
    ]

    patient_dices = [
        cast(float, np.mean(values)) for values in dice_values.values()
    ]

    precision, precision_sem = mean_and_sem(patient_precisions)
    recall, recall_sem = mean_and_sem(patient_recalls)
    f1, f1_sem = mean_and_sem(patient_f1s)
    dice, dice_sem = mean_and_sem(patient_dices)

    return {
        "mean": {
            "precision": precision,
            "precision_sem": precision_sem,
            "recall": recall,
            "recall_sem": recall_sem,
            "f1": f1,
            "f1_sem": f1_sem,
            "dice": dice,
            "dice_sem": dice_sem,
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds, _ = get_dataset_folds()
    fold_train_metadata, fold_val_metadata = folds[0]
    _ = fold_train_metadata

    if "UNET" in args.model_path.name:
        model = UNet(3, "BNORM" in args.model_path.name)
    elif "SegNet" in args.model_path.name:
        model = SegNet(
            3,
            MC_Dropout=True,
            num_passes=1,
            dropout_p=0.2,
        )
    else:
        raise ValueError

    state_dict = torch.load(
        args.model_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)

    validation_ds = BRATSDataset(fold_val_metadata)
    validation_dl = DataLoader(validation_ds)

    metrics = evaluate_fold1(
        model=model,
        dataloader=validation_dl,
        device=device,
    )

    print(
        ",".join(metrics["mean"].keys()),
        ",".join(str(x) for x in metrics["mean"].values()),
        sep="\n",
    )


if __name__ == "__main__":
    main()
