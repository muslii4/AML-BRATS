from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

DATA_PATH = Path("data/BraTS2020_training_data")


def _process_path(path_str: str) -> Path:
    """Convert the Kaggle CSV path to the local project path."""
    path = Path(path_str)
    filename = path.name
    return DATA_PATH / "content/data" / filename


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the metadata and processess all paths."""
    metadata = pd.read_csv(path)
    metadata["slice_path"] = metadata["slice_path"].map(_process_path)
    return metadata


def split_by_volume(
    metadata: pd.DataFrame,
    seed: int = 42,
    train: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the metadata by volume (patient ID)."""
    volumes = metadata["volume"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(volumes)

    n_total = len(volumes)
    n_train = int(n_total * train)

    train_volumes = volumes[:n_train]
    test_volumes = volumes[n_train:]

    train_data = metadata[metadata["volume"].isin(train_volumes)]
    test_data = metadata[metadata["volume"].isin(test_volumes)]

    return train_data, test_data


class BRATSDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        augmented: bool = False,
        base_path: Path = Path("."),
    ) -> None:
        self.metadata = metadata
        self.augmented = augmented
        self.base_path = base_path

        self.train_transform = A.Compose(
            [
                A.Rotate(limit=5, border_mode=0, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.02,
                    border_mode=0,
                    rotate_limit=0,
                    p=0.3,
                ),
                A.GaussNoise(std_range=(0.01, 0.02), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path = self.metadata.iloc[idx]["slice_path"]

        with h5py.File(self.base_path / path, "r") as f:
            image: np.ndarray = f["image"][()]
            mask: np.ndarray = f["mask"][()]

        image = image.astype(np.float32)

        brain_mask = np.any(image > 0, axis=-1)

        for channel in range(image.shape[-1]):
            channel_data = image[:, :, channel]

            brain_pixels = channel_data[brain_mask]

            if len(brain_pixels) > 0:
                mean = brain_pixels.mean()
                std = brain_pixels.std()

                if std > 0:
                    channel_data[brain_mask] = (
                        channel_data[brain_mask] - mean
                    ) / std

            channel_data[~brain_mask] = 0

            image[:, :, channel] = channel_data

        if self.augmented:
            augmented_version = self.train_transform(image=image, mask=mask)
            image = augmented_version["image"]
            mask = augmented_version["mask"]

        image = np.moveaxis(image, -1, 0).astype(np.float32)
        mask = np.moveaxis(mask, -1, 0).astype(np.float32)

        return {
            "image": image,
            "mask": mask,
        }


def make_cv_splits(
    train_metadata: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    # Splitting this with k cross-validation. (by patient)

    # to get all uniqur patient IDs from the training metadata
    volumes = train_metadata["volume"].unique()

    # random number generator so that we can have the same split every time we run
    rng = np.random.default_rng(seed)
    # to shuffle the patient ID
    rng.shuffle(volumes)

    # group the patients by splitting the shuffled ids
    volume_folds = np.array_split(volumes, k)

    cv_splits = []

    # go through each fold and use the current as a validation set while the rest are just training
    for i in range(k):
        val_volumes = volume_folds[i]

        train_volumes = np.concatenate(
            [volume_folds[j] for j in range(k) if j != i]
        )
        # training folds
        fold_train = train_metadata[
            train_metadata["volume"].isin(train_volumes)
        ]
        # validation folds
        fold_val = train_metadata[train_metadata["volume"].isin(val_volumes)]

        # save
        cv_splits.append((fold_train, fold_val))

    return cv_splits


def get_dataset_folds(
    metadata_dir: Path | str = DATA_PATH / "content/data/meta_data.csv",
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    metadata = load_metadata(Path(metadata_dir))
    train_metadata, test_metadata = split_by_volume(metadata)
    cv_splits = make_cv_splits(train_metadata, k=5)
    return cv_splits, test_metadata


if __name__ == "__main__":
    metadata = load_metadata(DATA_PATH / "content/data/meta_data.csv")
    train_metadata, test_metadata = split_by_volume(metadata)
    cv_splits = make_cv_splits(train_metadata, k=5)
    test_ds = BRATSDataset(test_metadata)
    fold_number = 1

    for fold_train_metadata, fold_val_metadata in cv_splits:
        print(f"Fold {fold_number}")
        fold_train_ds = BRATSDataset(fold_train_metadata, augmented=True)
        fold_val_ds = BRATSDataset(fold_val_metadata)
        print("Train size:", len(fold_train_ds))
        print("Validation size:", len(fold_val_ds))
        print()
        fold_number += 1

    print("Test size:", len(test_ds))
    print("One test example:")

    sample = test_ds[50]

    print("Image shape:", sample["image"].shape)
    print("Mask shape:", sample["mask"].shape)

    print("Image min:", sample["image"].min())
    print("Image max:", sample["image"].max())
    print(
        "Number of non-zero image pixels:", np.count_nonzero(sample["image"])
    )

    print("Mask unique values:", np.unique(sample["mask"]))
