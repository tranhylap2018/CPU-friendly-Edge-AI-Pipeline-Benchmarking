"""Dataset abstractions used by both benchmark tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A tiny wrapper around NumPy arrays so the DataLoader stays consistent."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.tensor(self.labels[index])


@dataclass(slots=True)
class DatasetBundle:
    """All metadata and arrays needed to reproduce an experiment."""

    name: str
    track: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    input_shape: tuple[int, ...]
    num_classes: int
    mean: np.ndarray
    std: np.ndarray
    sample_input_bytes: int
    raw_dataset_path: str
    cache_path: str

    def arrays_for_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        mapping = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
        }
        try:
            return mapping[split]
        except KeyError as exc:
            raise ValueError(f"Unknown split '{split}'.") from exc

    def make_loader(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        seed: int,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create deterministic loaders so latency numbers stay comparable."""

        features, labels = self.arrays_for_split(split)
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset = ArrayDataset(features=features, labels=labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
            pin_memory=False,
            drop_last=False,
        )

    def preprocess_batch(self, batch_x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Normalize the batch on the fly so preprocessing is measurable."""

        mean_tensor = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        std_tensor = torch.as_tensor(self.std, dtype=torch.float32, device=device)
        return (batch_x.to(device=device, dtype=torch.float32) - mean_tensor) / std_tensor

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "track": self.track,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "sample_input_bytes": self.sample_input_bytes,
            "raw_dataset_path": self.raw_dataset_path,
            "cache_path": self.cache_path,
        }
