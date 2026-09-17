"""Small CPU-friendly model zoo for both benchmark tracks."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from edge_bench.data.base import DatasetBundle


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)


class DigitsLinear(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(Flatten(), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitsMLP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitsCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitsCNNPlus(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARLinear(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(Flatten(), nn.Linear(9 * 128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARMLP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(9 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HARGRU(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=9,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


MODEL_REGISTRY: dict[str, dict[str, tuple[str, Callable[[int], nn.Module]]]] = {
    "digits": {
        "linear": ("Linear baseline", DigitsLinear),
        "mlp": ("Small MLP", DigitsMLP),
        "cnn": ("Tiny CNN", DigitsCNN),
        "cnn_plus": ("Wider CNN", DigitsCNNPlus),
    },
    "uci_har": {
        "linear": ("Linear baseline", HARLinear),
        "mlp": ("Small MLP", HARMLP),
        "cnn": ("1D CNN", HARCNN),
        "gru": ("Small GRU", HARGRU),
    },
}


def available_models_for_dataset(dataset_name: str) -> dict[str, str]:
    return {name: label for name, (label, _) in MODEL_REGISTRY[dataset_name].items()}


def build_model(model_name: str, dataset: DatasetBundle) -> nn.Module:
    try:
        _, builder = MODEL_REGISTRY[dataset.name][model_name]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY[dataset.name])
        raise ValueError(
            f"Model '{model_name}' is not available for dataset '{dataset.name}'. "
            f"Available models: {available}"
        ) from exc
    return builder(dataset.num_classes)
