"""UCI HAR dataset preparation for the sensor-data benchmark track."""

from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np
from sklearn.model_selection import train_test_split

from edge_bench.config import PROJECT_ROOT
from edge_bench.data.base import DatasetBundle


UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def _download_zip(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_HAR_URL, context=context) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _load_signal_matrix(root: Path, split: str) -> np.ndarray:
    tensors = []
    for signal_name in SIGNAL_NAMES:
        signal_path = root / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
        tensors.append(np.loadtxt(signal_path, dtype=np.float32))
    return np.stack(tensors, axis=1)


def _load_label_vector(root: Path, split: str) -> np.ndarray:
    label_path = root / split / f"y_{split}.txt"
    return np.loadtxt(label_path, dtype=np.int64) - 1


def prepare_uci_har_dataset(config: dict) -> DatasetBundle:
    """Download, cache, and load UCI HAR as a first-class time-series track."""

    raw_root = Path(PROJECT_ROOT / "data" / "raw")
    cache_path = Path(PROJECT_ROOT / "data" / "cache" / "uci_har_processed.npz")
    raw_zip = raw_root / "uci_har_dataset.zip"
    extracted_root = raw_root / "UCI HAR Dataset"

    if not cache_path.exists():
        if not raw_zip.exists():
            _download_zip(raw_zip)
        if not extracted_root.exists():
            with zipfile.ZipFile(raw_zip, "r") as archive:
                archive.extractall(raw_root)
        nested_zip = raw_root / "UCI HAR Dataset.zip"
        if nested_zip.exists() and not extracted_root.exists():
            with zipfile.ZipFile(nested_zip, "r") as archive:
                archive.extractall(raw_root)

        train_features = _load_signal_matrix(extracted_root, "train")
        test_features = _load_signal_matrix(extracted_root, "test")
        train_labels = _load_label_vector(extracted_root, "train")
        test_labels = _load_label_vector(extracted_root, "test")

        X_train, X_val, y_train, y_val = train_test_split(
            train_features,
            train_labels,
            test_size=float(config["dataset"]["val_split"]),
            random_state=int(config["experiment"]["seed"]),
            stratify=train_labels,
        )

        activity_names = []
        with (extracted_root / "activity_labels.txt").open("r", encoding="utf-8") as handle:
            for line in handle:
                _, label = line.strip().split(maxsplit=1)
                activity_names.append(label.lower().replace("_", " "))

        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

        np.savez_compressed(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=test_features,
            y_test=test_labels,
            class_names=np.asarray(activity_names, dtype=object),
            input_shape=np.asarray(X_train.shape[1:], dtype=np.int64),
            num_classes=np.asarray(len(activity_names), dtype=np.int64),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            sample_input_bytes=np.asarray(int(np.prod(X_train.shape[1:]) * 4), dtype=np.int64),
            raw_dataset_path=np.asarray(str(extracted_root), dtype=object),
        )

    cached = np.load(cache_path, allow_pickle=True)
    return DatasetBundle(
        name="uci_har",
        track="time_series",
        X_train=cached["X_train"],
        y_train=cached["y_train"],
        X_val=cached["X_val"],
        y_val=cached["y_val"],
        X_test=cached["X_test"],
        y_test=cached["y_test"],
        class_names=[str(name) for name in cached["class_names"].tolist()],
        input_shape=tuple(int(v) for v in cached["input_shape"]),
        num_classes=int(cached["num_classes"]),
        mean=cached["mean"],
        std=cached["std"],
        sample_input_bytes=int(cached["sample_input_bytes"]),
        raw_dataset_path=str(cached["raw_dataset_path"].item()),
        cache_path=str(cache_path),
    )
