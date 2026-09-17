"""Digits dataset preparation for the domain-agnostic benchmark track."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from edge_bench.config import PROJECT_ROOT
from edge_bench.data.base import DatasetBundle


def prepare_digits_dataset(config: dict) -> DatasetBundle:
    """Load or build a cached version of the sklearn digits dataset.

    Digits is intentionally the default because it is legal, tiny, and has zero
    credential friction. That makes it ideal for quick experiments and for
    showing that the benchmarking code works before moving to sensor data.
    """

    cache_path = Path(PROJECT_ROOT / "data" / "cache" / "digits_processed.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return DatasetBundle(
            name="digits",
            track="domain_agnostic",
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
            raw_dataset_path="sklearn.datasets.load_digits",
            cache_path=str(cache_path),
        )

    raw = load_digits()
    X = raw.images.astype(np.float32)[:, None, :, :]
    y = raw.target.astype(np.int64)
    class_names = [str(name) for name in raw.target_names]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=float(config["dataset"]["test_size"]),
        random_state=int(config["experiment"]["seed"]),
        stratify=y,
    )

    val_ratio = float(config["dataset"]["val_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=int(config["experiment"]["seed"]),
        stratify=y_train_val,
    )

    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6

    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=np.asarray(class_names, dtype=object),
        input_shape=np.asarray(X_train.shape[1:], dtype=np.int64),
        num_classes=np.asarray(len(class_names), dtype=np.int64),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sample_input_bytes=np.asarray(int(np.prod(X_train.shape[1:]) * 4), dtype=np.int64),
    )

    return DatasetBundle(
        name="digits",
        track="domain_agnostic",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        input_shape=tuple(int(v) for v in X_train.shape[1:]),
        num_classes=len(class_names),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sample_input_bytes=int(np.prod(X_train.shape[1:]) * 4),
        raw_dataset_path="sklearn.datasets.load_digits",
        cache_path=str(cache_path),
    )
