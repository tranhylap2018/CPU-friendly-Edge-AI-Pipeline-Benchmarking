"""Metrics helpers shared across training, benchmarking, and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def latency_stats(values_ms: Iterable[float]) -> dict[str, float]:
    """Summarize latency distributions using tail-sensitive percentiles."""

    array = np.asarray(list(values_ms), dtype=np.float64)
    if array.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Return task metrics that are useful for model and pipeline tradeoffs."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
        "class_names": class_names,
    }


def count_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def file_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024))


def rss_memory_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def efficiency_score(accuracy: float, p95_ms: float, model_size_mb: float) -> float:
    """A simple derived score to encourage discussing system tradeoffs.

    Higher is better. It rewards accuracy while penalizing tail latency and
    storage cost, two practical constraints for edge deployment.
    """

    return float(accuracy / max(p95_ms * max(model_size_mb, 0.01), 1e-6))


def latency_accuracy_ratio(mean_latency_ms: float, accuracy: float) -> float:
    return float(mean_latency_ms / max(accuracy, 1e-6))
