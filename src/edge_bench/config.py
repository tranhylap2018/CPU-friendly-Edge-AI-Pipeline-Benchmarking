"""Configuration loading and run-directory helpers.

The project is intentionally config-driven because that is how we make the
benchmarking workflow reproducible. A saved YAML file is much easier to discuss
with faculty than a notebook cell with hidden state.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "baseline_quick",
        "seed": 42,
        "deterministic": True,
        "device": "cpu",
        "num_workers": 0,
        "notes": "",
    },
    "dataset": {
        "name": "digits",
        "track": "domain_agnostic",
        "auto_download": True,
        "cache_processed": True,
        "val_split": 0.15,
        "test_size": 0.2,
    },
    "models": {
        "names": ["linear", "mlp", "cnn"],
    },
    "training": {
        "epochs": 6,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "early_stopping_patience": 3,
    },
    "benchmark": {
        "split": "test",
        "batch_sizes": [1, 16, 64],
        "primary_batch_size": 16,
        "warmup_batches": 2,
        "repetitions": 3,
        "track_memory": True,
    },
    "offload": {
        "enabled": True,
        "edge_model": "linear",
        "cloud_model": "cnn",
        "hybrid_secondary_model": None,
        "confidence_threshold": 0.75,
        "network": {
            "uplink_mbps": 12.0,
            "downlink_mbps": 40.0,
            "round_trip_ms": 35.0,
            "server_processing_ms": 15.0,
            "response_bytes": 128,
            "use_measured_cloud_inference": False,
        },
    },
    "analysis": {
        "save_confusion_matrix": True,
        "poster_title": "CPU-Friendly Edge AI Pipeline Benchmarking",
        "motivation": "Benchmark full edge AI pipelines under CPU and latency constraints.",
        "finding_count": 4,
    },
}


class ConfigError(ValueError):
    """Raised when a config file is malformed or inconsistent."""


def recursive_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating the inputs."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Config at {path} must contain a YAML mapping at the top level."
        raise ConfigError(msg)
    return data


def _validate_config(config: dict[str, Any]) -> None:
    dataset_name = config["dataset"]["name"]
    if dataset_name not in {"digits", "uci_har"}:
        raise ConfigError(f"Unsupported dataset '{dataset_name}'.")

    track = config["dataset"]["track"]
    if track not in {"domain_agnostic", "time_series"}:
        raise ConfigError(f"Unsupported dataset track '{track}'.")

    if config["experiment"]["device"] != "cpu":
        raise ConfigError("This project intentionally targets CPU-only experiments.")

    if not config["models"]["names"]:
        raise ConfigError("At least one model must be listed in models.names.")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config, merge defaults, and resolve project-relative paths."""

    path = Path(config_path).expanduser().resolve()
    raw = _read_yaml(path)
    config = recursive_merge(DEFAULT_CONFIG, raw)
    config["_meta"] = {
        "config_path": str(path),
        "project_root": str(PROJECT_ROOT),
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
    }
    config["paths"] = {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(PROJECT_ROOT / "data"),
        "results_root": str(PROJECT_ROOT / "results"),
        "figures_root": str(PROJECT_ROOT / "figures"),
        "reports_root": str(PROJECT_ROOT / "reports"),
    }
    _validate_config(config)
    return config


def make_timestamp() -> str:
    """Return a fresh timestamp for naming experiment runs.

    Run-directory timestamps are generated at creation time rather than config
    load time so that sequential CLI steps do not accidentally collide.
    """

    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


def create_run_dir(config: dict[str, Any], run_name: str | None = None) -> Path:
    """Create a new run directory under ``results/``."""

    results_root = Path(config["paths"]["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    base_name = run_name or config["experiment"]["name"]
    collision_index = 0
    while True:
        timestamp = make_timestamp()
        suffix = "" if collision_index == 0 else f"_{collision_index}"
        run_dir = results_root / f"{base_name}_{timestamp}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            collision_index += 1

    for relative in ("logs", "metrics", "models", "predictions", "figures", "analysis"):
        (run_dir / relative).mkdir(exist_ok=True)
    return run_dir


def save_resolved_config(config: dict[str, Any], destination: Path) -> None:
    """Persist the fully resolved config alongside experiment outputs."""

    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
