"""Reproducibility helpers.

These functions are deliberately explicit because reproducibility is part of the
project's educational value, not an afterthought.
"""

from __future__ import annotations

import importlib.metadata
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_environment_info() -> dict[str, Any]:
    """Capture enough environment metadata to reproduce a run later."""

    tracked_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "psutil",
        "PyYAML",
        "scikit-learn",
        "torch",
        "tqdm",
    ]
    packages: dict[str, str] = {}
    for package in tracked_packages:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "packages": packages,
        "torch_num_threads": torch.get_num_threads(),
    }
