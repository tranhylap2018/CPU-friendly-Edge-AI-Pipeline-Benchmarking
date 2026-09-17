from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase1_directories_exist() -> None:
    expected = [
        "src",
        "configs",
        "data",
        "notebooks",
        "scripts",
        "results",
        "reports",
        "figures",
        "tests",
        "docs",
    ]
    for relative in expected:
        assert (PROJECT_ROOT / relative).exists(), relative


def test_phase1_core_files_exist() -> None:
    expected = [
        "README.md",
        "requirements.txt",
        "main.py",
        "pyproject.toml",
        "configs/baseline_quick.yaml",
        "configs/balanced_benchmark.yaml",
        "configs/edge_cloud_tradeoff.yaml",
        "configs/time_series_full.yaml",
        "docs/architecture.md",
        "docs/implementation_phases.md",
    ]
    for relative in expected:
        assert (PROJECT_ROOT / relative).exists(), relative
