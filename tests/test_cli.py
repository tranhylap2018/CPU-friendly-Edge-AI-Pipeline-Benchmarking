from __future__ import annotations

from pathlib import Path

from edge_bench.cli import (
    _resolve_existing_run_dir,
    _resolve_run_dir_for_benchmark,
    _resolve_run_dir_for_training,
    build_parser,
)
from edge_bench.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_quick.yaml"


def test_cli_exposes_expected_top_level_commands() -> None:
    parser = build_parser()
    command_names = set()
    for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
        if hasattr(action, "choices"):
            command_names.update(action.choices.keys())
    assert {
        "prepare-data",
        "train",
        "benchmark",
        "analyze",
        "make-report-assets",
        "run-all",
    }.issubset(command_names)


def test_cli_parses_prepare_data_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["prepare-data", "--config", "configs/baseline_quick.yaml"])
    assert args.command == "prepare-data"
    assert args.config == "configs/baseline_quick.yaml"


def test_benchmark_reuses_latest_existing_run_by_default(tmp_path: Path) -> None:
    config = load_config(BASELINE_CONFIG)
    config["paths"]["results_root"] = str(tmp_path)
    config["experiment"]["name"] = "phase1_cli"

    train_run = _resolve_run_dir_for_training(config, None)
    benchmark_run = _resolve_run_dir_for_benchmark(config, None)

    assert benchmark_run == train_run


def test_analyze_uses_latest_existing_run(tmp_path: Path) -> None:
    config = load_config(BASELINE_CONFIG)
    config["paths"]["results_root"] = str(tmp_path)
    config["experiment"]["name"] = "phase1_cli_existing"

    created_run = _resolve_run_dir_for_training(config, None)
    resolved_run = _resolve_existing_run_dir(config, None)

    assert resolved_run == created_run
