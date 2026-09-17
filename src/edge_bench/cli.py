"""Command-line interface for the benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from edge_bench.config import create_run_dir, load_config, save_resolved_config
from edge_bench.data import prepare_dataset
from edge_bench.pipeline.analysis import generate_analysis_artifacts
from edge_bench.pipeline.benchmark import benchmark_all_models
from edge_bench.pipeline.train import train_all_models
from edge_bench.utils.io_utils import append_registry_row, write_json
from edge_bench.utils.logging_utils import configure_logging
from edge_bench.utils.reproducibility import collect_environment_info, set_global_seed


def _latest_run_dir(config: dict[str, Any]) -> Path | None:
    root = Path(config["paths"]["results_root"])
    candidates = sorted(root.glob(f"{config['experiment']['name']}_*"))
    return candidates[-1] if candidates else None


def _ensure_run_dir_layout(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("logs", "metrics", "models", "predictions", "figures", "analysis"):
        (run_dir / relative).mkdir(exist_ok=True)
    return run_dir


def _resolve_existing_run_dir(config: dict[str, Any], requested_run_dir: str | None) -> Path:
    if requested_run_dir:
        return _ensure_run_dir_layout(Path(requested_run_dir).expanduser().resolve())

    latest = _latest_run_dir(config)
    if latest is None:
        raise FileNotFoundError(
            f"No existing run found for experiment '{config['experiment']['name']}'."
        )
    return _ensure_run_dir_layout(latest)


def _resolve_run_dir_for_training(config: dict[str, Any], requested_run_dir: str | None) -> Path:
    """Training starts a fresh run unless the user explicitly chooses a run dir."""

    if requested_run_dir:
        return _ensure_run_dir_layout(Path(requested_run_dir).expanduser().resolve())
    return create_run_dir(config)


def _resolve_run_dir_for_benchmark(config: dict[str, Any], requested_run_dir: str | None) -> Path:
    """Benchmark prefers the latest existing run so train -> benchmark feels natural."""

    if requested_run_dir:
        return _ensure_run_dir_layout(Path(requested_run_dir).expanduser().resolve())
    latest = _latest_run_dir(config)
    if latest is not None:
        return _ensure_run_dir_layout(latest)
    return create_run_dir(config)


def _bootstrap_run(config: dict[str, Any], run_dir: Path) -> Any:
    log_path = run_dir / "logs" / "run.log"
    logger = configure_logging(log_path)
    set_global_seed(
        seed=int(config["experiment"]["seed"]),
        deterministic=bool(config["experiment"]["deterministic"]),
    )
    torch.set_num_threads(max(1, torch.get_num_threads()))
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_json(run_dir / "environment.json", collect_environment_info())
    return logger


def _register_run(config: dict[str, Any], run_dir: Path) -> None:
    append_registry_row(
        Path(config["paths"]["results_root"]) / "experiment_registry.csv",
        {
            "run_dir": str(run_dir),
            "experiment_name": config["experiment"]["name"],
            "dataset": config["dataset"]["name"],
            "track": config["dataset"]["track"],
            "config_path": config["_meta"]["config_path"],
            "timestamp_utc": config["_meta"]["timestamp_utc"],
        },
    )


def command_prepare_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset = prepare_dataset(config)
    print(f"Prepared dataset '{dataset.name}' at cache: {dataset.cache_path}")


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = _resolve_run_dir_for_training(config, args.run_dir)
    logger = _bootstrap_run(config, run_dir)
    dataset = prepare_dataset(config)
    train_all_models(config, dataset, run_dir, logger)
    _register_run(config, run_dir)
    print(run_dir)


def command_benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = _resolve_run_dir_for_benchmark(config, args.run_dir)
    logger = _bootstrap_run(config, run_dir)
    dataset = prepare_dataset(config)
    if not (run_dir / "metrics" / "training_summary.csv").exists():
        train_all_models(config, dataset, run_dir, logger)
    benchmark_all_models(config, dataset, run_dir, logger)
    _register_run(config, run_dir)
    print(run_dir)


def command_analyze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = _resolve_existing_run_dir(config, args.run_dir)
    logger = _bootstrap_run(config, run_dir)
    dataset = prepare_dataset(config)
    generate_analysis_artifacts(config, dataset, run_dir, logger)
    print(run_dir)


def command_make_report_assets(args: argparse.Namespace) -> None:
    command_analyze(args)


def command_run_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = create_run_dir(config)
    logger = _bootstrap_run(config, run_dir)
    dataset = prepare_dataset(config)
    train_all_models(config, dataset, run_dir, logger)
    benchmark_all_models(config, dataset, run_dir, logger)
    generate_analysis_artifacts(config, dataset, run_dir, logger)
    _register_run(config, run_dir)
    print(run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-friendly edge AI pipeline benchmarking suite.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", required=True, help="Path to YAML experiment config.")
        subparser.add_argument("--run-dir", default=None, help="Optional existing run directory.")

    prepare_data = subparsers.add_parser("prepare-data", help="Download/cache datasets.")
    prepare_data.add_argument("--config", required=True, help="Path to YAML experiment config.")
    prepare_data.set_defaults(func=command_prepare_data)

    train = subparsers.add_parser("train", help="Train models only.")
    add_shared_arguments(train)
    train.set_defaults(func=command_train)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark models and save metrics.")
    add_shared_arguments(benchmark)
    benchmark.set_defaults(func=command_benchmark)

    analyze = subparsers.add_parser("analyze", help="Regenerate plots and summary tables.")
    add_shared_arguments(analyze)
    analyze.set_defaults(func=command_analyze)

    report = subparsers.add_parser("make-report-assets", help="Build poster/report assets.")
    add_shared_arguments(report)
    report.set_defaults(func=command_make_report_assets)

    run_all = subparsers.add_parser("run-all", help="Execute prepare, train, benchmark, and analyze.")
    run_all.add_argument("--config", required=True, help="Path to YAML experiment config.")
    run_all.set_defaults(func=command_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
