from __future__ import annotations

from pathlib import Path

from edge_bench.config import DEFAULT_CONFIG, create_run_dir, load_config, recursive_merge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_quick.yaml"


def test_recursive_merge_keeps_defaults_and_overrides_nested_keys() -> None:
    override = {
        "experiment": {"name": "custom_run"},
        "benchmark": {"primary_batch_size": 32},
    }
    merged = recursive_merge(DEFAULT_CONFIG, override)
    assert merged["experiment"]["name"] == "custom_run"
    assert merged["benchmark"]["primary_batch_size"] == 32
    assert merged["training"]["epochs"] == DEFAULT_CONFIG["training"]["epochs"]


def test_create_run_dir_uses_fresh_timestamp_for_each_run(tmp_path: Path) -> None:
    config = load_config(BASELINE_CONFIG)
    config["paths"]["results_root"] = str(tmp_path)
    config["experiment"]["name"] = "phase1_test"

    first = create_run_dir(config)
    second = create_run_dir(config)

    assert first != second
    assert first.exists()
    assert second.exists()
