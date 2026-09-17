"""End-to-end pipeline benchmarking.

This module measures where time is spent across the pipeline, not only inside
the model. That framing is the key systems lesson of the project.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from edge_bench.data.base import DatasetBundle
from edge_bench.models.factory import available_models_for_dataset, build_model
from edge_bench.pipeline.offload import simulate_offload_policies
from edge_bench.utils.io_utils import write_json, write_rows_csv
from edge_bench.utils.metrics import (
    classification_summary,
    count_parameters,
    efficiency_score,
    file_size_mb,
    latency_accuracy_ratio,
    latency_stats,
    rss_memory_mb,
)


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    dataset: DatasetBundle,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(model_name=model_name, dataset=dataset).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _warmup_model(
    model: torch.nn.Module,
    dataset: DatasetBundle,
    split: str,
    batch_size: int,
    warmup_batches: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> None:
    loader = dataset.make_loader(
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    iterator = iter(loader)
    with torch.inference_mode():
        for _ in range(warmup_batches):
            try:
                raw_x, _ = next(iterator)
            except StopIteration:
                break
            x = dataset.preprocess_batch(raw_x, device=device)
            _ = model(x)


def _benchmark_single_model(
    model: torch.nn.Module,
    model_name: str,
    model_label: str,
    dataset: DatasetBundle,
    config: dict[str, Any],
    batch_size: int,
    checkpoint_path: Path,
    training_time_s: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(config["experiment"]["device"])
    split = str(config["benchmark"]["split"])
    repetitions = int(config["benchmark"]["repetitions"])
    warmup_batches = int(config["benchmark"]["warmup_batches"])
    num_workers = int(config["experiment"]["num_workers"])
    seed = int(config["experiment"]["seed"])

    _warmup_model(
        model=model,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )

    data_loading_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    total_batch_ms: list[float] = []
    total_sample_ms: list[float] = []
    sample_load_ms: list[float] = []

    collected_true: list[np.ndarray] = []
    collected_pred: list[np.ndarray] = []
    collected_conf: list[np.ndarray] = []
    sample_pre_ms: list[float] = []
    sample_infer_ms: list[float] = []
    sample_post_ms: list[float] = []

    baseline_rss_mb = rss_memory_mb()
    peak_rss_mb = baseline_rss_mb
    measured_samples = 0

    with torch.inference_mode():
        for repetition in range(repetitions):
            loader = dataset.make_loader(
                split=split,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                seed=seed + repetition + batch_size,
            )
            iterator = iter(loader)
            while True:
                load_start = time.perf_counter()
                try:
                    raw_x, raw_y = next(iterator)
                except StopIteration:
                    break
                load_elapsed_ms = (time.perf_counter() - load_start) * 1000.0

                preprocess_start = time.perf_counter()
                x = dataset.preprocess_batch(raw_x, device=device)
                y = raw_y.to(device=device, dtype=torch.long)
                preprocess_elapsed_ms = (time.perf_counter() - preprocess_start) * 1000.0

                inference_start = time.perf_counter()
                logits = model(x)
                inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000.0

                post_start = time.perf_counter()
                probabilities = torch.softmax(logits, dim=1)
                confidence, predictions = torch.max(probabilities, dim=1)
                post_elapsed_ms = (time.perf_counter() - post_start) * 1000.0

                batch_total_ms = (
                    load_elapsed_ms
                    + preprocess_elapsed_ms
                    + inference_elapsed_ms
                    + post_elapsed_ms
                )
                batch_item_count = int(raw_y.shape[0])
                per_sample_total_ms = batch_total_ms / batch_item_count
                per_sample_load_ms = load_elapsed_ms / batch_item_count
                per_sample_pre_ms = preprocess_elapsed_ms / batch_item_count
                per_sample_infer_ms = inference_elapsed_ms / batch_item_count
                per_sample_post_ms = post_elapsed_ms / batch_item_count

                data_loading_ms.append(load_elapsed_ms)
                preprocess_ms.append(preprocess_elapsed_ms)
                inference_ms.append(inference_elapsed_ms)
                postprocess_ms.append(post_elapsed_ms)
                total_batch_ms.append(batch_total_ms)
                total_sample_ms.extend([per_sample_total_ms] * batch_item_count)
                sample_load_ms.extend([per_sample_load_ms] * batch_item_count)
                measured_samples += batch_item_count
                peak_rss_mb = max(peak_rss_mb, rss_memory_mb())

                if repetition == 0:
                    collected_true.append(y.cpu().numpy())
                    collected_pred.append(predictions.cpu().numpy())
                    collected_conf.append(confidence.cpu().numpy())
                    sample_pre_ms.extend([per_sample_pre_ms] * batch_item_count)
                    sample_infer_ms.extend([per_sample_infer_ms] * batch_item_count)
                    sample_post_ms.extend([per_sample_post_ms] * batch_item_count)

    y_true = np.concatenate(collected_true)
    y_pred = np.concatenate(collected_pred)
    confidence = np.concatenate(collected_conf)
    classification = classification_summary(y_true=y_true, y_pred=y_pred, class_names=dataset.class_names)
    batch_latency = latency_stats(total_batch_ms)
    sample_latency = latency_stats(total_sample_ms)
    mean_load_sample_ms = float(np.mean(sample_load_ms))
    mean_pre_sample_ms = float(np.mean(sample_pre_ms))
    mean_infer_sample_ms = float(np.mean(sample_infer_ms))
    mean_post_sample_ms = float(np.mean(sample_post_ms))
    throughput = float(measured_samples / max(sum(total_batch_ms) / 1000.0, 1e-6))
    model_size = file_size_mb(checkpoint_path)
    parameter_count = count_parameters(model)

    summary_row = {
        "dataset": dataset.name,
        "track": dataset.track,
        "model": model_name,
        "model_label": model_label,
        "batch_size": batch_size,
        "training_time_s": training_time_s,
        "accuracy": float(classification["accuracy"]),
        "f1_macro": float(classification["f1_macro"]),
        "parameter_count": parameter_count,
        "serialized_model_size_mb": model_size,
        "mean_batch_latency_ms": float(batch_latency["mean_ms"]),
        "p50_batch_latency_ms": float(batch_latency["p50_ms"]),
        "p95_batch_latency_ms": float(batch_latency["p95_ms"]),
        "p99_batch_latency_ms": float(batch_latency["p99_ms"]),
        "mean_sample_latency_ms": float(sample_latency["mean_ms"]),
        "p50_sample_latency_ms": float(sample_latency["p50_ms"]),
        "p95_sample_latency_ms": float(sample_latency["p95_ms"]),
        "p99_sample_latency_ms": float(sample_latency["p99_ms"]),
        "mean_data_loading_ms": mean_load_sample_ms,
        "mean_preprocessing_ms": mean_pre_sample_ms,
        "mean_inference_ms": mean_infer_sample_ms,
        "mean_postprocessing_ms": mean_post_sample_ms,
        "mean_end_to_end_ms": float(sample_latency["mean_ms"]),
        "throughput_samples_per_s": throughput,
        "peak_rss_delta_mb": float(max(0.0, peak_rss_mb - baseline_rss_mb)),
        "latency_accuracy_ratio": latency_accuracy_ratio(
            mean_latency_ms=float(sample_latency["mean_ms"]),
            accuracy=float(classification["accuracy"]),
        ),
        "efficiency_score": efficiency_score(
            accuracy=float(classification["accuracy"]),
            p95_ms=float(sample_latency["p95_ms"]),
            model_size_mb=model_size,
        ),
    }

    prediction_artifact = {
        "y_true": y_true.astype(np.int64),
        "y_pred": y_pred.astype(np.int64),
        "confidence": confidence.astype(np.float32),
        "total_latency_ms": np.asarray(total_sample_ms[: len(y_true)], dtype=np.float32),
        "load_latency_ms": np.asarray(sample_load_ms[: len(y_true)], dtype=np.float32),
        "preprocess_latency_ms": np.asarray(sample_pre_ms, dtype=np.float32),
        "inference_latency_ms": np.asarray(sample_infer_ms, dtype=np.float32),
        "postprocess_latency_ms": np.asarray(sample_post_ms, dtype=np.float32),
        "confusion_matrix": np.asarray(classification["confusion_matrix"], dtype=np.int64),
    }
    return summary_row, prediction_artifact


def benchmark_all_models(
    config: dict[str, Any],
    dataset: DatasetBundle,
    run_dir: Path,
    logger: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark every trained model and save summary tables + predictions."""

    device = torch.device(config["experiment"]["device"])
    available = available_models_for_dataset(dataset.name)
    training_summary = pd.read_csv(run_dir / "metrics" / "training_summary.csv")
    training_lookup = training_summary.set_index("model").to_dict(orient="index")

    summary_rows: list[dict[str, Any]] = []
    primary_prediction_artifacts: dict[str, dict[str, np.ndarray]] = {}
    primary_batch_size = int(config["benchmark"]["primary_batch_size"])

    for model_name in config["models"]["names"]:
        model_label = available[model_name]
        checkpoint_path = run_dir / "models" / f"{model_name}.pt"
        model = _load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            dataset=dataset,
            device=device,
        )
        for batch_size in config["benchmark"]["batch_sizes"]:
            summary_row, artifact = _benchmark_single_model(
                model=model,
                model_name=model_name,
                model_label=model_label,
                dataset=dataset,
                config=config,
                batch_size=int(batch_size),
                checkpoint_path=checkpoint_path,
                training_time_s=float(training_lookup[model_name]["training_time_s"]),
            )
            summary_rows.append(summary_row)
            if int(batch_size) == primary_batch_size:
                primary_prediction_artifacts[model_name] = artifact
                np.savez_compressed(
                    run_dir / "predictions" / f"{model_name}.npz",
                    **artifact,
                )
            logger.info(
                "model=%s event=benchmark_complete batch_size=%s p95_sample_ms=%.4f accuracy=%.4f",
                model_name,
                batch_size,
                summary_row["p95_sample_latency_ms"],
                summary_row["accuracy"],
            )

    benchmark_summary = pd.DataFrame(summary_rows)
    benchmark_summary.to_csv(run_dir / "metrics" / "benchmark_summary.csv", index=False)
    write_rows_csv(run_dir / "metrics" / "benchmark_summary_export.csv", summary_rows)
    write_json(run_dir / "metrics" / "benchmark_summary.json", summary_rows)

    offload_rows = simulate_offload_policies(
        config=config,
        dataset=dataset,
        prediction_artifacts=primary_prediction_artifacts,
    )
    offload_columns = [
        "policy",
        "accuracy",
        "f1_macro",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "throughput_samples_per_s",
        "offloaded_fraction",
        "bandwidth_kb_per_sample",
        "edge_model",
        "cloud_model",
        "hybrid_secondary_model",
    ]
    offload_summary = pd.DataFrame(offload_rows, columns=offload_columns)
    offload_summary.to_csv(run_dir / "metrics" / "offload_summary.csv", index=False)
    write_json(run_dir / "metrics" / "offload_summary.json", offload_rows)
    return benchmark_summary, offload_summary
