"""Simple edge/cloud routing simulation used for systems tradeoff analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from edge_bench.utils.metrics import classification_summary, latency_stats


def _transfer_ms(payload_bytes: int | float, bandwidth_mbps: float) -> float:
    return float((payload_bytes * 8.0) / (bandwidth_mbps * 1_000_000.0) * 1000.0)


def _cloud_server_ms(network_config: dict[str, Any], cloud_inference_ms: np.ndarray) -> np.ndarray:
    if bool(network_config["use_measured_cloud_inference"]):
        return cloud_inference_ms
    return np.full_like(cloud_inference_ms, float(network_config["server_processing_ms"]))


def simulate_offload_policies(
    config: dict[str, Any],
    dataset: Any,
    prediction_artifacts: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Generate edge/cloud/hybrid comparisons from saved model predictions."""

    if not config["offload"]["enabled"]:
        return []

    network_config = config["offload"]["network"]
    edge_name = config["offload"]["edge_model"]
    cloud_name = config["offload"]["cloud_model"]
    secondary_name = config["offload"].get("hybrid_secondary_model")
    threshold = float(config["offload"]["confidence_threshold"])

    if edge_name not in prediction_artifacts or cloud_name not in prediction_artifacts:
        return []

    edge = prediction_artifacts[edge_name]
    cloud = prediction_artifacts[cloud_name]
    y_true = edge["y_true"]
    upload_ms = _transfer_ms(dataset.sample_input_bytes, float(network_config["uplink_mbps"]))
    download_ms = _transfer_ms(int(network_config["response_bytes"]), float(network_config["downlink_mbps"]))
    round_trip_ms = float(network_config["round_trip_ms"])
    server_ms = _cloud_server_ms(network_config, cloud["inference_latency_ms"])
    cloud_latency_ms = edge["preprocess_latency_ms"] + server_ms + upload_ms + download_ms + round_trip_ms + cloud["postprocess_latency_ms"]

    rows: list[dict[str, Any]] = []
    for policy_name, predictions, latency_vector, offloaded_fraction, bandwidth_bytes in (
        (
            "edge_only",
            edge["y_pred"],
            edge["total_latency_ms"],
            0.0,
            0.0,
        ),
        (
            "cloud_only",
            cloud["y_pred"],
            cloud_latency_ms,
            1.0,
            float(dataset.sample_input_bytes + int(network_config["response_bytes"])),
        ),
    ):
        metrics = classification_summary(y_true=y_true, y_pred=predictions, class_names=dataset.class_names)
        latency = latency_stats(latency_vector)
        rows.append(
            {
                "policy": policy_name,
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "mean_latency_ms": float(latency["mean_ms"]),
                "p50_latency_ms": float(latency["p50_ms"]),
                "p95_latency_ms": float(latency["p95_ms"]),
                "p99_latency_ms": float(latency["p99_ms"]),
                "throughput_samples_per_s": float(1000.0 / max(latency["mean_ms"], 1e-6)),
                "offloaded_fraction": float(offloaded_fraction),
                "bandwidth_kb_per_sample": float(bandwidth_bytes / 1024.0),
                "edge_model": edge_name,
                "cloud_model": cloud_name,
                "hybrid_secondary_model": secondary_name or "",
            }
        )

    if secondary_name and secondary_name in prediction_artifacts:
        secondary = prediction_artifacts[secondary_name]
        use_secondary = edge["confidence"] < threshold
        hybrid_predictions = np.where(use_secondary, secondary["y_pred"], edge["y_pred"])
        hybrid_latency = edge["total_latency_ms"] + np.where(
            use_secondary,
            secondary["inference_latency_ms"] + secondary["postprocess_latency_ms"],
            0.0,
        )
        bandwidth_bytes = 0.0
    else:
        use_secondary = edge["confidence"] < threshold
        hybrid_predictions = np.where(use_secondary, cloud["y_pred"], edge["y_pred"])
        hybrid_latency = edge["total_latency_ms"] + np.where(use_secondary, cloud_latency_ms, 0.0)
        bandwidth_bytes = float(dataset.sample_input_bytes + int(network_config["response_bytes"]))

    hybrid_metrics = classification_summary(
        y_true=y_true,
        y_pred=hybrid_predictions,
        class_names=dataset.class_names,
    )
    hybrid_latency_stats = latency_stats(hybrid_latency)
    rows.append(
        {
            "policy": "hybrid",
            "accuracy": float(hybrid_metrics["accuracy"]),
            "f1_macro": float(hybrid_metrics["f1_macro"]),
            "mean_latency_ms": float(hybrid_latency_stats["mean_ms"]),
            "p50_latency_ms": float(hybrid_latency_stats["p50_ms"]),
            "p95_latency_ms": float(hybrid_latency_stats["p95_ms"]),
            "p99_latency_ms": float(hybrid_latency_stats["p99_ms"]),
            "throughput_samples_per_s": float(1000.0 / max(hybrid_latency_stats["mean_ms"], 1e-6)),
            "offloaded_fraction": float(np.mean(use_secondary)),
            "bandwidth_kb_per_sample": float(np.mean(use_secondary) * bandwidth_bytes / 1024.0),
            "edge_model": edge_name,
            "cloud_model": cloud_name,
            "hybrid_secondary_model": secondary_name or "",
        }
    )
    return rows
