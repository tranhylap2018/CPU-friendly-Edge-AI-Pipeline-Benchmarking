import numpy as np

from edge_bench.pipeline.offload import simulate_offload_policies


def test_hybrid_policy_routes_low_confidence_samples() -> None:
    config = {
        "offload": {
            "enabled": True,
            "edge_model": "linear",
            "cloud_model": "cnn",
            "hybrid_secondary_model": None,
            "confidence_threshold": 0.75,
            "network": {
                "uplink_mbps": 10.0,
                "downlink_mbps": 20.0,
                "round_trip_ms": 30.0,
                "server_processing_ms": 12.0,
                "response_bytes": 128,
                "use_measured_cloud_inference": False,
            },
        }
    }
    dataset = type(
        "DummyDataset",
        (),
        {
            "sample_input_bytes": 256,
            "class_names": ["a", "b"],
        },
    )()
    prediction_artifacts = {
        "linear": {
            "y_true": np.array([0, 1, 1]),
            "y_pred": np.array([0, 0, 1]),
            "confidence": np.array([0.9, 0.4, 0.8], dtype=np.float32),
            "total_latency_ms": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "preprocess_latency_ms": np.array([0.2, 0.2, 0.2], dtype=np.float32),
            "inference_latency_ms": np.array([0.3, 0.3, 0.3], dtype=np.float32),
            "postprocess_latency_ms": np.array([0.1, 0.1, 0.1], dtype=np.float32),
        },
        "cnn": {
            "y_true": np.array([0, 1, 1]),
            "y_pred": np.array([0, 1, 1]),
            "confidence": np.array([0.95, 0.95, 0.95], dtype=np.float32),
            "total_latency_ms": np.array([2.0, 2.0, 2.0], dtype=np.float32),
            "preprocess_latency_ms": np.array([0.2, 0.2, 0.2], dtype=np.float32),
            "inference_latency_ms": np.array([0.8, 0.8, 0.8], dtype=np.float32),
            "postprocess_latency_ms": np.array([0.15, 0.15, 0.15], dtype=np.float32),
        },
    }
    rows = simulate_offload_policies(config, dataset, prediction_artifacts)
    hybrid_row = next(row for row in rows if row["policy"] == "hybrid")
    assert round(hybrid_row["offloaded_fraction"], 3) == round(1.0 / 3.0, 3)
    assert hybrid_row["accuracy"] >= 2 / 3
