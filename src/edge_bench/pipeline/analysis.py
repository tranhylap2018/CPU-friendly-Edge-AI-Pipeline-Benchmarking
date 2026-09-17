"""Analysis, plots, poster assets, and lightweight report generation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edge_bench.utils.io_utils import dataframe_to_markdown, write_json


PLOT_COLORS = {
    "linear": "#2E86AB",
    "mlp": "#F18F01",
    "cnn": "#C73E1D",
    "cnn_plus": "#6C8EAD",
    "gru": "#4D9078",
}


def _apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBF6",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def _plot_accuracy_vs_latency(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in frame.iterrows():
        ax.scatter(
            row["mean_sample_latency_ms"],
            row["accuracy"],
            s=max(80, row["serialized_model_size_mb"] * 800),
            color=PLOT_COLORS.get(row["model"], "#4C566A"),
            alpha=0.8,
        )
        ax.annotate(row["model"], (row["mean_sample_latency_ms"], row["accuracy"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean latency per sample (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. End-to-End Latency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_model_size_vs_latency(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        frame["serialized_model_size_mb"],
        frame["p95_sample_latency_ms"],
        c=[PLOT_COLORS.get(name, "#4C566A") for name in frame["model"]],
        s=110,
    )
    for _, row in frame.iterrows():
        ax.annotate(row["model"], (row["serialized_model_size_mb"], row["p95_sample_latency_ms"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Serialized model size (MB)")
    ax.set_ylabel("p95 latency per sample (ms)")
    ax.set_title("Model Size vs. Tail Latency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_throughput(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PLOT_COLORS.get(name, "#4C566A") for name in frame["model"]]
    ax.bar(frame["model"], frame["throughput_samples_per_s"], color=colors)
    ax.set_ylabel("Samples / second")
    ax.set_title("Pipeline Throughput")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_stage_latency(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(frame))
    load = frame["mean_data_loading_ms"].to_numpy()
    pre = frame["mean_preprocessing_ms"].to_numpy()
    infer = frame["mean_inference_ms"].to_numpy()
    post = frame["mean_postprocessing_ms"].to_numpy()
    ax.bar(x, load, label="Data loading", color="#66829E")
    ax.bar(x, pre, bottom=load, label="Preprocessing", color="#A1C181")
    ax.bar(x, infer, bottom=load + pre, label="Inference", color="#F4A259")
    ax.bar(x, post, bottom=load + pre + infer, label="Postprocessing", color="#BC4B51")
    ax.set_xticks(x, frame["model"])
    ax.set_ylabel("Mean latency per sample (ms)")
    ax.set_title("Stage-wise Pipeline Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_p95_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(frame["model"], frame["p95_sample_latency_ms"], color="#5A7D7C")
    ax.set_ylabel("p95 latency per sample (ms)")
    ax.set_title("Tail-Latency Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_pareto_frontier(frame: pd.DataFrame, output_path: Path) -> None:
    sorted_frame = frame.sort_values("mean_sample_latency_ms")
    best_accuracy = 0.0
    pareto_points: list[tuple[float, float]] = []
    for _, row in sorted_frame.iterrows():
        if row["accuracy"] >= best_accuracy:
            pareto_points.append((row["mean_sample_latency_ms"], row["accuracy"]))
            best_accuracy = row["accuracy"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sorted_frame["mean_sample_latency_ms"], sorted_frame["accuracy"], color="#7F5AF0", s=110)
    if pareto_points:
        x_values, y_values = zip(*pareto_points)
        ax.plot(x_values, y_values, color="#EF4565", linewidth=2.5, label="Pareto frontier")
    for _, row in sorted_frame.iterrows():
        ax.annotate(row["model"], (row["mean_sample_latency_ms"], row["accuracy"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean latency per sample (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Accuracy vs. Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), class_names)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="#1F1F1F")
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_offload_summary(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    if frame.empty:
        for axis in axes:
            axis.axis("off")
            axis.text(0.1, 0.5, "No offload results for this run.", fontsize=11)
    else:
        axes[0].bar(frame["policy"], frame["mean_latency_ms"], color=["#4C956C", "#F4D35E", "#EE964B"])
        axes[0].set_ylabel("Mean latency (ms)")
        axes[0].set_title("Edge vs. Cloud vs. Hybrid")
        axes[1].bar(frame["policy"], frame["accuracy"], color=["#4C956C", "#F4D35E", "#EE964B"])
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Policy Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_pipeline_diagram(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    boxes = [
        (0.05, 0.38, 0.14, 0.24, "Sensor / Data"),
        (0.23, 0.38, 0.14, 0.24, "Preprocess"),
        (0.41, 0.38, 0.14, 0.24, "Inference"),
        (0.59, 0.38, 0.14, 0.24, "Postprocess"),
        (0.77, 0.54, 0.16, 0.20, "Local decision"),
        (0.77, 0.22, 0.16, 0.20, "Hybrid / cloud"),
    ]
    for x, y, w, h, label in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor="#F8F3D4", edgecolor="#6B705C", linewidth=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, weight="bold")

    arrows = [
        ((0.19, 0.50), (0.23, 0.50)),
        ((0.37, 0.50), (0.41, 0.50)),
        ((0.55, 0.50), (0.59, 0.50)),
        ((0.73, 0.50), (0.77, 0.64)),
        ((0.73, 0.50), (0.77, 0.32)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "#355070"})
    ax.text(0.86, 0.08, "All paths log latency, accuracy,\nthroughput, memory, and model size.", ha="center", va="center")
    ax.set_title("Edge AI Pipeline Under Resource Constraints", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _best_row(frame: pd.DataFrame, sort_column: str, ascending: bool) -> pd.Series:
    return frame.sort_values(sort_column, ascending=ascending).iloc[0]


def _make_findings(frame: pd.DataFrame, offload_frame: pd.DataFrame) -> list[str]:
    fastest = _best_row(frame, "mean_sample_latency_ms", True)
    most_accurate = _best_row(frame, "accuracy", False)
    best_tail = _best_row(frame, "p95_sample_latency_ms", True)
    findings = [
        f"Fastest pipeline: {fastest['model']} at {fastest['mean_sample_latency_ms']:.2f} ms/sample.",
        f"Highest accuracy: {most_accurate['model']} at {most_accurate['accuracy']:.3f}.",
        f"Best tail latency: {best_tail['model']} with p95 = {best_tail['p95_sample_latency_ms']:.2f} ms.",
    ]
    if not offload_frame.empty:
        best_policy = _best_row(offload_frame, "mean_latency_ms", True)
        findings.append(
            f"Best policy latency: {best_policy['policy']} at {best_policy['mean_latency_ms']:.2f} ms."
        )
    return findings


def _make_poster_dashboard(
    config: dict[str, Any],
    frame: pd.DataFrame,
    offload_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    findings = _make_findings(frame, offload_frame)
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, 3, height_ratios=[0.6, 1.2, 1.1])

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.85, config["analysis"]["poster_title"], fontsize=22, weight="bold")
    title_ax.text(0.0, 0.45, config["analysis"]["motivation"], fontsize=12)
    title_ax.text(0.0, 0.08, "\n".join(f"- {finding}" for finding in findings), fontsize=11, va="bottom")

    scatter_ax = fig.add_subplot(grid[1, 0])
    scatter_ax.scatter(frame["mean_sample_latency_ms"], frame["accuracy"], color="#005F73", s=120)
    for _, row in frame.iterrows():
        scatter_ax.annotate(row["model"], (row["mean_sample_latency_ms"], row["accuracy"]), xytext=(4, 4), textcoords="offset points")
    scatter_ax.set_title("Accuracy vs. Latency")
    scatter_ax.set_xlabel("Mean sample latency (ms)")
    scatter_ax.set_ylabel("Accuracy")

    stage_ax = fig.add_subplot(grid[1, 1])
    x = np.arange(len(frame))
    stage_ax.bar(x, frame["mean_data_loading_ms"], label="Load", color="#7B9E89")
    stage_ax.bar(x, frame["mean_preprocessing_ms"], bottom=frame["mean_data_loading_ms"], label="Pre", color="#E9C46A")
    stage_ax.bar(
        x,
        frame["mean_inference_ms"],
        bottom=frame["mean_data_loading_ms"] + frame["mean_preprocessing_ms"],
        label="Infer",
        color="#F4A261",
    )
    stage_ax.bar(
        x,
        frame["mean_postprocessing_ms"],
        bottom=frame["mean_data_loading_ms"] + frame["mean_preprocessing_ms"] + frame["mean_inference_ms"],
        label="Post",
        color="#E76F51",
    )
    stage_ax.set_xticks(x, frame["model"])
    stage_ax.set_title("Stage Breakdown")
    stage_ax.set_ylabel("ms / sample")
    stage_ax.legend(fontsize=8)

    table_ax = fig.add_subplot(grid[1, 2])
    table_ax.axis("off")
    compact = frame[["model", "accuracy", "p95_sample_latency_ms", "throughput_samples_per_s"]].copy()
    compact["accuracy"] = compact["accuracy"].map(lambda value: f"{value:.3f}")
    compact["p95_sample_latency_ms"] = compact["p95_sample_latency_ms"].map(lambda value: f"{value:.2f}")
    compact["throughput_samples_per_s"] = compact["throughput_samples_per_s"].map(lambda value: f"{value:.1f}")
    table = table_ax.table(
        cellText=compact.values,
        colLabels=["Model", "Acc.", "p95 ms", "Throughput"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.5)
    table_ax.set_title("Key Metrics")

    pipeline_ax = fig.add_subplot(grid[2, 0])
    pipeline_ax.axis("off")
    pipeline_ax.text(0.02, 0.88, "Pipeline", fontsize=13, weight="bold")
    pipeline_ax.text(
        0.02,
        0.55,
        "source -> preprocessing -> inference -> postprocessing\n"
        "-> local decision / hybrid offload / cloud path\n"
        "-> metrics logging",
        fontsize=11,
        bbox={"facecolor": "#F1FAEE", "edgecolor": "#457B9D", "boxstyle": "round,pad=0.6"},
    )

    pareto_ax = fig.add_subplot(grid[2, 1])
    pareto_ax.scatter(frame["mean_sample_latency_ms"], frame["accuracy"], color="#9B2226", s=110)
    sorted_frame = frame.sort_values("mean_sample_latency_ms")
    frontier_x: list[float] = []
    frontier_y: list[float] = []
    best_accuracy = 0.0
    for _, row in sorted_frame.iterrows():
        if row["accuracy"] >= best_accuracy:
            frontier_x.append(row["mean_sample_latency_ms"])
            frontier_y.append(row["accuracy"])
            best_accuracy = row["accuracy"]
    if frontier_x:
        pareto_ax.plot(frontier_x, frontier_y, color="#3A86FF", linewidth=2)
    pareto_ax.set_title("Pareto Frontier")
    pareto_ax.set_xlabel("Latency (ms)")
    pareto_ax.set_ylabel("Accuracy")

    policy_ax = fig.add_subplot(grid[2, 2])
    if offload_frame.empty:
        policy_ax.axis("off")
        policy_ax.text(0.05, 0.5, "No offload simulation for this run.", fontsize=11)
    else:
        policy_ax.bar(offload_frame["policy"], offload_frame["mean_latency_ms"], color=["#588157", "#BC4749", "#3A5A40"])
        policy_ax.set_title("Policy Latency")
        policy_ax.set_ylabel("Mean latency (ms)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_analysis_artifacts(
    config: dict[str, Any],
    dataset: Any,
    run_dir: Path,
    logger: Any,
) -> dict[str, str]:
    """Create plots, summary tables, poster assets, and a sample report."""

    _apply_plot_style()
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    project_figures_dir = Path(config["paths"]["figures_root"])
    project_figures_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(metrics_dir / "benchmark_summary.csv")
    offload_path = metrics_dir / "offload_summary.csv"
    offload = pd.read_csv(offload_path) if offload_path.read_text(encoding="utf-8").strip() else pd.DataFrame()
    primary = benchmark[benchmark["batch_size"] == int(config["benchmark"]["primary_batch_size"])].copy()
    primary = primary.sort_values("mean_sample_latency_ms")

    output_map = {
        "accuracy_vs_latency": figures_dir / "accuracy_vs_latency.png",
        "model_size_vs_latency": figures_dir / "model_size_vs_latency.png",
        "throughput_bar": figures_dir / "throughput_bar.png",
        "stage_latency": figures_dir / "stage_latency.png",
        "p95_latency": figures_dir / "p95_latency.png",
        "pareto_frontier": figures_dir / "pareto_frontier.png",
        "policy_comparison": figures_dir / "policy_comparison.png",
        "pipeline_diagram": figures_dir / "pipeline_diagram.png",
        "poster_dashboard": figures_dir / "poster_dashboard.png",
    }

    _plot_accuracy_vs_latency(primary, output_map["accuracy_vs_latency"])
    _plot_model_size_vs_latency(primary, output_map["model_size_vs_latency"])
    _plot_throughput(primary, output_map["throughput_bar"])
    _plot_stage_latency(primary, output_map["stage_latency"])
    _plot_p95_comparison(primary, output_map["p95_latency"])
    _plot_pareto_frontier(primary, output_map["pareto_frontier"])
    _plot_offload_summary(offload, output_map["policy_comparison"])
    _draw_pipeline_diagram(output_map["pipeline_diagram"])
    _make_poster_dashboard(config, primary, offload, output_map["poster_dashboard"])

    for model_name in primary["model"]:
        artifact = np.load(run_dir / "predictions" / f"{model_name}.npz", allow_pickle=True)
        _plot_confusion_matrix(
            matrix=artifact["confusion_matrix"],
            class_names=dataset.class_names,
            title=f"Confusion Matrix: {model_name}",
            output_path=figures_dir / f"confusion_matrix_{model_name}.png",
        )

    dataframe_to_markdown(run_dir / "analysis" / "benchmark_summary.md", primary, index=False)
    dataframe_to_markdown(run_dir / "analysis" / "offload_summary.md", offload, index=False)
    write_json(run_dir / "analysis" / "top_findings.json", {"findings": _make_findings(primary, offload)})

    for figure_path in output_map.values():
        shutil.copy2(figure_path, project_figures_dir / f"{run_dir.name}_{figure_path.name}")

    _write_sample_report(config, dataset, run_dir, primary, offload)

    logger.info("event=analysis_complete figures_dir=%s", figures_dir)
    return {name: str(path) for name, path in output_map.items()}


def _write_sample_report(
    config: dict[str, Any],
    dataset: Any,
    run_dir: Path,
    benchmark: pd.DataFrame,
    offload: pd.DataFrame,
) -> None:
    reports_root = Path(config["paths"]["reports_root"])
    reports_root.mkdir(parents=True, exist_ok=True)
    fastest = benchmark.sort_values("mean_sample_latency_ms").iloc[0]
    most_accurate = benchmark.sort_values("accuracy", ascending=False).iloc[0]
    hybrid_line = ""
    if not offload.empty:
        hybrid = offload.loc[offload["policy"] == "hybrid"].iloc[0]
        hybrid_line = (
            f"- Hybrid routing offloaded {hybrid['offloaded_fraction'] * 100:.1f}% of samples "
            f"with mean latency {hybrid['mean_latency_ms']:.2f} ms.\n"
        )

    sample_report = f"""# Sample Report: {config['analysis']['poster_title']}

## Title
CPU-Friendly Edge AI Pipeline Benchmarking on {dataset.name}

## Abstract
This project benchmarks full edge AI pipelines on CPU-only hardware to study how accuracy, latency, throughput, memory, and model size trade off under practical constraints. The benchmark is intentionally framed as a systems project rather than a pure model-training exercise.

## Introduction
Edge AI deployment is shaped by end-to-end pipeline costs, not inference alone. This run focuses on {dataset.name}, a {dataset.track.replace('_', ' ')} benchmark, to compare lightweight models under a reproducible workflow.

## Related Motivation
For undergraduate research preparation, the interesting question is not just which model wins on accuracy, but which model is most appropriate under a specific latency, memory, and deployment budget.

## System Design
The pipeline measures data loading, preprocessing, inference, postprocessing, and optional edge/cloud routing. All experiments were run with deterministic seeds and saved configs.

## Methods
Models were trained in PyTorch on CPU and benchmarked with repeated passes over the test set to estimate mean, p50, p95, and p99 latency. Offload simulation used configurable bandwidth and server-delay assumptions.

## Datasets
- Dataset: {dataset.name}
- Track: {dataset.track}
- Input shape: {dataset.input_shape}
- Classes: {", ".join(dataset.class_names)}

## Models
Compared models: {", ".join(benchmark['model'].tolist())}

## Benchmark Metrics
Reported metrics include training time, accuracy, macro F1, latency percentiles, throughput, peak RSS delta, parameter count, serialized size, and simulated offloading costs.

## Experiment Setup
- Config: `{config['_meta']['config_path']}`
- Primary batch size: {config['benchmark']['primary_batch_size']}
- Epochs: {config['training']['epochs']}
- Device: CPU

## Results
- Fastest model: {fastest['model']} at {fastest['mean_sample_latency_ms']:.2f} ms/sample.
- Most accurate model: {most_accurate['model']} at accuracy {most_accurate['accuracy']:.3f}.
- Best throughput: {benchmark.sort_values('throughput_samples_per_s', ascending=False).iloc[0]['model']} at {benchmark['throughput_samples_per_s'].max():.1f} samples/s.
{hybrid_line}## Discussion
The run shows the core systems tradeoff: heavier models improve accuracy only if their added latency and size remain acceptable for the deployment target. Stage-wise timing also shows whether engineering effort should go into the model, preprocessing, or data movement.

## Limitations
The cloud path is simulated rather than deployed on a real server, and results depend on the chosen network-delay assumptions.

## Future Work
Future extensions could include ONNX Runtime benchmarking, quantized inference, sensor-stream jitter models, and online scheduling policies.

## Conclusion
This benchmark suite demonstrates research readiness by combining reproducible experimentation, systems instrumentation, and clear tradeoff analysis.

## References Placeholder
- Add papers on edge AI systems, embedded ML, and benchmarking methodology here.
"""
    sample_report_path = reports_root / "sample_report.md"
    sample_report_path.write_text(sample_report, encoding="utf-8")
