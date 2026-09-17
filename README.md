# CPU-Friendly Edge AI Pipeline Benchmarking

A complete, runnable, research-oriented benchmarking suite for studying edge AI pipelines on CPU-only hardware. The project is designed for a sophomore building undergraduate research assistant skills in areas such as computer systems, cyber-physical systems, machine learning systems, hardware/software interaction, and real-time evaluation.

## Why This Project Exists

Most beginner ML projects stop at model accuracy. Edge AI work in ECE settings is broader: we care about *system tradeoffs under constraints*. This project treats an AI workflow as a full pipeline:

`data loading -> preprocessing -> inference -> postprocessing -> optional edge/cloud routing -> metrics logging`

That framing makes it much more relevant to faculty interested in embedded intelligence, efficient systems, and deployment-aware machine learning.

## Why This Fits Undergraduate Research Preparation

- It teaches experiment design instead of only training scripts.
- It measures latency, throughput, memory, and model size alongside accuracy.
- It saves configs, environment metadata, and structured outputs for reproducibility.
- It creates figures, summary tables, and a poster-style dashboard automatically.
- It makes room for clear engineering discussion: where is the bottleneck, what changed, and why.

## Incremental Build Plan

This project can be developed in four phases:

1. Phase 1: structure, configs, README, requirements, CLI shape, and architecture.
2. Phase 2: datasets, baseline models, training, benchmarking, and logging.
3. Phase 3: plots, report assets, and poster-style summary figures.
4. Phase 4: tests, documentation polish, and CLI usability.

The repository already follows that structure so each stage can be discussed and extended independently.

## Stack Choice

This project uses **PyTorch + NumPy + scikit-learn utilities + matplotlib**.

Why PyTorch here instead of using only scikit-learn?

- A single framework can cover both benchmark tracks cleanly:
  - domain-agnostic image benchmarking on `sklearn` digits
  - time-series sensor benchmarking on UCI HAR
- PyTorch makes it easy to implement a consistent progression of CPU-friendly models:
  - linear baseline
  - small MLP
  - CNN
  - GRU for time series
- Parameter counting, checkpoint saving, and later extensions like ONNX export are much cleaner in one framework.

If the goal were only tabular benchmarking, scikit-learn would be simpler. Here, PyTorch is the better fit because the project intentionally spans multiple modalities while staying CPU-only.

## Benchmark Tracks

### 1. Domain-Agnostic Benchmark Mode

Default dataset: **Digits** from `sklearn.datasets`

- zero-credential
- tiny and fast
- ideal for quick, reproducible smoke tests
- useful for learning the benchmarking workflow before moving to sensor data

### 2. Time-Series Sensor Benchmark Mode

Dataset: **UCI HAR (Human Activity Recognition Using Smartphones)**

- public academic dataset
- lightweight enough for CPU laptops
- includes inertial sensor windows suitable for edge AI discussion
- automatically downloaded and cached by the project

## Models Included

### Digits track

- `linear`: linear classifier baseline
- `mlp`: small multilayer perceptron
- `cnn`: tiny CNN
- `cnn_plus`: wider CNN

### UCI HAR track

- `linear`: linear baseline
- `mlp`: small MLP
- `cnn`: 1D CNN
- `gru`: lightweight GRU

The point is not to chase state-of-the-art accuracy. The point is to compare models with increasing complexity and discuss whether the extra cost is worth it under deployment constraints.

## Metrics Reported

- training time
- mean, p50, p95, p99 latency
- per-sample and per-batch latency
- throughput
- peak RSS memory delta
- model parameter count
- serialized model size on disk
- accuracy
- macro F1
- confusion matrix
- data loading time
- preprocessing time
- inference time
- postprocessing time
- end-to-end latency
- latency/accuracy ratio
- efficiency score
- simulated bandwidth usage
- simulated offloading delay

## Edge/Cloud Simulation

The benchmark includes three system modes:

- `edge_only`
- `cloud_only`
- `hybrid`

The hybrid policy uses a confidence threshold. A lightweight local model handles confident samples locally; lower-confidence samples are escalated either to a heavier local model or to a simulated cloud path, depending on the config.

This is useful for discussing:

- when offloading helps
- when it hurts
- how network assumptions affect system behavior
- how confidence-based routing changes the latency/accuracy tradeoff

## Installation

Use Python 3.11+ in a virtual environment.

```bash
python3 -m venv .venv
.venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Quick Start

### 1. Prepare the default dataset cache

```bash
python3 main.py prepare-data --config configs/baseline_quick.yaml
```

### 2. Run the full digits benchmark

```bash
python3 main.py run-all --config configs/baseline_quick.yaml
```

### 3. Run the balanced digits benchmark

```bash
python3 main.py run-all --config configs/balanced_benchmark.yaml
```

### 4. Run the edge/cloud tradeoff experiment

```bash
python3 main.py run-all --config configs/edge_cloud_tradeoff.yaml
```

### 5. Run the full time-series experiment

```bash
python3 main.py run-all --config configs/time_series_full.yaml
```

### 6. Regenerate figures from a completed run

```bash
python3 main.py analyze --config configs/baseline_quick.yaml --run-dir results/<run_name>
```

## CLI Commands

- `prepare-data`
- `train`
- `benchmark`
- `analyze`
- `make-report-assets`
- `run-all`

Each command accepts a config path. `run-all` is the easiest path for a first run.

By default:

- `train` creates a fresh run directory
- `benchmark` reuses the latest matching run if one already exists
- `analyze` targets the latest matching run unless `--run-dir` is provided

## Project Structure

```text
CPU-friendly Edge AI Pipeline Benchmarking/
├── configs/                  # Ready-to-run experiment configs
├── data/                     # Raw data, processed caches
├── docs/                     # RA growth notes and supporting docs
├── figures/                  # Exported portfolio-ready figures
├── notebooks/                # Optional notebook area
├── reports/                  # Report template and generated sample report
├── results/                  # Per-run artifacts, metrics, logs, checkpoints
├── scripts/                  # Helper scripts for repeatable workflows
├── src/edge_bench/           # Package source code
├── tests/                    # Unit tests for core utilities
├── main.py                   # Root CLI wrapper
├── pyproject.toml
└── requirements.txt
```

## Experiment Guide

### `baseline_quick.yaml`

Fast smoke test on digits. Best for checking that the full workflow runs correctly on a laptop.

### `balanced_benchmark.yaml`

A more complete digits benchmark with all four domain-agnostic models.

### `edge_cloud_tradeoff.yaml`

Focuses on confidence-based hybrid routing and simulated network costs.

### `time_series_full.yaml`

The main time-series benchmark using UCI HAR and the full linear/MLP/CNN/GRU model set.

## Interpretation Guide

When you review results, do not ask only “which model is best?” Ask:

- Which model gives the best accuracy per millisecond?
- Is preprocessing a significant fraction of total latency?
- Which model has the worst tail latency?
- Does the hybrid policy improve accuracy enough to justify added delay?
- Would a smaller model be more appropriate for a stricter edge device?

This is the systems-thinking mindset faculty often care about.

## Output Artifacts

Each run stores:

- resolved config
- environment metadata
- training summary
- benchmark summary
- offload summary
- model checkpoints
- saved predictions
- plots
- markdown summary tables
- a generated sample report

## Suggested Resume Bullet Points

- Built a reproducible CPU-only edge AI benchmarking suite in Python and PyTorch to evaluate full inference pipelines under latency, throughput, memory, and model-size constraints.
- Implemented config-driven experiments, confidence-based edge/cloud offload simulation, automated plotting, and poster-style report assets for deployment-aware model analysis.
- Benchmarked lightweight linear, MLP, CNN, and GRU models across image and time-series datasets with structured logging, saved artifacts, and reproducible experiment metadata.

## Suggested Ways To Discuss This Project With Faculty

- “I wanted a project that emphasized systems tradeoffs, not just training accuracy.”
- “I measured stage-wise latency so I could separate preprocessing cost from inference cost.”
- “I used a config-driven workflow because I wanted experiments to be rerunnable and comparable.”
- “I added a hybrid offload policy to think about edge/cloud scheduling rather than only local inference.”
- “I chose small CPU-friendly models because I wanted the evaluation setup to reflect realistic student hardware.”

## Future Extensions

- ONNX export and ONNX Runtime benchmarking
- quantized inference comparisons
- online sensor-stream scheduling
- energy estimation or power proxies
- bootstrap confidence intervals
- real network-backed cloud benchmarking
- model compression experiments

## A Good Study Order For Students

1. `configs/baseline_quick.yaml`
2. `src/edge_bench/cli.py`
3. `src/edge_bench/pipeline/benchmark.py`
4. `src/edge_bench/pipeline/offload.py`
5. `src/edge_bench/pipeline/analysis.py`

That path helps you understand experiment control first, then timing methodology, then systems interpretation.
