# CPU-Friendly Edge AI Pipeline Benchmarking

A reproducible benchmarking framework for evaluating lightweight machine-learning pipelines under CPU-constrained edge-computing conditions.

The project measures the complete inference workflow — from data loading and preprocessing to model inference, postprocessing, and optional edge/cloud routing — with a focus on both predictive performance and system-level efficiency.

Rather than optimizing only for accuracy, this project explores the trade-offs between model quality, latency, throughput, memory usage, model size, and deployment cost.

---

## Overview

The benchmark treats an Edge AI application as a complete system:

```text
Data Loading
    ↓
Preprocessing
    ↓
Model Inference
    ↓
Postprocessing
    ↓
Optional Edge / Cloud Routing
    ↓
Metrics and Artifact Logging
```

This makes it possible to study questions such as:

- How much latency comes from preprocessing versus inference?
- How does model complexity affect tail latency?
- How much accuracy is gained by using a larger model?
- When does cloud offloading improve system performance?
- When is a smaller model more appropriate for constrained hardware?
- How do network assumptions affect edge/cloud trade-offs?

---

## Key Features

- CPU-only benchmarking workflow
- Config-driven experiments
- Multiple datasets and model families
- Stage-wise latency measurement
- Throughput and memory analysis
- Model parameter and serialized-size tracking
- Accuracy and macro-F1 evaluation
- Confidence-based edge/cloud routing simulation
- Reproducible experiment metadata
- Saved predictions, checkpoints, tables, and plots
- Automated report and visualization generation
- Unit tests for core utilities and workflow components

---

## Technology Stack

The project uses:

- Python
- PyTorch
- NumPy
- scikit-learn utilities
- matplotlib
- PyYAML
- pytest

PyTorch is used as the primary modeling framework so the same benchmarking pipeline can support several lightweight architectures across image and time-series workloads.

---

## Benchmark Tracks

### 1. Digits Benchmark

Dataset:

**`sklearn.datasets.load_digits`**

This track provides a small, fast, and highly reproducible environment for testing the complete benchmarking workflow.

Models:

- `linear` — linear classifier baseline
- `mlp` — small multilayer perceptron
- `cnn` — lightweight convolutional neural network
- `cnn_plus` — wider CNN variant

This track is useful for rapid experimentation and validating the benchmark pipeline before running larger experiments.

---

### 2. UCI HAR Sensor Benchmark

Dataset:

**UCI Human Activity Recognition Using Smartphones**

The dataset contains inertial-sensor windows collected from smartphone accelerometers and gyroscopes.

Models:

- `linear` — linear baseline
- `mlp` — multilayer perceptron
- `cnn` — lightweight 1D CNN
- `gru` — lightweight recurrent model

This track provides a more realistic time-series workload for studying Edge AI and sensor-processing systems.

Downloaded and processed dataset files are cached locally and are not required to be stored directly in the repository.

---

## Benchmark Metrics

The framework measures both machine-learning quality and system behavior.

### Model Quality

- Accuracy
- Macro F1 score
- Confusion matrix

### Latency

- Mean latency
- P50 latency
- P95 latency
- P99 latency
- Per-sample latency
- Per-batch latency

### Pipeline Timing

- Data loading time
- Preprocessing time
- Inference time
- Postprocessing time
- End-to-end latency

### Resource Usage

- Peak RSS memory delta
- Model parameter count
- Serialized model size

### Efficiency

- Throughput
- Latency / accuracy ratio
- Efficiency score

### Edge / Cloud Simulation

- Simulated bandwidth usage
- Simulated offloading delay
- Local-processing fraction
- Offloaded-sample behavior

---

## Edge / Cloud Simulation

The benchmark supports three execution modes:

```text
edge_only
cloud_only
hybrid
```

### Edge Only

All samples are processed locally.

This represents a deployment where network communication is unavailable or undesirable.

### Cloud Only

Samples are evaluated through a simulated cloud path that includes configurable communication delay.

### Hybrid

A lightweight local model processes each input first.

If the model confidence exceeds a configured threshold, the prediction is handled locally.

Lower-confidence samples may be escalated to:

- a heavier local model, or
- a simulated cloud path

depending on the experiment configuration.

This allows the benchmark to explore the trade-off between:

```text
local efficiency
        vs.
higher-cost escalation
        vs.
predictive quality
```

---

## Experiment Configurations

Ready-to-run experiments are stored in:

```text
configs/
```

### `baseline_quick.yaml`

Fast Digits benchmark for checking that the complete pipeline works correctly.

### `balanced_benchmark.yaml`

More complete Digits experiment using the available model variants.

### `edge_cloud_tradeoff.yaml`

Focuses on confidence-based routing and simulated network costs.

### `time_series_full.yaml`

Full UCI HAR time-series experiment using:

```text
linear
mlp
cnn
gru
```

Configuration files control experiment settings so runs can be reproduced and compared consistently.

---

## Installation

Python 3.11+ is recommended.

### macOS / Linux

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## Quick Start

### Prepare Data

```bash
python3 main.py prepare-data --config configs/baseline_quick.yaml
```

### Run a Quick Digits Benchmark

```bash
python3 main.py run-all --config configs/baseline_quick.yaml
```

### Run the Balanced Digits Benchmark

```bash
python3 main.py run-all --config configs/balanced_benchmark.yaml
```

### Run the Edge / Cloud Trade-Off Experiment

```bash
python3 main.py run-all --config configs/edge_cloud_tradeoff.yaml
```

### Run the UCI HAR Time-Series Benchmark

```bash
python3 main.py run-all --config configs/time_series_full.yaml
```

### Regenerate Analysis for an Existing Run

```bash
python3 main.py analyze \
  --config configs/baseline_quick.yaml \
  --run-dir results/<run_name>
```

---

## CLI

The main CLI supports:

```text
prepare-data
train
benchmark
analyze
make-report-assets
run-all
```

For most experiments, the easiest entry point is:

```bash
python3 main.py run-all --config <config>
```

The workflow is designed so individual stages can also be executed independently when debugging or analyzing experiments.

---

## Repository Structure

```text
CPU-friendly Edge AI Pipeline Benchmarking/
│
├── configs/
│   └── Experiment configuration files
│
├── data/
│   └── Local dataset downloads and processed caches
│
├── docs/
│   └── Architecture and implementation documentation
│
├── figures/
│   └── Exported benchmark visualizations
│
├── reports/
│   └── Generated and template reports
│
├── results/
│   └── Experiment metrics, checkpoints, logs, and artifacts
│
├── scripts/
│   └── Helper scripts for repeatable workflows
│
├── src/
│   └── edge_bench/
│       └── Core benchmark package
│
├── tests/
│   └── Unit and workflow tests
│
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

Large downloaded datasets and local caches are excluded from version control.

---

## Experiment Outputs

A benchmark run can generate:

- Resolved experiment configuration
- Environment metadata
- Training summary
- Benchmark summary
- Edge/cloud routing summary
- Model checkpoints
- Saved predictions
- CSV metrics
- Markdown summary tables
- Benchmark visualizations
- Generated reports

Raw experiment artifacts are stored under:

```text
results/
```

Reusable or presentation-oriented figures are stored under:

```text
figures/
```

---

## How to Interpret the Results

The goal of this project is not simply to identify the model with the highest accuracy.

A useful Edge AI comparison should consider several dimensions at the same time.

For example:

### Accuracy vs. Latency

A larger model may improve accuracy while significantly increasing inference time.

The useful question is:

> Is the additional predictive performance worth the additional latency?

### Average vs. Tail Latency

Mean latency alone can hide occasional slow predictions.

P95 and P99 latency help identify whether a model has unstable execution behavior.

### Model Size vs. Performance

Smaller models may be preferable when memory, storage, or deployment constraints are more important than small differences in accuracy.

### Pipeline Bottlenecks

Inference may not always dominate total runtime.

Preprocessing, data movement, or postprocessing can become significant parts of end-to-end latency.

### Edge vs. Cloud

Cloud escalation may improve predictive quality but also introduces communication overhead.

Hybrid policies attempt to balance these costs by processing confident inputs locally while escalating difficult inputs.

---

## Reproducibility

The project is designed around repeatable experiments.

Each benchmark configuration defines the experiment settings, while completed runs preserve metadata and structured outputs.

The workflow records information such as:

- Experiment configuration
- Model parameters
- Environment metadata
- Benchmark metrics
- Saved predictions
- Generated artifacts

This makes it easier to compare experiments without relying on manually recorded terminal output.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

The repository includes tests for areas such as:

- CLI behavior
- Configuration handling
- Benchmark metrics
- Edge/cloud offloading logic
- Project structure
- Reproducibility utilities

---

## Design Philosophy

The benchmark intentionally uses relatively small models.

The goal is not to reproduce state-of-the-art deep-learning systems.

Instead, the project focuses on making system-level behavior easy to measure and interpret:

```text
model complexity
      ↓
latency
      ↓
memory
      ↓
throughput
      ↓
accuracy
      ↓
deployment trade-offs
```

This provides a foundation for studying more advanced Edge AI systems later.

---

## Current Limitations

- Benchmarks currently focus on CPU execution.
- Edge/cloud network behavior is simulated rather than measured over a real network.
- The current benchmark does not directly measure hardware power consumption.
- Results depend on the CPU and operating environment used for each run.
- The included models are intentionally lightweight rather than production-scale architectures.
- The benchmark does not yet evaluate specialized accelerators such as GPUs, NPUs, or embedded AI hardware.

---

## Future Work

Potential extensions include:

- ONNX export
- ONNX Runtime benchmarking
- INT8 quantization
- Float32 vs. quantized inference comparisons
- Raspberry Pi deployment
- Embedded-device benchmarking
- Hardware accelerator evaluation
- Energy or power measurements
- Online sensor-stream evaluation
- Rolling latency analysis
- Bootstrap confidence intervals
- Model pruning
- Model compression
- Real network-backed cloud experiments
- Dynamic offloading policies
- Hardware-aware model selection

---

## Motivation

Edge AI requires more than training an accurate model.

Real deployment decisions involve interactions between:

```text
machine learning
systems
hardware constraints
latency
memory
communication
and workload characteristics
```

This project provides a controlled environment for studying those interactions while keeping the experiments small enough to run on ordinary CPU hardware.

---

## License

This project is released under the MIT License.

See `LICENSE` for details.