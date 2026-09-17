# Architecture Overview

This document explains the **Phase 1 project foundation**: what the folders mean, how the modules connect, and why the structure is designed this way.

## Design Goal

The project is meant to feel like a small research benchmark suite, not a single script. The core architectural idea is:

`config -> dataset prep -> model training -> pipeline benchmarking -> analysis/report assets`

That separation matters because it makes the workflow:

- reproducible
- easier to debug
- easier to extend
- easier to discuss in research settings

## Folder Roles

- `src/edge_bench/`: main Python package
- `configs/`: named experiment settings
- `data/`: downloaded raw data and processed caches
- `results/`: per-run logs, metrics, checkpoints, predictions, and regenerated artifacts
- `figures/`: exported top-level figures suitable for portfolios or posters
- `reports/`: report template and generated sample writeups
- `scripts/`: convenience wrappers for common workflows
- `tests/`: unit tests for the most reusable core logic
- `docs/`: supporting documentation and research-framing notes

## Package Layout

### `edge_bench/config.py`

Loads YAML config files, merges defaults, validates settings, and creates run directories.

Why this matters:

- experiments become rerunnable
- named configs are easier to compare than manual code edits
- every run can save the exact resolved configuration

### `edge_bench/cli.py`

Defines the command-line interface:

- `prepare-data`
- `train`
- `benchmark`
- `analyze`
- `make-report-assets`
- `run-all`

Why this matters:

- the workflow is explicit
- the student learns tool-oriented experimentation
- the project is easier to demonstrate than a notebook-only setup
- separate commands still compose into a natural workflow:
  `train` makes a fresh run, while `benchmark` and `analyze` can reuse the latest run

### `edge_bench/data/`

Contains dataset-specific preparation code plus the shared dataset bundle abstraction.

- `digits.py`: domain-agnostic default benchmark
- `uci_har.py`: time-series sensor-data benchmark
- `base.py`: shared typed bundle and DataLoader helpers

Why this matters:

- different data modalities still share one benchmark pipeline
- preprocessing logic stays measurable and easy to inspect

### `edge_bench/models/`

Stores lightweight model definitions and the model registry.

Why this matters:

- the code can map config names to actual model classes
- adding a new model becomes a controlled change instead of scattered edits

### `edge_bench/pipeline/`

Contains the main experimental stages:

- `train.py`
- `benchmark.py`
- `offload.py`
- `analysis.py`

Why this matters:

- training and benchmarking are separate concerns
- offload simulation is treated as a systems layer
- analysis does not need to rerun training

### `edge_bench/utils/`

Shared helpers for:

- logging
- metrics
- file output
- reproducibility

Why this matters:

- repeated logic stays centralized
- tests can target small, reusable pieces

## Phase Framing

### Phase 1

Focus on scaffolding and architecture:

- folder structure
- package layout
- configs
- CLI command shape
- documentation
- basic validation tests

### Phase 2

Focus on working ML experiments:

- dataset loading
- baseline models
- training loops
- benchmarking and logging

### Phase 3

Focus on research communication:

- plots
- report template
- poster summary assets

### Phase 4

Focus on polish:

- tests
- usability
- documentation improvements
- ergonomics for rerunning experiments

## Recommended Reading Order

1. `README.md`
2. `configs/baseline_quick.yaml`
3. `src/edge_bench/cli.py`
4. `src/edge_bench/config.py`
5. `src/edge_bench/data/base.py`
6. `src/edge_bench/pipeline/benchmark.py`

That order helps a new student understand the experiment flow before diving into model details.
