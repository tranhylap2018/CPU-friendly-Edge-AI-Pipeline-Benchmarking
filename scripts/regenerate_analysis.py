#!/usr/bin/env python3
"""Regenerate analysis artifacts from an existing run directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures and markdown tables.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "main.py",
        "analyze",
        "--config",
        args.config,
        "--run-dir",
        args.run_dir,
    ]
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
