#!/usr/bin/env python3
"""Convenience wrapper for running the full benchmark workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full edge AI benchmark pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "main.py", "run-all", "--config", args.config]
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
