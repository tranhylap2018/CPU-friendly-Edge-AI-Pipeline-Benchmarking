"""File and tabular output helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_registry_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def dataframe_to_markdown(path: Path, frame: pd.DataFrame, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _simple_markdown_table(frame, index=index)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(markdown)
        handle.write("\n")


def _simple_markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a small markdown table without requiring optional dependencies."""

    working = frame.copy()
    if index:
        working.insert(0, "index", working.index)
    headers = [str(column) for column in working.columns]
    rows = [[str(value) for value in row] for row in working.to_numpy()]

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])
