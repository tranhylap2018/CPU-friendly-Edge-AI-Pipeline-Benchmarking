"""Model training code.

The training module is intentionally simple: the project is about system
tradeoffs, so we avoid overly complex optimization tricks that would distract
from benchmarking methodology.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from edge_bench.data.base import DatasetBundle
from edge_bench.models.factory import available_models_for_dataset, build_model
from edge_bench.utils.io_utils import write_json, write_rows_csv
from edge_bench.utils.metrics import classification_summary


def _evaluate_model(
    model: nn.Module,
    dataset: DatasetBundle,
    split: str,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = dataset.make_loader(
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    losses: list[float] = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for raw_x, raw_y in loader:
            x = dataset.preprocess_batch(raw_x, device=device)
            y = raw_y.to(device=device, dtype=torch.long)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()))
            predictions = torch.argmax(logits, dim=1)
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(predictions.cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    metrics = classification_summary(y_true=y_true, y_pred=y_pred, class_names=dataset.class_names)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_all_models(
    config: dict[str, Any],
    dataset: DatasetBundle,
    run_dir: Path,
    logger: Any,
) -> pd.DataFrame:
    """Train every model named in the config and save checkpoints + metrics."""

    device = torch.device(config["experiment"]["device"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    patience = int(config["training"]["early_stopping_patience"])
    num_workers = int(config["experiment"]["num_workers"])
    seed = int(config["experiment"]["seed"])

    available = available_models_for_dataset(dataset.name)
    history_dir = run_dir / "analysis" / "training_histories"
    history_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for model_name in config["models"]["names"]:
        if model_name not in available:
            raise ValueError(
                f"Model '{model_name}' is not available for dataset '{dataset.name}'. "
                f"Available models: {', '.join(available)}"
            )

        model = build_model(model_name, dataset).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        train_loader = dataset.make_loader(
            split="train",
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history_rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()

        epoch_iterator = tqdm(
            range(1, epochs + 1),
            desc=f"training:{model_name}",
            leave=False,
        )
        for epoch in epoch_iterator:
            model.train()
            batch_losses: list[float] = []
            for raw_x, raw_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                x = dataset.preprocess_batch(raw_x, device=device)
                y = raw_y.to(device=device, dtype=torch.long)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            train_loss = float(np.mean(batch_losses))
            val_metrics = _evaluate_model(
                model=model,
                dataset=dataset,
                split="val",
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                seed=seed,
            )
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_f1_macro": float(val_metrics["f1_macro"]),
                }
            )
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                val_acc=f"{val_metrics['accuracy']:.3f}",
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "model=%s event=early_stop epoch=%s best_val_loss=%.4f",
                        model_name,
                        epoch,
                        best_val_loss,
                    )
                    break

        training_time_s = float(time.perf_counter() - start_time)
        model.load_state_dict(best_state)
        test_metrics = _evaluate_model(
            model=model,
            dataset=dataset,
            split="test",
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            seed=seed,
        )

        checkpoint_path = run_dir / "models" / f"{model_name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "dataset_name": dataset.name,
                "class_names": dataset.class_names,
                "input_shape": dataset.input_shape,
            },
            checkpoint_path,
        )

        history_frame = pd.DataFrame(history_rows)
        history_frame.to_csv(history_dir / f"{model_name}_history.csv", index=False)
        write_json(history_dir / f"{model_name}_history.json", history_rows)

        summary_rows.append(
            {
                "model": model_name,
                "model_label": available[model_name],
                "training_time_s": training_time_s,
                "best_val_loss": best_val_loss,
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1_macro": float(test_metrics["f1_macro"]),
                "checkpoint_path": str(checkpoint_path),
            }
        )
        logger.info(
            "model=%s event=training_complete training_time_s=%.3f test_accuracy=%.4f",
            model_name,
            training_time_s,
            float(test_metrics["accuracy"]),
        )

    training_summary = pd.DataFrame(summary_rows)
    training_summary.to_csv(run_dir / "metrics" / "training_summary.csv", index=False)
    write_rows_csv(run_dir / "metrics" / "training_summary_export.csv", summary_rows)
    return training_summary
