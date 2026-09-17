"""Model registry entrypoints."""

from edge_bench.models.factory import available_models_for_dataset, build_model

__all__ = ["available_models_for_dataset", "build_model"]
