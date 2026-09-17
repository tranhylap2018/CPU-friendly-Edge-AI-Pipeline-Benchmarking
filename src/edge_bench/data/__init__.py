"""Dataset preparation and loading entrypoints."""

from edge_bench.data.base import DatasetBundle
from edge_bench.data.digits import prepare_digits_dataset
from edge_bench.data.uci_har import prepare_uci_har_dataset


def prepare_dataset(config: dict) -> DatasetBundle:
    name = config["dataset"]["name"]
    if name == "digits":
        return prepare_digits_dataset(config)
    if name == "uci_har":
        return prepare_uci_har_dataset(config)
    raise ValueError(f"Unsupported dataset '{name}'.")
