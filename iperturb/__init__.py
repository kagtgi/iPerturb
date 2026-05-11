"""iPerturb — Gene Regulatory Neural Network for CRISPRi Perturb-seq data."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("iperturb")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .model import GRNN
from .data import PerturbseqDataset, split_dataset
from .train import grnn_loss, train_grnn, evaluate_all
from .grn import build_grn

__all__ = [
    "GRNN",
    "PerturbseqDataset",
    "split_dataset",
    "grnn_loss",
    "train_grnn",
    "evaluate_all",
    "build_grn",
]
