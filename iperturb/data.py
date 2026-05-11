"""
Perturb-seq dataset for GRNN training.

Supports Replogle-style CRISPRi datasets (K562, RPE1) stored as AnnData
.h5ad files with log1p-normalised counts.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from .utils import logmean

log = logging.getLogger("iperturb")


class PerturbseqDataset(Dataset):
    """
    CRISPRi Perturb-seq dataset.

    Parameters
    ----------
    adata            : AnnData with log1p-normalised counts (from scanpy).
    gene_names       : ordered list of HVG symbols to use as features.
    perturbation_col : obs column containing the perturbed gene symbol.
    control_label    : value in perturbation_col for non-targeting controls.
    min_cells        : minimum cells per perturbation to be included.
    ensg2sym         : {ENSG_id: symbol} mapping (if var_names are ENSG IDs).
    verbose          : whether to print a summary on construction.

    Baseline x0
    ───────────
    Computed as log1p(mean(expm1(X))) in the control cells — arithmetic mean in
    count space then re-log.  This is consistent with GRNN's log_V init and
    avoids the Jensen's-inequality bias of averaging in log space.

    Perturbed value
    ───────────────
    Set to the observed mean expression of the perturbed gene in the knockdown
    cells (not a fixed constant).  For CRISPRi this is typically ≤30% of the
    control baseline and varies gene-by-gene.
    """

    def __init__(
        self,
        adata,
        gene_names: list[str],
        perturbation_col: str = "gene",
        control_label: str = "non-targeting",
        min_cells: int = 5,
        ensg2sym: dict[str, str] | None = None,
        verbose: bool = True,
    ):
        if ensg2sym is None:
            ensg2sym = {}

        var_ids  = adata.var_names.tolist()
        sym2pos: dict[str, int] = {}
        for pos, ensg in enumerate(var_ids):
            sym = ensg2sym.get(ensg, ensg)
            if sym not in sym2pos:
                sym2pos[sym] = pos

        hvg_pos, valid_genes = [], []
        for sym in gene_names:
            if sym in sym2pos:
                hvg_pos.append(sym2pos[sym])
                valid_genes.append(sym)
        if not valid_genes:
            raise ValueError("No valid HVG genes found in adata.var_names.")

        self.gene_names = valid_genes
        self.gene2idx   = {g: i for i, g in enumerate(valid_genes)}
        self.hvg_pos    = np.array(hvg_pos, dtype=np.int64)

        ctrl_mask = adata.obs[perturbation_col].values == control_label
        self.x0   = torch.tensor(
            logmean(adata.X[ctrl_mask])[ self.hvg_pos], dtype=torch.float32
        )

        self.samples: list[tuple[int, float, torch.Tensor]] = []
        for gene_sym, grp in adata.obs.groupby(perturbation_col):
            if gene_sym == control_label:
                continue
            if gene_sym not in self.gene2idx:
                continue
            if len(grp) < min_cells:
                continue
            gene_idx   = self.gene2idx[gene_sym]
            x_obs_full = logmean(adata[grp.index].X)[self.hvg_pos]
            pv         = float(x_obs_full[gene_idx])
            x_obs      = torch.tensor(x_obs_full, dtype=torch.float32)
            self.samples.append((gene_idx, pv, x_obs))

        if not self.samples:
            raise ValueError("No perturbation samples created — check perturbation_col and control_label.")
        if verbose:
            log.info(
                "PerturbseqDataset: %d experiments | %d HVGs",
                len(self.samples), len(valid_genes),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        g, p, x = self.samples[idx]
        return {
            "perturbed_idx":   int(g),
            "perturbed_value": float(p),
            "x_obs":           x,
            "x0":              self.x0,
        }


def split_dataset(
    dataset: PerturbseqDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    seed: int = 42,
):
    """Split dataset into train / validation / test subsets."""
    n       = len(dataset)
    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)
    n_test  = n - n_train - n_val
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


def load_dataset(
    h5ad_path: str,
    gene_names: list[str],
    perturbation_col: str = "gene",
    control_label: str = "non-targeting",
    min_cells: int = 5,
    symbol_col: str = "gene_name",
    n_top_hvgs: int = 2_000,
    normalize: bool = True,
) -> tuple[PerturbseqDataset, list[str]]:
    """
    Load an AnnData .h5ad, run HVG selection, and return a dataset + gene list.

    Parameters
    ----------
    h5ad_path     : path to the .h5ad file
    gene_names    : pre-computed HVG symbol list; if empty, computed here.
    perturbation_col, control_label, min_cells : passed to PerturbseqDataset.
    symbol_col    : adata.var column holding HGNC symbols.
    n_top_hvgs    : number of HVGs to select (used only when gene_names=[]).
    normalize     : if True, run scanpy normalize_total + log1p.

    Returns
    -------
    (dataset, gene_names)
    """
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if not gene_names:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_hvgs, flavor="seurat_v3")
        ensg_hvg  = adata.var_names[adata.var["highly_variable"]].tolist()
        ensg2sym  = adata.var[symbol_col].to_dict() if symbol_col in adata.var.columns else {}
        gene_names = [ensg2sym.get(e, e) for e in ensg_hvg]
    else:
        ensg2sym = adata.var[symbol_col].to_dict() if symbol_col in adata.var.columns else {}

    dataset = PerturbseqDataset(
        adata,
        gene_names,
        perturbation_col=perturbation_col,
        control_label=control_label,
        min_cells=min_cells,
        ensg2sym=ensg2sym,
        verbose=True,
    )
    return dataset, gene_names
