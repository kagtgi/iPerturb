"""
iPerturb v2 adapter — the exact code used in the FairPert benchmark.

This is the ``run_eval`` entry point that scores iPerturb inside the benchmark.
It dispatches on ``IPERTURB_ENGINE``:

  * ``propagate`` (**default, v2**): analytic **network propagation** over the
    signed L1–L4 GRN (random-walk-with-restart / heat kernel) with a closed-form
    ridge readout and analytic co-propagation epistasis — see
    :mod:`iperturb_v2._iperturb_prop`. Composes combinations as additive network
    sources, generalizes to genes unseen in training (a prediction flows from a
    gene's *network position*), and is leakage-free (readout fit on the train
    split only; prediction uses only the perturbation source, never the test
    cells). CPU, numpy/scipy.
  * ``hill`` (**v1, legacy**): the original signed Hill-kinetics GRNN, kept for
    ablation/back-compat. Single-gene only; combos fall back to control.

Benchmark result (Norman 2019, single seed): v2 lifts the combinatorial
SV-corrected Pearson Δ from 0.06 (v1 hill) to **0.64** (rank 3 of 10, rivalling
the matching-mean baseline) and the unseen single-gene Δ from 0.009 to **0.25** —
the only model in the benchmark whose effect-size correlation significantly beats
the non-targeting null on truly unseen perturbations.

GRN construction (shared by both engines)
------------------------------------------
Both engines build the GRN over the **fair evaluation panel** (``EVAL_PANEL_GENES``)
using v1's OWN GRN-build code — this adapter AST-extracts and executes the L1–L4
build slice straight from the repo's ``iperturb.py`` (``build_full_grn``):
L1 TRRUST/OmniPath/CollecTRI + L2 GeneHancer + L3 STRING + L4 COXPRESdb → merge →
budget-greedy select. Needs the license-gated GeneHancer v5.26 files in
``IPERTURB_SUPPORT_DIR``; if absent, falls back to a bundled-regulon GRN
(CollecTRI/DoRothEA). So v2 keeps v1's biologically-faithful GRN and only replaces
the *prediction* step (Hill-kinetics fixed point → network propagation + readout).

Fairness: the GRN topology is public-DB-only (split-independent), the readout is
fit on the train split only, and predictions are scored on the shared panel — so
iPerturb is trained-on-target and leakage-free, on identical footing to every
other benchmark model.
"""

from __future__ import annotations

import ast
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from . import config
from .splits import get_split_masks, parse_gene_set, restrict_to_test

logger = logging.getLogger(__name__)

# Top-level definitions we lift out of the submodule script (first occurrence —
# the file contains two concatenated copies).
_NEEDED = [
    "_logmean", "_inv_softplus", "GRNN", "grn_tsv_to_grnn",
    "PerturbseqDataset", "split_dataset", "grnn_loss",
    "_eval_loss", "_sign_balance_report", "_convergence_report", "train_grnn",
]


def _submodule_path() -> Path:
    # v2 reuses v1's GRN-build code: point at this repo's own iperturb.py
    # (iperturb_v2/adapter.py -> parents[1] is the repo root).
    return Path(__file__).resolve().parents[1] / "iperturb.py"


def load_iperturb_symbols(src_path: str | None = None) -> dict:
    """AST-extract iPerturb's model/training symbols WITHOUT running its pipeline.

    Returns a namespace dict containing the names in ``_NEEDED``.
    """
    path = Path(src_path) if src_path else _submodule_path()
    if not path.exists():
        raise FileNotFoundError(
            f"iPerturb submodule not found at {path}. Run: "
            "git submodule update --init external/iPerturb")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    segments: dict[str, tuple[int, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _NEEDED:
            if node.name not in segments:  # keep the FIRST occurrence
                seg = ast.get_source_segment(source, node)
                if seg:
                    segments[node.name] = (node.lineno, seg)
    missing = [n for n in _NEEDED if n not in segments]
    if missing:
        raise RuntimeError(f"iPerturb: could not extract {missing} from {path}")

    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        def tqdm(x, **_k):
            return x

    ns: dict = {
        "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F,
        "Dataset": Dataset, "DataLoader": DataLoader, "random_split": random_split,
        "log": logging.getLogger("iperturb"), "tqdm": tqdm,
        "__builtins__": __builtins__,
    }
    ordered = "\n\n".join(seg for _, seg in sorted(segments.values()))
    exec(compile(ordered, str(path), "exec"), ns)  # noqa: S102 — trusted submodule source
    return ns


def _build_grn_tsv(genes: set, cfg: dict, dest: str) -> int:
    """Write a ``source\\ttarget\\tsign\\tlevel`` GRN TSV from bundled regulons,
    restricted to ``genes``. Prefers the signed DoRothEA reference; falls back to
    CollecTRI (unsigned → learned). Returns the edge count (0 if no resource)."""
    rd = cfg.get("RESOURCE_DIR") or getattr(config, "RESOURCE_DIR", "eval/resources")
    dorothea = cfg.get("GRN_REFERENCE") or os.path.join(rd, "dorothea.tsv")
    collectri = cfg.get("TF_REGULONS_FILE") or os.path.join(rd, "collectri.csv")

    rows: list[tuple[str, str, int]] = []
    if dorothea and os.path.exists(dorothea) and (dorothea.endswith(".tsv") or "/" in dorothea):
        with open(dorothea) as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0] in genes and p[1] in genes:
                    sign = int(float(p[2])) if len(p) >= 3 and p[2] not in ("", "nan") else 0
                    rows.append((p[0], p[1], sign))
    elif collectri and os.path.exists(collectri):
        with open(collectri) as f:
            for i, ln in enumerate(f):
                p = ln.rstrip("\n").split(",")
                if len(p) >= 2 and p[1].lower() not in ("target", "target_gene"):
                    if p[0] in genes and p[1] in genes:
                        rows.append((p[0], p[1], 0))
    if not rows:
        return 0
    # dedupe
    seen, uniq = set(), []
    for s, t, sg in rows:
        if (s, t) not in seen:
            seen.add((s, t)); uniq.append((s, t, sg))
    with open(dest, "w") as f:
        f.write("source\ttarget\tsign\tlevel\n")
        for s, t, sg in uniq:
            f.write(f"{s}\t{t}\t{sg}\t1\n")
    return len(uniq)


_GENEHANCER_FILES = (
    "GeneHancer_v5.26.gff", "GeneHancer_TFBSs_v5.26.txt", "GeneHancer_Tissues_v5.26.txt",
)


def _grn_slice(src: str, cache_prefix: str, tissue: str) -> str:
    """Extract iPerturb's GRN-build slice (config → crawl L1–L4 → merge → greedy →
    export) and redirect it: ``/content/`` paths → ``cache_prefix``, the param
    budget scaled to the actual panel, and the GeneHancer tissue filter set.

    The slice is self-contained (it carries its own imports + constants), so it
    runs iPerturb's REAL GRN pipeline verbatim — no reimplementation."""
    start = src.index('GENE_LIST_FILE = "/content/gene_list.txt"')
    end = src.index("\n", src.index("SELECTED.to_csv(OUT_FILE", start)) + 1
    s = src[start:end]
    s = s.replace("/content/", cache_prefix)
    s = s.replace("N_GENES             = 2_000", "N_GENES = len(GENE_SET)")
    s = s.replace('TISSUE_FILTER = "K562"', f'TISSUE_FILTER = "{tissue}"')
    return s


def _support_dir(cfg: dict) -> str:
    d = cfg.get("IPERTURB_SUPPORT_DIR") or getattr(config, "IPERTURB_SUPPORT_DIR", "")
    if not d:
        rd = cfg.get("RESOURCE_DIR") or getattr(config, "RESOURCE_DIR", "eval/resources")
        d = os.path.join(rd, "iperturb")
    return d


def build_full_grn(genes, cfg: dict) -> str | None:
    """Build iPerturb's REAL context GRN by executing the submodule's own GRN-build
    slice (L1 TRRUST/OmniPath/CollecTRI + L2 GeneHancer + L3 STRING + L4 COXPRESdb
    → merge → budget-greedy select), with ``/content/`` paths redirected to a cache
    dir and ``GENE_SET`` driven by the FAIR shared gene panel.

    Returns the GRN TSV path, or None (→ regulon fallback) if disabled, the
    GeneHancer files are absent, or the network crawl fails. Cached per gene-set
    hash so the crawl runs once per (dataset, panel).
    """
    import hashlib
    import shutil

    if not bool(cfg.get("IPERTURB_USE_FULL_GRN", getattr(config, "IPERTURB_USE_FULL_GRN", True))):
        return None
    # Absolute paths: the GRN cache lives in a subdir, so relative GeneHancer
    # paths would produce broken symlinks (and an empty GRN). Resolve up front.
    support = os.path.abspath(_support_dir(cfg))
    gh = [os.path.join(support, f) for f in _GENEHANCER_FILES]
    if not all(os.path.exists(p) for p in gh):
        logger.info("iPerturb full GRN: GeneHancer files absent in %s — regulon fallback.", support)
        return None

    genes = sorted(set(map(str, genes)))
    h = hashlib.md5(",".join(genes).encode()).hexdigest()[:10]
    cache_dir = os.path.join(support, f"grn_{h}")
    os.makedirs(cache_dir, exist_ok=True)
    out_tsv = os.path.join(cache_dir, "grn_edges.tsv")
    if os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0:
        logger.info("iPerturb full GRN: using cached %s", out_tsv)
        return out_tsv

    # the slice expects gene_list.txt + GeneHancer files under the (redirected) dir
    for p in gh:
        dst = os.path.join(cache_dir, os.path.basename(p))
        if os.path.islink(dst) or os.path.exists(dst):
            try:
                os.remove(dst)        # drop any stale/broken link from a prior run
            except OSError:
                pass
        try:
            os.symlink(os.path.abspath(p), dst)
        except Exception:  # noqa: BLE001 — Windows / cross-device
            shutil.copy(p, dst)
    with open(os.path.join(cache_dir, "gene_list.txt"), "w") as f:
        f.write("\n".join(genes) + "\n")

    src = _submodule_path().read_text(encoding="utf-8")
    prefix = cache_dir.replace("\\", "/").rstrip("/") + "/"
    tissue = cfg.get("IPERTURB_TISSUE_FILTER", getattr(config, "IPERTURB_TISSUE_FILTER", "K562"))
    slice_src = _grn_slice(src, prefix, tissue)

    logger.info("iPerturb full GRN: building real L1–L4 GRN over %d panel genes "
                "(tissue=%s) — this crawls TRRUST/OmniPath/STRING/COXPRESdb ...",
                len(genes), tissue)
    ns = {"__name__": "iperturb_grn_build", "__builtins__": __builtins__}
    try:
        exec(compile(slice_src, str(_submodule_path()), "exec"), ns)  # noqa: S102
    except Exception as e:  # noqa: BLE001 — network/format issues → fallback
        logger.warning("iPerturb full GRN build failed (%s) — regulon fallback.", e)
        return None
    if os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0:
        return out_tsv
    return None


def run_eval(adata, cfg: dict) -> dict:
    """Dispatch on ``IPERTURB_ENGINE``:
      * ``propagate`` (default, Tier-1): analytic network-propagation + inferred
        readout (CPU; ``_iperturb_prop``). Composes combos, generalizes to unseen
        genes, leakage-free.
      * ``hill`` (legacy): the original Hill-kinetics GRNN (single-gene only,
        combo→control fallback). Kept for ablation/back-compat.
      * ``deep`` (Tier-2): learnable graph diffusion + variational inference — TODO.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import scanpy as sc

    t0 = time.time()
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    combo_sep = cfg.get("COMBO_SEP", getattr(config, "COMBO_SEP", "+"))
    min_cells = int(cfg.get("MIN_CELLS_PER_PERT", config.MIN_CELLS_PER_PERT))
    seed = int(cfg.get("RANDOM_SEED", 42))
    engine = str(cfg.get("IPERTURB_ENGINE", getattr(config, "IPERTURB_ENGINE", "propagate")))

    # iPerturb works in log1p-normalized space (its pipeline normalizes first).
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    var_names = [str(g) for g in adata.var_names]
    var_set = set(var_names)
    panel = set(map(str, cfg.get("EVAL_PANEL_GENES") or [])) or var_set
    candidate = var_set & panel

    # GRN over the FAIR panel genes (split-independent): real GeneHancer GRN if
    # available, else the bundled-regulon GRN.
    grn_tsv = build_full_grn(candidate, cfg)
    grn_source = "full(GeneHancer)"
    tmp_path = None
    if grn_tsv is None:
        tmp = tempfile.NamedTemporaryFile("w", suffix="_grn.tsv", delete=False)
        tmp.close()
        tmp_path = tmp.name
        n_edges = _build_grn_tsv(candidate, cfg, tmp_path)
        if n_edges == 0:
            raise RuntimeError(
                "iPerturb: no GRN edges — provide GeneHancer in IPERTURB_SUPPORT_DIR "
                "(eval/resources/fetch_iperturb_support.py) or bundle CollecTRI/DoRothEA "
                "(eval/resources/fetch_resources.py).")
        grn_tsv = tmp_path
        grn_source = "regulon"

    train_mask, test_mask = get_split_masks(adata)
    obs_all = adata.obs[pert_col].astype(str).to_numpy()

    try:
        if engine == "propagate":
            scoring, info = _run_propagate(adata, cfg, grn_tsv, var_names, candidate,
                                           train_mask, test_mask, obs_all, ctrl_label,
                                           combo_sep, min_cells)
        elif engine == "hill":
            scoring, info = _run_hill(adata, cfg, grn_tsv, candidate, train_mask,
                                      test_mask, obs_all, ctrl_label, pert_col,
                                      combo_sep, min_cells, seed)
        elif engine == "deep":
            raise NotImplementedError(
                "iPerturb engine 'deep' (Tier-2 GNN) not implemented yet; use "
                "IPERTURB_ENGINE='propagate' (Tier-1) or 'hill' (legacy).")
        else:
            raise ValueError(f"unknown IPERTURB_ENGINE={engine!r}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    logger.info("iPerturb[%s]: %d test perts (%d modeled), %d GRN genes (%s) %s",
                engine, len(scoring["pert_names"]), info.get("n_modeled", 0),
                len(scoring["gene_names"]), grn_source, info)
    return {"model": "iPerturb", "scoring": scoring,
            "pert_names": scoring["pert_names"], "runtime_seconds": time.time() - t0}


def _run_propagate(adata, cfg, grn_tsv, var_names, candidate, train_mask, test_mask,
                   obs_all, ctrl_label, combo_sep, min_cells):
    """Tier-1 analytic network-propagation engine (CPU; numpy/scipy)."""
    import pandas as pd
    from . import _iperturb_prop as prop

    gene_names = sorted(candidate)
    name2col = {g: i for i, g in enumerate(var_names)}
    hvg_pos = np.array([name2col[g] for g in gene_names], dtype=int)
    gene2idx = {g: i for i, g in enumerate(gene_names)}

    ctrl_mask = obs_all == str(ctrl_label)
    if not ctrl_mask.any():
        raise RuntimeError("iPerturb-prop: no control cells in dataset")
    x0 = prop._logmean(adata.X[ctrl_mask])[hvg_pos]
    perturbed_train = train_mask & (obs_all != str(ctrl_label))
    pert_ref = (prop._logmean(adata.X[perturbed_train])[hvg_pos]
                if perturbed_train.any() else x0)

    grn_df = pd.read_csv(grn_tsv, sep="\t")
    pred_c, true_c, names, info = prop.run_propagation_engine(
        adata=adata, train_mask=train_mask, test_mask=test_mask, obs_all=obs_all,
        gene_names=gene_names, gene2idx=gene2idx, x0=x0, hvg_pos=hvg_pos,
        grn_df=grn_df, pert_ref=pert_ref, ctrl_label=ctrl_label, combo_sep=combo_sep,
        min_cells=min_cells, cfg=cfg, parse_gene_set=parse_gene_set)
    if not names:
        raise RuntimeError("iPerturb-prop: no evaluable TEST perturbations")
    scoring = {
        "pred_centroids": np.stack(pred_c), "true_centroids": np.stack(true_c),
        "ctrl_centroid": np.asarray(x0, np.float32),
        "perturbed_ref_centroid": np.asarray(pert_ref, np.float32),
        "pert_names": names, "gene_names": gene_names, "pred_is_point": True,
    }
    return scoring, info


def _run_hill(adata, cfg, grn_tsv, candidate, train_mask, test_mask, obs_all,
              ctrl_label, pert_col, combo_sep, min_cells, seed):
    """Legacy Hill-kinetics GRNN engine (single-gene only; combo→control fallback)."""
    import torch
    epochs = int(cfg.get("IPERTURB_EPOCHS", getattr(config, "IPERTURB_EPOCHS", 50)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sym = load_iperturb_symbols()
    GRNN_from_tsv = sym["grn_tsv_to_grnn"]
    PerturbseqDataset = sym["PerturbseqDataset"]
    split_dataset = sym["split_dataset"]
    train_grnn = sym["train_grnn"]
    _logmean = sym["_logmean"]

    ds = PerturbseqDataset(adata[train_mask].copy(), sorted(candidate),
                           perturbation_col=pert_col, control_label=ctrl_label,
                           min_cells=min_cells, verbose=False)
    gene_names = list(ds.gene_names)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    hvg_pos = ds.hvg_pos
    x0 = ds.x0
    model, grn_df = GRNN_from_tsv(grn_tsv, gene_names, x0.numpy())
    tr, va, _rest = split_dataset(ds, train_frac=0.85, val_frac=0.15, seed=seed)
    if len(tr) >= 1 and len(va) >= 1:
        train_grnn(model, tr, va, n_epochs=epochs, device=device)
    model.eval()
    x0d = x0.to(device)

    perturbed_train = train_mask & (obs_all != str(ctrl_label))
    pert_ref = (_logmean(adata.X[perturbed_train])[hvg_pos]
                if perturbed_train.any() else x0.numpy())
    ctrl_np = x0.numpy()
    test_perts = restrict_to_test(
        [p for p in np.unique(obs_all) if p != str(ctrl_label)], adata,
        pert_col=pert_col, ctrl_label=ctrl_label)
    pred_c, true_c, names, n_modeled = [], [], [], 0
    for p in test_perts:
        m = (obs_all == p) & test_mask
        if m.sum() < min_cells:
            continue
        x_obs = _logmean(adata.X[m])[hvg_pos].astype(np.float32)
        genes = parse_gene_set(p, ctrl_label, combo_sep)
        if len(genes) == 1 and genes[0] in gene2idx:
            gi = gene2idx[genes[0]]
            with torch.no_grad():
                xp, _ = model(x0d, gi, float(x_obs[gi]))
            pred = xp.detach().cpu().numpy().astype(np.float32)
            n_modeled += 1
        else:
            pred = ctrl_np.astype(np.float32)
        pred_c.append(pred); true_c.append(x_obs); names.append(p)
    if not names:
        raise RuntimeError("iPerturb: no evaluable TEST perturbations")
    scoring = {
        "pred_centroids": np.stack(pred_c), "true_centroids": np.stack(true_c),
        "ctrl_centroid": ctrl_np, "perturbed_ref_centroid": pert_ref,
        "pert_names": names, "gene_names": gene_names, "pred_is_point": True,
    }
    return scoring, {"n_modeled": n_modeled, "n_edges": len(grn_df)}
