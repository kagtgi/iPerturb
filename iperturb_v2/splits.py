"""
Shared train/test split logic for fair perturbation-prediction benchmarking.

A split is assigned **once** in ``eval_runner.run()`` (right after the shared
stratified subsample) and persisted in ``adata.obs`` so every model subprocess
reads an identical partition.  This mirrors the role of ``eval.sampling`` for
cell subsampling: centralization is what makes the comparison fair.

Two obs columns are written:
  * ``split_role`` ∈ {train, test, control, excluded} — honest, for reporting.
  * ``split``      ∈ {train, test, excluded}          — control folded into
    ``train`` so "data a model may fit on" is the single mask ``split == 'train'``.

Generalizability regimes (``SPLIT_MODE``) cover Level 6 of the metric taxonomy:
  * ``none``           — no holdout; every perturbation is train and scored
                         (back-compat: reproduces the legacy all-perts behavior).
  * ``random`` /
    ``unseen_pert``    — hold out a fraction of perturbations (all contexts seen).
  * ``simulation``     — Systema/GEARS: hold out combinatorial perturbations while
                         keeping their constituent singles in train.
  * ``unseen_context`` — hold out a cell context (``CONTEXT_COL``), all perts seen.
  * ``unseen_both``    — hold out perturbations AND contexts jointly (hardest);
                         off-diagonal cells are excluded from both train and test.
"""

from __future__ import annotations

import logging

import numpy as np

from . import config

logger = logging.getLogger(__name__)

VALID_MODES = (
    "none", "random", "simulation",
    "unseen_pert", "unseen_context", "unseen_both",
)


# ───────────────────────────────────────────────────────────────────────────
# Perturbation-label parsing
# ───────────────────────────────────────────────────────────────────────────

def parse_gene_set(
    cond: str,
    ctrl_label: str | None = None,
    combo_sep: str | None = None,
) -> list[str]:
    """Parse a perturbation label into its constituent perturbed genes.

    Splits on ``combo_sep`` and drops control tokens.  Consistent with the
    GEARS condition convention (``gears.py`` maps ctrl→"ctrl", gene→"<gene>+ctrl").

    Examples (ctrl_label="non-targeting", combo_sep="+")::

        "STAT1"            -> ["STAT1"]
        "GENEA+GENEB"      -> ["GENEA", "GENEB"]
        "GENE+ctrl"        -> ["GENE"]
        "ctrl+GENE"        -> ["GENE"]
        "non-targeting"    -> []        (pure control)
    """
    ctrl_label = ctrl_label if ctrl_label is not None else config.CTRL_LABEL
    combo_sep = combo_sep if combo_sep is not None else getattr(config, "COMBO_SEP", "+")
    drop = {"ctrl", str(ctrl_label)}
    genes = []
    for tok in str(cond).split(combo_sep):
        tok = tok.strip()
        if tok and tok not in drop:
            genes.append(tok)
    return genes


# ───────────────────────────────────────────────────────────────────────────
# Test-set selection per mode
# ───────────────────────────────────────────────────────────────────────────

def _choose_test_perts(perts, mode, test_frac, rng, combo_sep):
    """Return the list of perturbation labels assigned to the test set."""
    perts = sorted(perts)
    if not perts:
        return []
    if mode in ("random", "unseen_pert"):
        k = max(1, round(len(perts) * test_frac))
        return rng.choice(perts, size=min(k, len(perts)), replace=False).tolist()
    if mode == "simulation":
        singles = [p for p in perts if combo_sep not in p]
        combos = [p for p in perts if combo_sep in p]
        if combos:
            k = max(1, round(len(combos) * test_frac))
            chosen = rng.choice(combos, size=min(k, len(combos)), replace=False).tolist()
            train_singles = set(singles)
            # Keep only test combos with >=1 constituent single in train.
            kept = [c for c in chosen if any(g in train_singles for g in c.split(combo_sep))]
            if kept:
                return kept
            logger.warning("simulation split: no test combo has a train single; "
                           "falling back to all chosen combos.")
            return chosen
        # Single-gene-only dataset: degrade to random single hold-out.
        logger.warning("simulation split: dataset has no combinatorial perturbations; "
                       "falling back to random single-gene hold-out.")
        k = max(1, round(len(singles) * test_frac))
        return rng.choice(singles, size=min(k, len(singles)), replace=False).tolist()
    return []


# ───────────────────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────────────────

def assign_split(
    adata,
    *,
    mode: str | None = None,
    test_frac: float | None = None,
    seed: int | None = None,
    pert_col: str | None = None,
    ctrl_label: str | None = None,
    combo_sep: str | None = None,
    context_col: str | None = None,
):
    """Annotate ``adata.obs`` with a seeded, perturbation-level train/test split.

    Returns the same ``adata`` (mutated in place) with ``obs['split']`` and
    ``obs['split_role']``.  Idempotent for ``mode='none'``.  All cells of a given
    perturbation share a label (no perturbation leaks across train/test).
    """
    mode = mode if mode is not None else getattr(config, "SPLIT_MODE", "none")
    test_frac = test_frac if test_frac is not None else getattr(config, "TEST_FRAC", 0.20)
    seed = seed if seed is not None else config.RANDOM_SEED
    pert_col = pert_col or config.PERT_COL
    ctrl_label = ctrl_label if ctrl_label is not None else config.CTRL_LABEL
    combo_sep = combo_sep if combo_sep is not None else getattr(config, "COMBO_SEP", "+")
    context_col = context_col if context_col is not None else getattr(config, "CONTEXT_COL", None)

    if mode not in VALID_MODES:
        raise ValueError(f"Unknown SPLIT_MODE '{mode}'. Valid: {VALID_MODES}")

    rng = np.random.default_rng(seed)
    obs_pert = adata.obs[pert_col].astype(str)
    is_ctrl = (obs_pert == str(ctrl_label)).to_numpy()
    perts = [p for p in obs_pert.unique() if p != str(ctrl_label)]

    n = adata.n_obs
    role = np.array(["train"] * n, dtype=object)

    needs_context = mode in ("unseen_context", "unseen_both")
    has_context = bool(context_col) and context_col in adata.obs.columns
    if needs_context and not has_context:
        logger.warning("SPLIT_MODE='%s' needs CONTEXT_COL='%s' in obs but it is "
                       "absent — falling back to 'random'.", mode, context_col)
        mode = "random"
        needs_context = False

    if mode == "none":
        # Legacy behavior: no held-out split. Deliberately write NO split column
        # so get_split_masks() uses its fallback (train = all cells, test = all
        # non-control perts) — i.e. every perturbation is scored, exactly as before.
        logger.info("Split mode 'none': all perturbations scored (no held-out test).")
        return adata
    if mode in ("random", "unseen_pert", "simulation"):
        test_perts = set(_choose_test_perts(perts, mode, test_frac, rng, combo_sep))
        role = np.where(is_ctrl, "control",
                        np.where(obs_pert.isin(test_perts).to_numpy(), "test", "train"))
    elif mode == "unseen_context":
        ctx = adata.obs[context_col].astype(str)
        contexts = sorted(ctx.unique())
        k = max(1, round(len(contexts) * test_frac))
        test_ctx = set(rng.choice(contexts, size=min(k, len(contexts)), replace=False).tolist())
        in_test_ctx = ctx.isin(test_ctx).to_numpy()
        role = np.where(is_ctrl, "control",
                        np.where(in_test_ctx, "test", "train"))
    elif mode == "unseen_both":
        ctx = adata.obs[context_col].astype(str)
        contexts = sorted(ctx.unique())
        kp = max(1, round(len(perts) * test_frac))
        kc = max(1, round(len(contexts) * test_frac))
        test_perts = set(rng.choice(sorted(perts), size=min(kp, len(perts)), replace=False).tolist())
        test_ctx = set(rng.choice(contexts, size=min(kc, len(contexts)), replace=False).tolist())
        in_tp = obs_pert.isin(test_perts).to_numpy()
        in_tc = ctx.isin(test_ctx).to_numpy()
        # test = unseen pert AND unseen context; train = seen pert AND seen context;
        # off-diagonal cells are excluded from both (standard scPerturBench protocol).
        role = np.full(n, "excluded", dtype=object)
        role[in_tp & in_tc] = "test"
        role[(~in_tp) & (~in_tc)] = "train"
        role[is_ctrl] = "control"

    role = np.asarray(role, dtype=object)
    # split: control folds into train; excluded stays excluded.
    split = np.where(role == "control", "train", role)

    adata.obs["split_role"] = role
    adata.obs["split"] = split
    logger.info("Split (%s): %s", mode, split_summary(adata, pert_col=pert_col))
    return adata


# ───────────────────────────────────────────────────────────────────────────
# Consumers / helpers
# ───────────────────────────────────────────────────────────────────────────

def get_split_masks(adata, split_col: str = "split"):
    """Return ``(train_mask, test_mask)`` boolean arrays.

    Fallback-aware: if ``split_col`` is absent, behaves like today's pipeline
    (every cell is usable as "train"; every non-control cell is "test")."""
    if split_col not in adata.obs.columns:
        n = adata.n_obs
        train_mask = np.ones(n, dtype=bool)
        ctrl_label = config.CTRL_LABEL
        pert_col = config.PERT_COL
        if pert_col in adata.obs.columns:
            test_mask = (adata.obs[pert_col].astype(str) != str(ctrl_label)).to_numpy()
        else:
            test_mask = np.ones(n, dtype=bool)
        return train_mask, test_mask
    split = adata.obs[split_col].astype(str).to_numpy()
    return split == "train", split == "test"


def restrict_to_test(
    perts,
    adata,
    *,
    pert_col: str | None = None,
    ctrl_label: str | None = None,
) -> list[str]:
    """Filter ``perts`` to those present in the test split.

    If no test split exists (legacy/``none`` mode), returns ``perts`` unchanged
    so existing zero-shot behavior is preserved.
    """
    pert_col = pert_col or config.PERT_COL
    ctrl_label = ctrl_label if ctrl_label is not None else config.CTRL_LABEL
    if "split" not in adata.obs.columns:
        return list(perts)
    obs_pert = adata.obs[pert_col].astype(str)
    split = adata.obs["split"].astype(str)
    test_perts = set(obs_pert[split == "test"].unique()) - {str(ctrl_label)}
    if not test_perts:
        return list(perts)
    return [p for p in perts if str(p) in test_perts]


def split_summary(adata, *, pert_col: str | None = None) -> dict:
    """Summary dict for logging / leaderboard annotation."""
    pert_col = pert_col or config.PERT_COL
    out: dict = {}
    if "split" in adata.obs.columns:
        out["cells"] = adata.obs["split"].astype(str).value_counts().to_dict()
    if "split_role" in adata.obs.columns:
        obs_pert = adata.obs[pert_col].astype(str)
        role = adata.obs["split_role"].astype(str)
        test_perts = sorted(set(obs_pert[role == "test"].unique()))
        train_perts = sorted(set(obs_pert[role == "train"].unique()))
        out["n_test_perts"] = len(test_perts)
        out["n_train_perts"] = len(train_perts)
        # matching-mean degenerates to non-control mean when no held-out pert
        # shares a constituent single gene with the train set.
        combo_sep = getattr(config, "COMBO_SEP", "+")
        train_singles = {
            g for p in train_perts for g in parse_gene_set(p, combo_sep=combo_sep)
            if combo_sep not in p
        }
        degenerate = all(
            not any(g in train_singles for g in parse_gene_set(p, combo_sep=combo_sep))
            for p in test_perts
        ) if test_perts else False
        out["matching_degenerate"] = bool(degenerate)
    return out
