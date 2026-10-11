"""Unique-target score tables for figures/tables (average across folds/seeds, then bootstrap over targets)."""
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path(__file__).parents[2] / "artifacts"
METRICS = ["pcc_c", "pcc_resid", "pcc", "top20_c", "rmse", "deg_ap", "cent_rank"]
LINES = ("K562_essential", "rpe1")
LINE_NAME = {"K562_essential": "K562", "rpe1": "RPE1"}


def unique_target_scores(line: str) -> pd.DataFrame:
    S = ART / "scores"
    g4 = pd.read_csv(S / f"g4_{line}.tsv", sep="\t")
    g4["mid"] = g4.model + "|" + g4.graph + "|" + g4.svd
    g5 = pd.read_csv(S / f"g5_{line}.tsv", sep="\t").groupby(["fold", "target", "graph"], as_index=False)[METRICS].mean()
    g5["mid"] = "B5_gcn|" + g5.graph + "|self"
    parts = [g4[["fold", "target", "mid"] + METRICS], g5[["fold", "target", "mid"] + METRICS]]
    gp = S / f"gears_{line}.tsv"
    if gp.exists():
        parts.append(pd.read_csv(gp, sep="\t")[["fold", "target", "mid"] + METRICS])
    for ext in ("biolord", "cpa", "scgpt", "state_zeroshot", "state_fewshot"):
        ep = S / f"{ext}_{line}.tsv"
        if ep.exists():
            parts.append(pd.read_csv(ep, sep="\t")[["fold", "target", "mid"] + METRICS])
    g12p = S / f"g12_{line}.tsv"
    if g12p.exists():
        g12 = pd.read_csv(g12p, sep="\t")
        g12 = g12[g12.model.str.startswith("prop[L1+L3]") | (g12.model == "prop[integrated]|nnls")].copy()
        g12["mid"] = g12.model
        for m in METRICS:
            if m not in g12:
                g12[m] = np.nan
        parts.append(g12[["fold", "target", "mid"] + METRICS])
    df = pd.concat(parts)
    return df.groupby(["mid", "target"], as_index=False)[METRICS].mean()


def mean_ci(values, n_boot=4000, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    bm = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return v.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)
