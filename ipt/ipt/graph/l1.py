"""Level-1 curated TF->target edges from cached CollecTRI / OmniPath / TRRUST tables (no network).

Sign policy (plan 4.1): sign 0 is never promoted to +1. Edges whose sources disagree on sign get sign 0
and are excluded from the signed core. Provenance = set of databases that support the edge.
"""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path


def _omni(path: Path, label: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    st = d["is_stimulation"].fillna(False).astype(bool); inh = d["is_inhibition"].fillna(False).astype(bool)
    sign = np.where(st & ~inh, 1, np.where(inh & ~st, -1, 0))
    return pd.DataFrame({"source": d.source_genesymbol.astype(str).str.upper(),
                         "target": d.target_genesymbol.astype(str).str.upper(), "sign": sign, "db": label})


def _trrust(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", header=None, names=["s", "t", "mode", "pmid"])
    sign = np.where(d["mode"].str.contains("Activation", case=False), 1,
                    np.where(d["mode"].str.contains("Repression", case=False), -1, 0))
    return pd.DataFrame({"source": d.s.str.upper(), "target": d.t.str.upper(), "sign": sign, "db": "TRRUST"})


def load_l1(cache: str | Path) -> pd.DataFrame:
    cache = Path(cache)
    raw = pd.concat([_omni(cache / "omnipath_collectri.tsv", "CollecTRI"),
                     _omni(cache / "omnipath_tf_target.tsv", "OmniPath"), _trrust(cache / "trrust.tsv")],
                    ignore_index=True)
    raw = raw[raw.source != raw.target]
    g = raw.groupby(["source", "target"])
    out = g.agg(dbs=("db", lambda x: ",".join(sorted(set(x)))), n_db=("db", "nunique")).reset_index()
    pos = g["sign"].apply(lambda x: (x == 1).any()).values; neg = g["sign"].apply(lambda x: (x == -1).any()).values
    out["sign"] = np.where(pos & ~neg, 1, np.where(neg & ~pos, -1, 0))
    out["sign_conflict"] = pos & neg
    return out
