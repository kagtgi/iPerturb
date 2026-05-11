"""Merge multi-level GRN edge pools into a deduplicated DataFrame."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("iperturb.grn")


def merge_edges(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate edges from multiple evidence levels, then collapse duplicates.

    For each (source, target) pair:
      level → keep the minimum (highest confidence).
      sign  → keep the value if all non-null values agree; otherwise 0 (unknown).
      db    → join unique source names with a comma.

    Returns
    -------
    pd.DataFrame with columns: source, target, level, sign, db
    """
    valid = [d for d in dfs if d is not None and not d.empty]
    if not valid:
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    all_e = pd.concat(valid, ignore_index=True)

    def _sign_agg(x):
        vals = x.dropna()
        return int(vals.iloc[0]) if vals.nunique() == 1 else 0

    agg_dict = {
        "level": "min",
        "sign":  _sign_agg,
        "db":    lambda x: ",".join(sorted(x.unique())),
    }
    merged = all_e.groupby(["source", "target"], as_index=False).agg(agg_dict)
    log.info("Merged pool: %d unique (source, target) pairs", len(merged))
    return merged
