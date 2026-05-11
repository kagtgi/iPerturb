"""
Parameter-budget greedy edge selection for GRNN construction.

Rule-of-10 budget
─────────────────
  gene params   : 3 × N_genes                  = 6 000
  edge budget   : (N_obs/10) − gene_params      = 22 000

Sign-aware parameter cost
──────────────────────────
  sign ∈ {+1, −1}  →  2 params  (K_d, n;  sign fixed from DB)
  sign = 0         →  3 params  (K_d, n, s_ij; sign learned)

Algorithm
──────────
  Phase 0: Restrict pool to the Largest Connected Component (LCC).
  Phase 1: Kruskal spanning tree (connectivity guarantee).
  Phase 2: Reward-penalty greedy fill up to edge_param_budget.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("iperturb.grn")

_PARAMS_SIGNED   = 2
_PARAMS_UNSIGNED = 3


def _edge_cost(sign: int) -> int:
    return _PARAMS_SIGNED if sign != 0 else _PARAMS_UNSIGNED


def greedy_select_connected(
    edges: pd.DataFrame,
    edge_param_budget: int = 22_000,
    reward: float = 0.15,
    level_conf: dict | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Select GRN edges subject to a parameter budget.

    Parameters
    ----------
    edges             : merged edge pool (source, target, sign, level, db)
    edge_param_budget : maximum number of edge parameters (default 22 000)
    reward            : per-iteration accumulation reward for pending edges
    level_conf        : confidence weight per level {1:1.0, 2:0.60, 3:0.35, 4:0.20}

    Returns
    -------
    (selected_df, params_used)
    """
    if level_conf is None:
        level_conf = {1: 1.0, 2: 0.60, 3: 0.35, 4: 0.20}

    edges = edges.copy().reset_index(drop=True)
    signs = edges["sign"].to_numpy(dtype=int)
    costs = np.array([_edge_cost(s) for s in signs], dtype=int)

    # ── Phase 0: LCC restriction ──────────────────────────────────────────────
    G_pool = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.Graph())
    comps  = sorted(nx.connected_components(G_pool), key=len, reverse=True)

    if len(comps) > 1:
        dropped = set().union(*comps[1:])
        log.warning(
            "Pool has %d components → restricting to LCC (%d nodes). "
            "Dropping %d unreachable nodes: %s …",
            len(comps), len(comps[0]), len(dropped), sorted(dropped)[:8],
        )
        lcc   = comps[0]
        edges = edges[edges["source"].isin(lcc) & edges["target"].isin(lcc)].reset_index(drop=True)
        signs = edges["sign"].to_numpy(dtype=int)
        costs = np.array([_edge_cost(s) for s in signs], dtype=int)
        log.warning("Pool after LCC: %d edges, %d nodes.", len(edges), len(lcc))
    else:
        log.info("Pool connected (%d nodes). No repair needed.", len(comps[0]))

    # ── Arrays ────────────────────────────────────────────────────────────────
    conf_arr = edges["level"].map(level_conf).fillna(0.1).to_numpy(dtype=float)
    src_arr  = edges["source"].to_numpy()
    tgt_arr  = edges["target"].to_numpy()

    all_nodes = sorted(set(src_arr) | set(tgt_arr))
    N         = len(all_nodes)
    node2id   = {n: i for i, n in enumerate(all_nodes)}
    src_ids   = np.array([node2id[s] for s in src_arr], dtype=int)
    tgt_ids   = np.array([node2id[t] for t in tgt_arr], dtype=int)

    span_cost = sum(sorted([_edge_cost(s) for s in signs])[: N - 1])
    if span_cost > edge_param_budget:
        log.warning(
            "Spanning tree alone costs %d params > budget %d. "
            "Output may not be fully connected.", span_cost, edge_param_budget,
        )

    # ── Union-Find ────────────────────────────────────────────────────────────
    parent = np.arange(N, dtype=int)
    rnk    = np.zeros(N, dtype=int)

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return False
        if rnk[rx] < rnk[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rnk[rx] == rnk[ry]:
            rnk[rx] += 1
        return True

    remaining   = np.ones(len(edges), dtype=bool)
    selected    = []
    params_used = 0

    # ── Phase 1: Kruskal spanning tree ───────────────────────────────────────
    n_comp  = N
    bridged = 0
    for idx in np.argsort(-conf_arr):
        if n_comp <= 1:
            break
        cost = costs[idx]
        if params_used + cost > edge_param_budget:
            continue
        if _union(src_ids[idx], tgt_ids[idx]):
            selected.append(int(idx))
            remaining[int(idx)] = False
            params_used += cost
            n_comp      -= 1
            bridged     += 1

    if n_comp > 1:
        log.error("Phase 1: %d components remain after spanning tree.", n_comp)
    else:
        log.info("Phase 1: %d spanning-tree edges (%d params used) → connected ✓", bridged, params_used)

    # ── Phase 2: reward-penalty greedy ───────────────────────────────────────
    accum = np.zeros(len(edges), dtype=float)

    while params_used < edge_param_budget and remaining.any():
        accum[remaining] += reward

        best = int(np.argmax(np.where(remaining, conf_arr + accum, -np.inf)))
        cost = costs[best]

        if params_used + cost > edge_param_budget:
            signed_remaining = remaining & (costs == _PARAMS_SIGNED)
            if not signed_remaining.any():
                break
            best = int(np.argmax(np.where(signed_remaining, conf_arr + accum, -np.inf)))
            cost = costs[best]
            if params_used + cost > edge_param_budget:
                break

        selected.append(best)
        remaining[best] = False
        params_used    += cost

        accum[remaining & (src_arr == src_arr[best])] -= reward
        accum[remaining & (tgt_arr == tgt_arr[best])] -= reward

    # ── Build result ──────────────────────────────────────────────────────────
    result = edges.iloc[selected].drop(columns=["conf"], errors="ignore").copy()

    # ── Hard connectivity check ───────────────────────────────────────────────
    U      = nx.from_pandas_edgelist(result, "source", "target", create_using=nx.Graph())
    out_cc = sorted(nx.connected_components(U), key=len, reverse=True)
    if len(out_cc) > 1:
        raise RuntimeError(
            f"Output NOT connected ({len(out_cc)} components). "
            f"Sizes: {[len(c) for c in out_cc[:10]]}"
        )

    n_signed   = int((result["sign"] != 0).sum())
    n_unsigned = int((result["sign"] == 0).sum())
    z_actual   = len(result) / max(N, 1)

    log.info(
        "✓  %d edges | %d nodes | 1 component | %d params used / %d budget\n"
        "   signed (%d×2=%d params)  unsigned (%d×3=%d params)  z=%.2f",
        len(result), U.number_of_nodes(), params_used, edge_param_budget,
        n_signed,   n_signed   * _PARAMS_SIGNED,
        n_unsigned, n_unsigned * _PARAMS_UNSIGNED,
        z_actual,
    )
    return result, params_used
