"""GRN construction subpackage."""

from .sources import (
    crawl_trrust,
    crawl_omnipath_robust,
    parse_genehancer,
    crawl_string_ppi,
    build_l3_transitive,
    parse_coxpresdb_bulk,
)
from .merge import merge_edges
from .select import greedy_select_connected

__all__ = [
    "crawl_trrust",
    "crawl_omnipath_robust",
    "parse_genehancer",
    "crawl_string_ppi",
    "build_l3_transitive",
    "parse_coxpresdb_bulk",
    "merge_edges",
    "greedy_select_connected",
    "build_grn",
]


def build_grn(
    gene_set: set,
    cache_dir: str,
    gh_gff_path: str = "",
    gh_tfbs_path: str = "",
    gh_tissue_path: str = "",
    tissue_filter: str = "",
    string_min_score: int = 700,
    coex_topn: int = 5,
    coxpresdb_zip: str = "",
    skip_coxpresdb: bool = False,
    greedy_reward: float = 0.15,
    level_conf: dict | None = None,
    edge_param_budget: int = 22_000,
):
    """
    End-to-end GRN construction pipeline.

    Runs all four evidence levels, merges the edge pool, and applies the
    parameter-budget greedy selection algorithm.

    Returns
    -------
    selected : pd.DataFrame
        Selected edges with columns: source, target, sign, level, db.
    params_used : int
        Number of edge parameters consumed.
    """
    import logging
    from pathlib import Path

    import pandas as pd

    if level_conf is None:
        level_conf = {1: 1.0, 2: 0.60, 3: 0.35, 4: 0.20}

    log = logging.getLogger("iperturb.grn")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _cached(name, fn):
        p = Path(cache_dir) / name
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            log.info("[cache] %s: %d edges", name, len(df))
            return df
        df = fn()
        if not df.empty:
            df.to_csv(p, sep="\t", index=False)
        return df

    l1_trrust    = _cached("l1_trrust.tsv",    lambda: crawl_trrust(gene_set))
    l1_omnipath  = _cached("l1_omnipath.tsv",  lambda: crawl_omnipath_robust(gene_set, ["tf_target", "collectri"], "OmniPath",  cache_dir))
    l1_collectri = _cached("l1_collectri.tsv", lambda: crawl_omnipath_robust(gene_set, ["collectri"],               "CollecTRI", cache_dir))

    l2_gh = pd.DataFrame(columns=["source", "target", "sign", "level", "db"])
    if gh_gff_path and gh_tfbs_path and gh_tissue_path:
        l2_gh = _cached("l2_genehancer.tsv",
                         lambda: parse_genehancer(gene_set, gh_gff_path, gh_tfbs_path,
                                                  gh_tissue_path, tissue_filter))

    ppi_raw = _cached("l3_ppi_raw.tsv",
                       lambda: crawl_string_ppi(gene_set, string_min_score))
    l3      = _cached("l3_transitive.tsv",
                       lambda: build_l3_transitive(l2_gh, ppi_raw))

    l4 = pd.DataFrame(columns=["source", "target", "sign", "level", "db"])
    if not skip_coxpresdb and coxpresdb_zip:
        l4 = _cached("l4_coex.tsv",
                      lambda: parse_coxpresdb_bulk(gene_set, top_n=coex_topn,
                                                   zip_path=coxpresdb_zip))

    pool = merge_edges([l1_trrust, l1_omnipath, l1_collectri, l2_gh, l3, l4])
    log.info("Pool total: %d edges", len(pool))

    selected, params_used = greedy_select_connected(
        pool,
        edge_param_budget=edge_param_budget,
        reward=greedy_reward,
        level_conf=level_conf,
    )
    return selected, params_used
