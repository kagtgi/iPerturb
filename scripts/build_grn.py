#!/usr/bin/env python3
"""
build_grn.py — GRN construction pipeline for iPerturb.

Usage
-----
    python scripts/build_grn.py \\
        --gene-list gene_list.txt \\
        --cache-dir grn_cache \\
        --out grn_edges.tsv \\
        --cell-line K562 \\
        [--gh-gff GeneHancer_v5.26.gff] \\
        [--gh-tfbs GeneHancer_TFBSs_v5.26.txt] \\
        [--gh-tissue GeneHancer_Tissues_v5.26.txt] \\
        [--coxpresdb-zip Hsa_union_coex.zip] \\
        [--string-min-score 700] \\
        [--coex-topn 5]

Outputs
-------
  <out>           — selected GRN edges TSV (source, target, sign, level, db)
  grn_full.graphml — full GRN for Cytoscape desktop
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="iPerturb GRN construction")
    parser.add_argument("--gene-list",       required=True,  help="Gene list file (one symbol per line)")
    parser.add_argument("--cache-dir",       default="grn_cache")
    parser.add_argument("--out",             default="grn_edges.tsv")
    parser.add_argument("--cell-line",       default="K562", choices=["K562", "RPE1"])
    parser.add_argument("--gh-gff",          default="")
    parser.add_argument("--gh-tfbs",         default="")
    parser.add_argument("--gh-tissue",       default="")
    parser.add_argument("--tissue-filter",   default="",  help="Substring match for GeneHancer tissue filter")
    parser.add_argument("--coxpresdb-zip",   default="")
    parser.add_argument("--skip-coxpresdb",  action="store_true")
    parser.add_argument("--string-min-score",type=int, default=700)
    parser.add_argument("--coex-topn",       type=int, default=5)
    parser.add_argument("--greedy-reward",   type=float, default=0.15)
    parser.add_argument("--edge-param-budget",type=int, default=22_000)
    parser.add_argument("--graphml-out",     default="grn_full.graphml")
    parser.add_argument("--log-level",       default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("iperturb")

    # Load gene set
    with open(args.gene_list) as f:
        gene_set = {line.strip().split()[0].upper() for line in f if line.strip()}
    log.info("Gene set: %d genes loaded from %s", len(gene_set), args.gene_list)

    tissue_filter = args.tissue_filter or args.cell_line

    from iperturb.grn import build_grn
    selected, params_used = build_grn(
        gene_set=gene_set,
        cache_dir=args.cache_dir,
        gh_gff_path=args.gh_gff,
        gh_tfbs_path=args.gh_tfbs,
        gh_tissue_path=args.gh_tissue,
        tissue_filter=tissue_filter,
        string_min_score=args.string_min_score,
        coex_topn=args.coex_topn,
        coxpresdb_zip=args.coxpresdb_zip,
        skip_coxpresdb=args.skip_coxpresdb,
        greedy_reward=args.greedy_reward,
        edge_param_budget=args.edge_param_budget,
    )

    selected.to_csv(args.out, sep="\t", index=False)
    log.info("✓ GRN edges saved → %s  (%d edges, %d params used)",
             args.out, len(selected), params_used)

    n_s = int((selected["sign"] != 0).sum())
    n_u = len(selected) - n_s
    print(f"\n=== Final GRN ===")
    print(f"  Edges total    : {len(selected):,}")
    print(f"  Signed  (±1)   : {n_s:,}  →  {n_s*2:,} params")
    print(f"  Unsigned (0)   : {n_u:,}  →  {n_u*3:,} params")
    print(f"  Edge params    : {params_used:,} / {args.edge_param_budget:,}")

    if args.graphml_out:
        from iperturb.visualize import export_graphml
        export_graphml(selected, args.graphml_out)
        log.info("✓ GraphML saved → %s", args.graphml_out)


if __name__ == "__main__":
    sys.exit(main())
