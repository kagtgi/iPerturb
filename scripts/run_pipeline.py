#!/usr/bin/env python3
"""
run_pipeline.py — Full iPerturb pipeline (GRN construction + GRNN training).

Runs build_grn → train_grnn for a single cell line.

Usage
-----
    python scripts/run_pipeline.py \\
        --h5ad K562.h5ad \\
        --cell-line K562 \\
        [--gh-gff GeneHancer_v5.26.gff] \\
        [--gh-tfbs GeneHancer_TFBSs_v5.26.txt] \\
        [--gh-tissue GeneHancer_Tissues_v5.26.txt] \\
        [--coxpresdb-zip Hsa_union_coex.zip] \\
        [--n-epochs 50] [--device cuda]
"""

import argparse
import logging
import sys
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="iPerturb full pipeline")

    # Data
    parser.add_argument("--h5ad",          required=True)
    parser.add_argument("--cell-line",     default="K562", choices=["K562", "RPE1"])
    parser.add_argument("--symbol-col",    default="gene_name")
    parser.add_argument("--perturb-col",   default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--min-cells",     type=int, default=5)
    parser.add_argument("--n-top-hvgs",    type=int, default=2_000)

    # GRN
    parser.add_argument("--cache-dir",         default="grn_cache")
    parser.add_argument("--grn-out",           default="grn_edges.tsv")
    parser.add_argument("--gh-gff",            default="")
    parser.add_argument("--gh-tfbs",           default="")
    parser.add_argument("--gh-tissue",         default="")
    parser.add_argument("--tissue-filter",     default="",
                        help="Override tissue filter (defaults to cell-line name)")
    parser.add_argument("--coxpresdb-zip",     default="")
    parser.add_argument("--skip-coxpresdb",    action="store_true")
    parser.add_argument("--string-min-score",  type=int,   default=700)
    parser.add_argument("--coex-topn",         type=int,   default=5)
    parser.add_argument("--greedy-reward",     type=float, default=0.15)
    parser.add_argument("--edge-param-budget", type=int,   default=22_000)

    # Training
    parser.add_argument("--n-epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max-iter",      type=int,   default=100)
    parser.add_argument("--train-frac",    type=float, default=0.70)
    parser.add_argument("--val-frac",      type=float, default=0.10)
    parser.add_argument("--lam-wmse",      type=float, default=1.0)
    parser.add_argument("--lam-afda",      type=float, default=1.0)
    parser.add_argument("--lam-delta",     type=float, default=1.0)
    parser.add_argument("--lam-deg",       type=float, default=2.0)
    parser.add_argument("--lam-balance",   type=float, default=0.5)
    parser.add_argument("--gamma-focus",   type=float, default=2.0)
    parser.add_argument("--top-k-deg",     type=int,   default=20)

    # Evaluation
    parser.add_argument("--n-eval-runs",    type=int,   default=10)
    parser.add_argument("--subsample-frac", type=float, default=0.30)
    parser.add_argument("--top-k-pearson",  type=int,   default=20)
    parser.add_argument("--lfc-threshold",  type=float, default=0.1)

    # Output
    parser.add_argument("--plot-dir",   default="grn_plots")
    parser.add_argument("--graphml-out",default="grn_full.graphml")
    parser.add_argument("--save-model", default="")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--log-level",  default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("iperturb")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── 1. Load data + HVG selection ─────────────────────────────────────────
    from iperturb.data import load_dataset, split_dataset
    dataset, gene_names = load_dataset(
        h5ad_path=args.h5ad,
        gene_names=[],
        perturbation_col=args.perturb_col,
        control_label=args.control_label,
        min_cells=args.min_cells,
        symbol_col=args.symbol_col,
        n_top_hvgs=args.n_top_hvgs,
    )
    x0_numpy = dataset.x0.numpy()

    # Save gene list for reference
    gene_list_path = Path(args.cache_dir) / "gene_list.txt"
    gene_list_path.parent.mkdir(parents=True, exist_ok=True)
    gene_list_path.write_text("\n".join(gene_names))
    log.info("✓ Gene list saved → %s", gene_list_path)

    # ── 2. Build GRN ──────────────────────────────────────────────────────────
    from iperturb.grn import build_grn
    tissue_filter = args.tissue_filter or args.cell_line
    selected, params_used = build_grn(
        gene_set=set(gene_names),
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
    selected.to_csv(args.grn_out, sep="\t", index=False)
    log.info("✓ GRN saved → %s  (%d edges, %d params)", args.grn_out, len(selected), params_used)

    if args.graphml_out:
        from iperturb.visualize import export_graphml
        export_graphml(selected, args.graphml_out)

    # ── 3. Build model ────────────────────────────────────────────────────────
    from iperturb.model import grn_tsv_to_grnn
    torch.manual_seed(args.seed)
    model, grn_df = grn_tsv_to_grnn(args.grn_out, gene_names, x0_numpy, max_iter=args.max_iter)

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = split_dataset(
        dataset, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
    )

    # ── 4. Pre-training heatmap ───────────────────────────────────────────────
    from iperturb.visualize import save_weight_heatmap, plot_loss_curves
    from iperturb.visualize import save_before_figure, save_after_figure, save_changed_figure
    eff_before = model.eff_map()
    save_weight_heatmap(eff_before, "before", args.cell_line, args.plot_dir)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    from iperturb.train import train_grnn, convergence_report, subsample_evaluate
    train_h, val_h = train_grnn(
        model, train_ds, val_ds,
        n_epochs=args.n_epochs, lr=args.lr,
        top_k_deg=args.top_k_deg,
        lam_wmse=args.lam_wmse, lam_afda=args.lam_afda,
        lam_delta=args.lam_delta, lam_deg=args.lam_deg,
        lam_balance=args.lam_balance, gamma_focus=args.gamma_focus,
        device=device,
    )

    # ── 6. Post-training heatmap + loss curve ─────────────────────────────────
    eff_after = model.eff_map()
    save_weight_heatmap(eff_after, "after", args.cell_line, args.plot_dir)
    plot_loss_curves(train_h, val_h, args.cell_line, args.plot_dir)

    # Convergence diagnostics
    x0_dev = dataset.x0.to(device)
    med_it, f_nc = convergence_report(model, test_ds, x0_dev, device)
    log.info("Convergence (test): median=%d iters | non-converged=%.1f%%", med_it, 100 * f_nc)

    # ── 7. Evaluation ─────────────────────────────────────────────────────────
    import numpy as np
    import pandas as pd

    all_metrics = subsample_evaluate(
        model, test_ds,
        n_runs=args.n_eval_runs,
        subsample_frac=args.subsample_frac,
        top_k_pearson=args.top_k_pearson,
        lfc_threshold=args.lfc_threshold,
        device=device,
    )

    k = args.top_k_pearson
    print(f"\n{'='*60}")
    print(f"  iPerturb — {args.cell_line}  ({args.n_eval_runs} eval runs)")
    print(f"{'='*60}")
    for key, label in [
        (f"pearson_delta{k}_mean",  f"(a) Pearson Δ{k}  "),
        ("pearson_delta_all_mean",   "(a) Pearson Δ-all "),
        ("centroid_accuracy",        "(b) Centroid acc  "),
        ("mse_delta",                "(c) MSE (delta)   "),
        ("directional_accuracy",     "(d) Directional   "),
    ]:
        vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if vals:
            print(f"  {label} : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    results_df = pd.DataFrame(all_metrics)
    results_path = f"{args.cell_line}_metrics_{args.n_eval_runs}runs.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    log.info("✓ Metrics → %s", results_path)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        log.info("✓ Model → %s", args.save_model)

    log.info("✓ Done.  Figures in %s/", args.plot_dir)


if __name__ == "__main__":
    sys.exit(main())
