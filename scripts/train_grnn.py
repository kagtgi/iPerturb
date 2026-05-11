#!/usr/bin/env python3
"""
train_grnn.py — Train and evaluate the GRNN for iPerturb.

Usage
-----
    python scripts/train_grnn.py \\
        --h5ad K562.h5ad \\
        --grn grn_edges.tsv \\
        --cell-line K562 \\
        --n-epochs 50 \\
        --lr 1e-3 \\
        --plot-dir grn_plots \\
        [--n-eval-runs 10] \\
        [--device cuda]

Outputs
-------
  grn_plots/<cell_line>_heatmap_before.*  — pre-training effective-weight heatmap
  grn_plots/<cell_line>_heatmap_after.*   — post-training effective-weight heatmap
  grn_plots/<cell_line>_loss_curves.*     — training / validation loss curves
  grn_plots/<cell_line>_<gene>_before.*   — per-gene subgraph (before)
  grn_plots/<cell_line>_<gene>_after.*    — per-gene subgraph (after)
  grn_plots/<cell_line>_<gene>_changed.*  — per-gene Δ-weight subgraph
  <cell_line>_metrics_<n>runs.tsv        — evaluation metrics
  <cell_line>_model.pt                   — saved model state dict
"""

import argparse
import logging
import sys
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="iPerturb GRNN training")
    parser.add_argument("--h5ad",          required=True)
    parser.add_argument("--grn",           required=True,  help="GRN edges TSV (from build_grn.py)")
    parser.add_argument("--cell-line",     default="K562", choices=["K562", "RPE1"])
    parser.add_argument("--symbol-col",    default="gene_name")
    parser.add_argument("--perturb-col",   default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--min-cells",     type=int,   default=5)
    parser.add_argument("--n-top-hvgs",    type=int,   default=2_000)
    parser.add_argument("--n-epochs",      type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--train-frac",    type=float, default=0.70)
    parser.add_argument("--val-frac",      type=float, default=0.10)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max-iter",      type=int,   default=100)
    parser.add_argument("--lam-wmse",      type=float, default=1.0)
    parser.add_argument("--lam-afda",      type=float, default=1.0)
    parser.add_argument("--lam-delta",     type=float, default=1.0)
    parser.add_argument("--lam-deg",       type=float, default=2.0)
    parser.add_argument("--lam-balance",   type=float, default=0.5)
    parser.add_argument("--gamma-focus",   type=float, default=2.0)
    parser.add_argument("--top-k-deg",     type=int,   default=20)
    parser.add_argument("--n-eval-runs",   type=int,   default=10)
    parser.add_argument("--subsample-frac",type=float, default=0.30)
    parser.add_argument("--top-k-pearson", type=int,   default=20)
    parser.add_argument("--lfc-threshold", type=float, default=0.1)
    parser.add_argument("--plot-dir",      default="grn_plots")
    parser.add_argument("--viz-genes",     nargs="*",  default=[],
                        help="Gene symbols to produce per-gene subgraph panels")
    parser.add_argument("--device",        default=None)
    parser.add_argument("--save-model",    default="",
                        help="Path to save trained model state dict (.pt)")
    parser.add_argument("--log-level",     default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("iperturb")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────────
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

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = split_dataset(
        dataset, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
    )
    log.info("Dataset split: train %d / val %d / test %d",
             len(train_ds), len(val_ds), len(test_ds))

    # ── Build model ───────────────────────────────────────────────────────────
    from iperturb.model import grn_tsv_to_grnn
    torch.manual_seed(args.seed)
    model, grn_df = grn_tsv_to_grnn(args.grn, gene_names, x0_numpy, max_iter=args.max_iter)

    # ── Before heatmap ────────────────────────────────────────────────────────
    from iperturb.visualize import save_weight_heatmap
    eff_before = model.eff_map()
    log.info("Generating pre-training heatmap …")
    save_weight_heatmap(eff_before, "before", args.cell_line, args.plot_dir)

    # ── Train ─────────────────────────────────────────────────────────────────
    from iperturb.train import train_grnn, convergence_report
    train_h, val_h = train_grnn(
        model, train_ds, val_ds,
        n_epochs=args.n_epochs, lr=args.lr,
        top_k_deg=args.top_k_deg,
        lam_wmse=args.lam_wmse, lam_afda=args.lam_afda,
        lam_delta=args.lam_delta, lam_deg=args.lam_deg,
        lam_balance=args.lam_balance, gamma_focus=args.gamma_focus,
        device=device,
    )

    # ── After heatmap + loss curve ────────────────────────────────────────────
    from iperturb.visualize import plot_loss_curves
    eff_after = model.eff_map()
    log.info("Generating post-training heatmap …")
    save_weight_heatmap(eff_after, "after", args.cell_line, args.plot_dir)
    plot_loss_curves(train_h, val_h, args.cell_line, args.plot_dir)

    # ── Convergence on test set ───────────────────────────────────────────────
    x0_dev = dataset.x0.to(device)
    med_it, f_nc = convergence_report(model, test_ds, x0_dev, device)
    log.info("Convergence (test): median=%d iters | non-converged=%.1f%%",
             med_it, 100 * f_nc)
    if f_nc > 0.05:
        log.warning(">5%% of samples did not converge — consider increasing --max-iter")

    # ── Per-gene subgraph panels ──────────────────────────────────────────────
    from iperturb.visualize import save_before_figure, save_after_figure, save_changed_figure
    for gene in args.viz_genes:
        log.info("Per-gene plots: %s", gene)
        save_before_figure(grn_df, gene, args.cell_line, eff_before, eff_after, args.plot_dir)
        save_after_figure(grn_df, gene, args.cell_line, eff_before, eff_after, model, args.plot_dir)
        save_changed_figure(grn_df, gene, args.cell_line, eff_before, eff_after, args.plot_dir)

    # ── Evaluation ────────────────────────────────────────────────────────────
    import numpy as np
    import pandas as pd
    from iperturb.train import subsample_evaluate

    all_metrics = subsample_evaluate(
        model, test_ds,
        n_runs=args.n_eval_runs,
        subsample_frac=args.subsample_frac,
        top_k_pearson=args.top_k_pearson,
        lfc_threshold=args.lfc_threshold,
        device=device,
    )

    k = args.top_k_pearson
    metric_keys = [
        (f"pearson_delta{k}_mean",  f"(a) Pearson Δ{k}  "),
        ("pearson_delta_all_mean",   "(a) Pearson Δ-all "),
        ("centroid_accuracy",        "(b) Centroid acc  "),
        ("mse_delta",                "(c) MSE (delta)   "),
        ("directional_accuracy",     "(d) Directional   "),
    ]
    print(f"\n{'='*60}")
    print(f"  iPerturb — {args.cell_line}  ({args.n_eval_runs} random subsample eval runs)")
    print(f"  Subsample fraction = {args.subsample_frac:.2f}")
    print(f"{'='*60}")
    for key, label in metric_keys:
        vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if vals:
            print(f"  {label} : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    results_df   = pd.DataFrame(all_metrics)
    results_path = f"{args.cell_line}_metrics_{args.n_eval_runs}runs.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    log.info("✓ Metrics saved → %s", results_path)

    # ── Save model ────────────────────────────────────────────────────────────
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        log.info("✓ Model saved → %s", args.save_model)

    log.info("✓ All figures saved to %s/", args.plot_dir)


if __name__ == "__main__":
    sys.exit(main())
