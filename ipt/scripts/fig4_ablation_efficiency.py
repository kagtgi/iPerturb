"""Fig. 4: (a) Hill vs linear gate, (b) evidence-layer ablations (+context-specific L2), (c) shared-Hill efficiency, (d) per-edge shapes memory."""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.metrics import paired_bootstrap
from ipt.viz.style import *
from ipt.viz.load import LINES, LINE_NAME, unique_target_scores

apply_style(); ART = Path(__file__).parents[1] / "artifacts"; S = ART / "scores"
fig, axes = plt.subplots(2, 2, figsize=(122 / 25.4, 112 / 25.4))
CL = {"K562_essential": GBLUE, "rpe1": GGREEN}


def dotci(ax, y, d, col, marker, label=None):
    ax.errorbar(d["mean_diff"], y, xerr=[[d["mean_diff"] - d["ci_low"]], [d["ci_high"] - d["mean_diff"]]], fmt=marker, ms=3.6, color=col, elinewidth=1, label=label, capsize=0)


# (a) G2 gate on reachable targets
ax = axes[0, 0]
labs = [("H2_hill_regk", "real", "B3_linear", "real", "Hill vs linear\n(same graph)"), ("H2_hill_regk", "real", "B2_linear", "-", "Hill vs no-graph\nlinear (B2)"),
        ("H2_hill_regk", "real", "H2_hill_regk", "rewired", "real vs rewired\ngraph (Hill)")]
for li, (line, f) in enumerate((("K562_essential", "g2all_K562_essential"), ("rpe1", "g2_rpe1"))):
    d = pd.read_csv(S / f"{f}.tsv", sep="\t"); R = d[d.reachable]
    pt = lambda m, g: R[(R.model == m) & (R.graph == g)].groupby("target").pcc_c.mean()
    for yi, (ma, ga, mb, gb, lab) in enumerate(labs):
        a_, b_ = pt(ma, ga), pt(mb, gb); j = a_.index.intersection(b_.index)
        r = paired_bootstrap(a_.loc[j].to_numpy(), b_.loc[j].to_numpy(), 10000, 0)
        dotci(ax, yi + (-0.15, 0.15)[li], r, CL[line], ("o", "s")[li], f"{LINE_NAME[line]} (n={len(j)})" if yi == 0 else None)
ax.axvline(0, color=GGREY, lw=0.8, ls=":"); ax.set_yticks(range(3)); ax.set_yticklabels([l[4] for l in labs], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("Δρ$_c$ on reachable targets"); ax.legend(fontsize=6.5, loc="lower right", frameon=False); ax.set_title("Regulatory Hill channel", fontsize=8.5, loc="left")
# (b) evidence layers
ax = axes[0, 1]
rows = [("smooth_L3|real|self", "STRING"), ("smooth_L1+L3|real|self", "L1+L3 GRN"), ("smooth_L1+L3|rewired|self", "rewired"), ("smooth_dataknn|real|self", "data kNN"),
        ("smooth_L1+L3|real|noself", "self-zeroed")]
for li, line in enumerate(LINES):
    U = unique_target_scores(line); base = U[U.mid == "smooth_none|real|self"].set_index("target").pcc_c
    for yi, (mid, lab) in enumerate(rows):
        x = U[U.mid == mid].set_index("target").pcc_c; j = x.index.intersection(base.index)
        dotci(ax, yi + (-0.15, 0.15)[li], paired_bootstrap(x.loc[j].to_numpy(), base.loc[j].to_numpy(), 10000, 0), CL[line], ("o", "s")[li])
ax.axvline(0, color=GGREY, lw=0.8, ls=":"); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("Δρ$_c$ vs no-graph linear"); ax.set_title("Evidence layers", fontsize=8.5, loc="left")
# (c) shared-Hill step time, A100, assoc graph
ax = axes[1, 0]; g = json.load(open(ART / "bench_gpu.json")); t = pd.DataFrame(g["timing"]); t = t[(t.graph == "assoc")]
impls = [("E0_exact_edge", "edge gather\n(exact)", GRED), ("E1_exact_spmm", "sparse\nmatmul", GBLUE), ("E2_exact_dense", "dense\nmatmul", GBLUE_L), ("C1_cheb16_spmm", "Cheb-16\n+ sparse", GPURPLE)]
w = 0.2
for bi, B in enumerate((32, 128, 512)):
    for ii, (k, lab, col) in enumerate(impls):
        v = t[(t.B == B) & (t.impl == k)].sec.iloc[0] * 1e3
        ax.bar(bi + (ii - 1.5) * w, v, w * 0.92, color=col, label=lab if bi == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels(["B=32", "B=128", "B=512"]); ax.set_ylabel("ms / fwd+bwd step (A100)"); ax.set_yscale("log")
ax.set_ylim(40, 12000); ax.legend(fontsize=6, ncol=2, loc="upper left", frameon=False, columnspacing=0.8, handlelength=1.2); ax.set_title("Shared Hill, 247k-edge graph", fontsize=8.5, loc="left")
# (d) per-edge shapes memory
ax = axes[1, 1]; p = json.load(open(ART / "bench_peredge_gpu.json")); t = pd.DataFrame(p["timing"]); t = t[t.graph == "assoc"]
for r_, ls in ((8, "-"), (16, "--")):
    for impl, col, lab in (("edge", GRED, "edge-wise"), ("spmm", GBLUE, "matmul")):
        d = t[(t.r == r_) & (t.impl == impl)].sort_values("B"); ok = d[~d.mem_mb.isna()]
        ax.plot(ok.B, ok.mem_mb / 1024, ls, color=col, marker="o", ms=3, label=f"{lab}, r={r_}")
        for _, row in d[d.mem_mb.isna()].iterrows():
            ax.plot(row.B, 40, "x", color=col, ms=6); 
ax.axhline(40, color=GGREY, lw=0.8, ls=":"); ax.set_xscale("log", base=2); ax.set_xticks([32, 128, 512]); ax.set_xticklabels(["32", "128", "512"]); ax.set_ylim(0, 66)
ax.set_xlabel("batch size B"); ax.set_ylabel("peak GPU memory (GB)"); ax.legend(fontsize=6, ncol=2, loc="upper left", frameon=False, columnspacing=0.8, handlelength=1.4); ax.set_title("Per-edge shapes, 247k edges", fontsize=8.5, loc="left")
for a_, c in zip(axes.ravel(), "abcd"): a_.text(-0.02, 1.13, f"({c})", transform=a_.transAxes, fontweight="bold", fontsize=10, va="bottom")
fig.tight_layout(h_pad=1.6, w_pad=1.0)
for ext in ("pdf", "png"): fig.savefig(f"paper/figures/fig4_ablation_efficiency.{ext}")
