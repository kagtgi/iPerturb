"""Fig. 3: held-out-target prediction on K562 and RPE1 (unique-target means, 95% bootstrap CI)."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.viz.style import *
from ipt.viz.load import LINES, LINE_NAME, mean_ci, unique_target_scores

apply_style()
MODELS = [  # (mid, label, color)
    ("B0_nochange|-|-", "No change (B0)", GRED_L), ("B1_trainmean|-|-", "Train mean (B1)", GRED_L),
    ("B2_linear|-|-", "Linear low-rank (B2)", GRED), ("B4_gears|-|-", "GEARS (B4)", GRED),
    ("smooth_L1+L3|rewired|self", "Smoothing, rewired GRN", GGREY_L), ("smooth_dataknn|real|self", "Smoothing, data-kNN", GGREY_L),
    ("smooth_L3|real|self", "Smoothing, STRING only", GBLUE_L), ("smooth_L1+L3|real|self", "iPerturb: smoothing head", GBLUE),
    ("B5_gcn|real|self", "iPerturb: GCN head (B5)", GGREEN), ("nbrmean_L3|real|self", "iPerturb: neighbour-mean head", GYELLOW),
]
PANELS = [("pcc_c", "Centered Pearson"), ("pcc_resid", "Residualised Pearson"),
          ("top20_c", "Top-20 Pearson"), ("cent_rank", "Centroid rank")]
U = {l: unique_target_scores(l) for l in LINES}
have = {m for l in LINES for m in U[l].mid.unique()}
models = [m for m in MODELS if m[0] in have]
fig, axes = plt.subplots(2, 2, figsize=(122 / 25.4, 112 / 25.4), sharey=True)
for ax, (met, title) in zip(axes.ravel(), PANELS):
    for li, l in enumerate(LINES):
        for yi, (mid, lab, col) in enumerate(models):
            v = U[l][U[l].mid == mid][met]
            if v.empty: continue
            m, lo, hi = mean_ci(v)
            off = (-0.16, 0.16)[li]
            ax.errorbar(m, yi + off, xerr=[[m - lo], [hi - m]], fmt=("o", "s")[li], ms=3.6, color=col, ecolor=col, elinewidth=1,
                        mec="white" if li == 0 else col, mfc=col if li == 0 else "white", capsize=0)
    ax.set_title(title, fontsize=8.5, loc="left"); ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[1] for m in models], fontsize=7); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    if met == "cent_rank": ax.axvline(0.5, color=GGREY, lw=0.8, ls=":")
for ax, c in zip(axes.ravel(), "abcd"):
    ax.text(-0.02, 1.16, f"({c})", transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker="o", ls="", color=GGREY, ms=4), Line2D([], [], marker="s", ls="", mfc="white", color=GGREY, ms=4)],
           ["K562 (filled)", "RPE1 (open)"], loc="lower center", ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.8, w_pad=0.8)
Path("paper/figures").mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(f"paper/figures/fig3_generalization.{ext}")
print({LINE_NAME[l]: int(U[l].target.nunique()) for l in LINES})
