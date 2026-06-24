"""Figure 3 (regenerated, Google style): four panels comparing iPerturb to five
baselines on K562 and RPE1. Grouped bars (a,b,d) with visible error bars + a
red->amber->green heatmap (c). Methods keep a fixed colour across panels
(iPerturb=blue=proposed, STATE=red=main competitor). Reads figs/fig3_values.csv.
Outputs figure3_combined.{pdf,png}."""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, "figs")
from style_google import (apply_style, mono_ticks, GREY300, GREY500, GREY700, GREY900,
                          GBLUE_L, GRED_L, GYELLOW_L, GGREEN_L, GPURPLE_L, GGREY_L,
                          SEQ_BAD_GOOD_L)
apply_style()

METHODS = ["CPA", "GEARS", "scGPT", "STATE", "Cell2Sentence", "iPerturb"]
COLOR = {"CPA": GPURPLE_L, "GEARS": GYELLOW_L, "scGPT": GGREEN_L, "STATE": GRED_L,
         "Cell2Sentence": GGREY_L, "iPerturb": GBLUE_L}
CELLS = ["K562", "RPE1"]

D = {}
with open("figs/fig3_values.csv") as f:
    for r in csv.DictReader(f):
        D[(r["metric"], r["cell"], r["method"])] = (float(r["mean"]), float(r["sd"]))

def grouped(ax, metric, title, letter, ylim=None):
    n = len(METHODS); w = 0.13
    for i, m in enumerate(METHODS):
        xs = [g + (i - (n-1)/2)*w for g in range(len(CELLS))]
        ys = [D[(metric, c, m)][0] for c in CELLS]
        es = [D[(metric, c, m)][1] for c in CELLS]
        ax.bar(xs, ys, w, yerr=es, label=m, color=COLOR[m],
               edgecolor=(GREY900 if m == "iPerturb" else GREY500),
               linewidth=(1.0 if m == "iPerturb" else 0.5),
               error_kw=dict(elinewidth=0.7, capsize=1.6, ecolor=GREY700))
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels(CELLS)
    ax.set_title(title, loc="center", fontsize=8.5)
    ax.set_axisbelow(True); ax.grid(axis="x", visible=False)
    if ylim: ax.set_ylim(*ylim)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="bottom")
    mono_ticks(ax)

def heatmap(ax, letter):
    cols = [("K562", "d20"), ("K562", "d"), ("RPE1", "d20"), ("RPE1", "d")]
    M = np.array([[D[(f"pearson_{mm}", cc, meth)][0] for cc, mm in cols]
                  for meth in METHODS])
    cmap = LinearSegmentedColormap.from_list("bad_good", SEQ_BAD_GOOD_L)
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=0.75, aspect="auto")
    ax.set_xticks(range(4)); ax.set_yticks(range(len(METHODS)))
    ax.set_xticklabels(["K562\n$\\Delta_{20}$", "K562\n$\\Delta$",
                        "RPE1\n$\\Delta_{20}$", "RPE1\n$\\Delta$"])
    ax.set_yticklabels(METHODS)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "iPerturb": lbl.set_fontweight("bold")
    for i in range(len(METHODS)):
        for j in range(4):
            v = M[i, j]; rgba = cmap(v/0.75)
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    family="monospace", color="white" if lum < 0.5 else GREY900)
    ax.set_title("Pearson correlation", fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="bottom")

fig, axs = plt.subplots(2, 2, figsize=(6.8, 2.55))
grouped(axs[0, 0], "directional", "Directional accuracy", "a", ylim=(0.5, 0.95))
grouped(axs[0, 1], "mse", "Mean squared error", "b", ylim=(0, 0.85))
heatmap(axs[1, 0], "c")
grouped(axs[1, 1], "centroid", "Centroid accuracy", "d", ylim=(0, 0.75))

handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in METHODS]
fig.legend(handles, METHODS, loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.015), handlelength=1.1, columnspacing=1.1,
           handletextpad=0.4, fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.0, h_pad=2.0)
for ext in (".pdf", ".png"):
    fig.savefig("figure3_combined" + ext)
print("wrote figure3_combined.pdf/.png")
