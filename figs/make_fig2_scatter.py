"""Figure 2 (improved): TF target-count scatter, K562 vs RPE1, in Google style.
- symlog axes (linear near 0) un-stack the origin pile-up the reviewers flagged
- leader-line labels for the five highlighted, cell-type-divergent TFs
- >=7-8 pt fonts at final print size; reconstructed cloud + exact highlighted TFs
Outputs tf_k562_rpe1_scatter_colored.{pdf,png} into the paper directory."""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "figs")
from style_google import apply_style, mono_ticks, GBLUE, GREY500, GREY300, GREY900

apply_style()
cloud, hi = [], []
with open("figs/fig2_points.csv") as f:
    for r in csv.DictReader(f):
        (hi if r["cls"] == "highlight" else cloud).append(
            (float(r["k562"]), float(r["rpe1"]), r["label"]))

fig, ax = plt.subplots(figsize=(3.15, 2.15))
LT = 10  # symlog linear threshold
for a in ("x", "y"):
    getattr(ax, f"set_{a}scale")("symlog", linthresh=LT, linscale=0.7)

# identity diagonal
ax.plot([0, 900], [0, 900], ls=(0, (4, 3)), color=GREY300, lw=1.0, zorder=1)

# background cloud (reconstructed)
ax.scatter([c[0] for c in cloud], [c[1] for c in cloud], s=11, c=GREY500,
           alpha=0.55, linewidths=0, zorder=2, label="TF")
# highlighted TFs (exact)
ax.scatter([h[0] for h in hi], [h[1] for h in hi], s=46, c=GBLUE,
           edgecolors="white", linewidths=0.8, zorder=4, label="Highlighted TF")

# leader-line labels, hand-placed (dx, dy in points, ha) to avoid collisions
OFF = {"TRIM24": (-18, 20, "right"), "ATF4": (20, 8, "left"),
       "ZEB2": (-10, -20, "center"), "REST": (-32, -14, "right"),
       "REL": (-30, 8, "right")}
for x, y, name in hi:
    dx, dy, ha = OFF.get(name, (10, 10, "center"))
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, fontweight="bold", color=GREY900, zorder=5, ha=ha,
                arrowprops=dict(arrowstyle="-", color=GREY500, lw=0.7,
                                shrinkA=0, shrinkB=4))

ticks = [0, 10, 50, 100, 300, 500, 800]
ax.set_xticks([t for t in ticks if t <= 550]); ax.set_yticks(ticks)
ax.set_xlim(-2, 640); ax.set_ylim(-2, 1050)
ax.set_xlabel("K562 target count"); ax.set_ylabel("RPE1 target count")
mono_ticks(ax)
ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
fig.tight_layout(pad=0.4)
for ext in (".pdf", ".png"):
    fig.savefig("tf_k562_rpe1_scatter_colored" + ext)
print("wrote tf_k562_rpe1_scatter_colored.pdf/.png   cloud=%d highlighted=%d"
      % (len(cloud), len(hi)))
