"""Figure 1 (redrawn, Google/Material style): the iPerturb framework overview.
Cell line + perturbation -> build a context-specific Template GRN from public
databases -> fit signed Hill kinetics on CRISPRi Perturb-seq (Fitted GRN) ->
predict the response by multi-hop message passing. Outputs iPerturb_overview.pdf."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
sys.path.insert(0, "figs")
from style_google import (apply_style, GBLUE, GRED, GGREEN, GREY300, GREY500,
                          GREY700, GREY900)
apply_style()

ACT, REP = GGREEN, GRED
SURF, BORDER = "#F8F9FA", GREY300
BLUE_BG, GREEN_BG = "#E8F0FE", "#E6F4EA"
LINK, GREENTX = "#1A73E8", "#137333"

# gene mini-network (shared topology; weights shown only in the fitted panel)
NODES = {"g_2": (-1.0, 0.85), "g_1": (0.65, 1.05), "g_3": (0.0, 0.0),
         "g_x": (-1.15, -0.6), "g_y": (0.0, -1.2), "g_z": (1.1, -0.6)}
EDGES = [("g_1", "g_3", "act", 0.5), ("g_2", "g_3", "act", 0.7),
         ("g_2", "g_x", "rep", 0.9), ("g_x", "g_3", "act", 0.1),
         ("g_3", "g_z", "rep", 0.2), ("g_y", "g_3", "rep", 0.4),
         ("g_z", "g_y", "act", 0.7)]

def draw_network(ax, cx, cy, s, weighted):
    nr = 1.05
    pos = {k: (cx + x*s, cy + y*s) for k, (x, y) in NODES.items()}
    for a, b, sign, w in EDGES:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        dx, dy = x2-x1, y2-y1; L = (dx*dx+dy*dy)**0.5; ux, uy = dx/L, dy/L
        sx, sy = x1+ux*nr, y1+uy*nr
        ex, ey = x2-ux*nr, y2-uy*nr
        col = ACT if sign == "act" else REP
        if sign == "act":
            ax.annotate("", (ex, ey), (sx, sy), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                        shrinkA=0, shrinkB=0, mutation_scale=7))
        else:
            ax.plot([sx, ex], [sy, ey], color=col, lw=1.0, solid_capstyle="round", zorder=3)
            px, py = -uy, ux; bl = 0.7
            ax.plot([ex-px*bl, ex+px*bl], [ey-py*bl, ey+py*bl], color=col,
                    lw=1.4, solid_capstyle="round", zorder=3)
        if weighted:
            mx, my = sx*0.55+ex*0.45, sy*0.55+ey*0.45
            ax.text(mx, my, f"{w:.1f}", fontsize=5.6, family="monospace", color=col,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=0.9))
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), nr, fc="white", ec=GREY700, lw=0.9, zorder=5))
        ax.text(x, y, "$"+k+"$", fontsize=7.0, ha="center", va="center",
                zorder=6, color=GREY900)

def card(ax, x0, y0, x1, y1, title, sub):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                 boxstyle="round,pad=0,rounding_size=2.2", fc="white",
                 ec=BORDER, lw=1.0, zorder=1))
    ax.text((x0+x1)/2, y1-3.0, title, fontsize=8, color=GREY900,
            ha="center", va="center", zorder=7, fontweight="medium")
    ax.text((x0+x1)/2, y0+2.4, sub, fontsize=6.0, family="monospace",
            color=GREY700, ha="center", va="center", zorder=7)

def pill(ax, cx, cy, w, h, lines, bg, fg):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0,rounding_size=3.0", fc=bg, ec="none", zorder=2))
    ax.text(cx, cy, lines, fontsize=6.6, color=fg, ha="center", va="center",
            zorder=3, fontweight="medium")

def badge(ax, x, y, n):
    ax.add_patch(Circle((x, y), 1.7, fc=GBLUE, ec="none", zorder=8))
    ax.text(x, y, n, fontsize=6.5, family="monospace", color="white",
            ha="center", va="center", zorder=9, fontweight="bold")

def flow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=11, color=GREY500, lw=1.4, zorder=2,
                 shrinkA=0, shrinkB=0))

fig, ax = plt.subplots(figsize=(6.8, 1.98))
ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")

# iPerturb container
ax.add_patch(FancyBboxPatch((14.5, 3.0), 72, 24.5,
             boxstyle="round,pad=0,rounding_size=2.4", fc="none",
             ec=GREY500, lw=1.0, ls=(0, (4, 3)), zorder=0))
ax.text(50, 4.3, "iPerturb", fontsize=6.5, family="monospace", color=GREY500,
        ha="center", va="center", zorder=1)

# input / output pills
pill(ax, 6.8, 15.8, 12.6, 10.5, "cell line\n+ pert.", BLUE_BG, LINK)
pill(ax, 93.2, 15.8, 12.6, 10.5, "predicted\nexpression", GREEN_BG, GREENTX)

# top data inputs
for cx, txt in [(33, "public GRN\ndatabases"), (64, "CRISPRi\nPerturb-seq")]:
    ax.add_patch(FancyBboxPatch((cx-9, 27.9), 18, 2.9,
                 boxstyle="round,pad=0,rounding_size=1.3", fc=SURF, ec=BORDER, lw=0.9, zorder=2))
    ax.text(cx, 29.35, txt, fontsize=5.2, family="monospace", color=GREY700,
            ha="center", va="center", zorder=3)
    ax.add_patch(FancyArrowPatch((cx, 27.8), (cx, 25.9), arrowstyle="-|>",
                 mutation_scale=7, color=GREY500, lw=1.0, zorder=2, shrinkA=0, shrinkB=0))

# two GRN cards
card(ax, 19, 6.6, 47, 25.7, "Template GRN", "signed edges")
draw_network(ax, 33, 16.1, 3.8, weighted=False); badge(ax, 21.2, 23.6, "1")
card(ax, 50, 6.6, 78, 25.7, "Fitted GRN", "Hill kinetics")
draw_network(ax, 64, 16.1, 3.8, weighted=True);  badge(ax, 52.2, 23.6, "2")

# flow arrows
flow(ax, 13.2, 15.8, 19.0, 15.8)
flow(ax, 47.2, 15.8, 50.0, 15.8)
flow(ax, 78.2, 15.8, 86.9, 15.8)
ax.text(48.6, 17.9, "fit", fontsize=5.2, family="monospace", color=GREY500, ha="center")
badge(ax, 84, 20.3, "3")
ax.text(83.5, 17.3, "message\npassing", fontsize=5.0, family="monospace",
        color=GREY500, ha="center", va="center")

fig.tight_layout(pad=0.2)
for ext in (".pdf", ".png"):
    fig.savefig("iPerturb_overview" + ext)
print("wrote iPerturb_overview.pdf/.png")
