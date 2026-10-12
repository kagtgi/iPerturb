"""Fig. 1: iPerturb pipeline schematic (122 mm x 74 mm, LNCS text width).
Run from the repo root:  PYTHONPATH=. python scripts/fig1_overview.py
Pure schematic: no data are plotted; the only numbers drawn are the STRING threshold
(>=700) and cell-line names."""
import os
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import (FancyBboxPatch, Rectangle, Circle, Ellipse, Arc,
                                FancyArrowPatch)
from ipt.viz.style import apply_style, GREY300, GREY500, GREY700, GREY900, ARROW_KW, LW_ARROW, FS_TITLE
from ipt.viz.style import GBLUE_L, GGREEN, GRED, GBLUE, GGREY, GPURPLE_L, GYELLOW

FONTDIR = r"D:\iPerturb\lnbi\figure_tools\assets\fonts\Google_Sans\static"
for f in os.listdir(FONTDIR):
    if f.endswith(".ttf"):
        fm.fontManager.addfont(os.path.join(FONTDIR, f))
apply_style()
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Google Sans"],
                     "pdf.fonttype": 42, "axes.grid": False, "savefig.bbox": "standard"})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

W, H = 122.0, 74.0
MM = 1 / 25.4
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FS = 7.0
TEXTS = []          # (artist, container-rect or None)
ACT, REP = GGREEN, GRED
ASSOC = GGREY
DARK = GREY900


def T(x, y, s, ha="center", va="center", w="regular", c=DARK, fs=FS, box=None, ls=1.12):
    t = ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=w, color=c, linespacing=ls, zorder=8)
    TEXTS.append((t, box)); return t


def rbox(x, y, w, h, fc="white", ec=GREY300, lw=0.6, r=1.0, z=1, ls="-"):
    p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p); return p


def seg(p, q, shrink0=0, shrink1=0):
    p, q = np.array(p, float), np.array(q, float); d = q - p; L = np.hypot(*d); u = d / L
    return p + u * shrink0, q - u * shrink1, u


def edge(p, q, kind, r0=0.8, r1=0.8, lw=0.6, z=3):
    a, b, u = seg(p, q, r0, r1)
    if kind == "act":
        ax.add_patch(FancyArrowPatch(a, b, **ARROW_KW, lw=LW_ARROW, color=ACT, shrinkA=0, shrinkB=0, zorder=z))
    elif kind == "rep":
        ax.plot([a[0], b[0]], [a[1], b[1]], color=REP, lw=lw, zorder=z, solid_capstyle="butt")
        n = np.array([-u[1], u[0]]) * 0.8
        ax.plot([b[0] - n[0], b[0] + n[0]], [b[1] - n[1], b[1] + n[1]], color=REP, lw=lw * 1.4, zorder=z, solid_capstyle="butt")
    elif kind == "assoc":
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ASSOC, lw=lw, ls=(0, (2, 1.4)), zorder=z, solid_capstyle="butt")
    elif kind == "plain":
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GREY700, lw=lw, zorder=z)
    elif kind == "flow":
        ax.add_patch(FancyArrowPatch(a, b, **ARROW_KW, lw=LW_ARROW, color="black", shrinkA=0, shrinkB=0, zorder=z))


def node(p, r=0.8, fc=GBLUE, ec="white", z=4, lw=0.3):
    ax.add_patch(Circle(p, r, fc=fc, ec=ec, lw=lw, zorder=z))


def mini_grn(x0, y0, w, h, nodes, edges, hl=None):
    pts = [(x0 + a * w, y0 + b * h) for a, b in nodes]
    for i, j, k in edges:
        edge(pts[i], pts[j], k, lw=0.55)
    for i, p in enumerate(pts):
        node(p, 0.85, fc=(hl or {}).get(i, GBLUE))
    return pts


# ---------------------------------------------------------------- layout
BY0, BY1 = 10.0, 69.0
OY0, EY1 = 46.3, 40.8          # output box bottom / explanation box top
widths = [18.3, 47.6, 21.7, 24.4, 24.4]
gap = 3.2
x = (W - (sum(widths[:4]) + gap * 3)) / 2
XS = []
for w in widths[:4]:
    XS.append(x); x += w + gap
XS.append(XS[3]); widths = widths[:4] + [widths[3]]
BOXES = [(XS[0], BY0, BY1, "Inputs"), (XS[1], BY0, BY1, "Context-specific GRN"),
         (XS[2], BY0, BY1, "Hill-form\npropagation"), (XS[3], OY0, BY1, "Output"),
         (XS[3], BY0, EY1, "Explanation")]
for i, (x0, y0, y1, title) in enumerate(BOXES):
    w = widths[i]
    core = i in (1, 2)
    rbox(x0, y0, w, y1 - y0, fc="#F8F9FA" if not core else "#F3F7FE",
         ec=GREY300 if not core else GBLUE_L, lw=0.7, r=1.2)
    ax.add_patch(Circle((x0 + w / 2, y1), 1.45, fc=GBLUE, ec="white", lw=0.5, zorder=6))
    T(x0 + w / 2, y1, str(i + 1), w="bold", c="white", fs=FS)
    T(x0 + w / 2, y1 - (6.0 if "\n" in title else 4.6), title, w="bold", fs=FS_TITLE, box=(x0, y0, w, y1 - y0))
for i in range(3):
    xa = XS[i] + widths[i]
    edge((xa + 0.15, 40 if i < 2 else 56), (xa + gap - 0.15, 40 if i < 2 else 56), "flow", 0, 0)
edge((XS[3] + widths[3] / 2, OY0 - 0.15), (XS[3] + widths[3] / 2, EY1 + 1.45 + 0.15), "flow", 0, 0)

# ---------------------------------------------------------------- 1 Inputs
P = 3.1   # line pitch (mm) for 7 pt stacked text
x0, w = XS[0], widths[0]; cx = x0 + w / 2
box0 = (x0, BY0, w, BY1 - BY0)
yk = 58.0
for k, nm in enumerate(["K562", "RPE1"]):
    xx = x0 + 1.3 + k * 8.0
    rbox(xx, yk - 1.5, 7.4, 3.0, fc="white", ec=GBLUE, lw=0.6, r=1.0, z=2)
    T(xx + 3.7, yk, nm, fs=FS, box=box0)
for k, s in enumerate(["cell line", "+ CRISPRi", "target gene"]):
    T(cx, 53.8 - k * P, s, box=box0)
node((cx - 3.0, 43.8), 1.1, fc=GBLUE)
ax.plot([cx - 1.9, cx + 3.7], [43.8, 43.8], color=REP, lw=0.6, zorder=3)
ax.plot([cx + 3.7, cx + 3.7], [42.7, 44.9], color=REP, lw=0.85, zorder=3)
ax.plot([x0 + 1.5, x0 + w - 1.5], [40, 40], color=GREY300, lw=0.5, zorder=2)
rng = np.random.RandomState(3)
for k in range(9):
    a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0, 3.4)
    ax.add_patch(Ellipse((cx + r * np.cos(a), 35.2 + r * np.sin(a) * 0.8), 1.7, 1.7,
                         fc="white", ec=GGREY, lw=0.5, zorder=3))
    ax.add_patch(Circle((cx + r * np.cos(a), 35.2 + r * np.sin(a) * 0.8), 0.45, fc=GGREY, ec="none", zorder=4))
for k, s in enumerate(["Perturb-seq", "control cells", "+ training", "perturbations"]):
    T(cx, 29.8 - k * P, s, box=box0)
for k, s in enumerate(["mean response", "as target"]):
    T(cx, 16.2 - k * P, s, box=box0)

# ---------------------------------------------------------------- 2 Context-specific GRN
x0, w = XS[1], widths[1]
box1 = (x0, BY0, w, BY1 - BY0)
CW = 33.5
cxl = x0 + 1.2
chips = [
    ("L1", ["L1 signed TF→gene", "CollecTRI, OmniPath,", "TRRUST"]),
    ("L2", ["L2 GeneHancer", "enhancer + TFBS,", "active in cell line"]),
    ("L3", ["L3 STRING protein", "association ≥700"]),
    ("L4", ["L4 COXPRESdb", "coexpression"]),
    ("L5", ["L5 training-", "perturbation response"]),
]
LH = P
ytop = BY1 - 7.0
yy = ytop
for tag, lines in chips:
    h = len(lines) * LH + 1.4
    yb = yy - h
    cb = (cxl, yb, CW, h)
    rbox(cxl, yb, CW, h, fc="white", ec=GREY300, lw=0.6, r=0.8, z=2)
    mx0, mx1, my = cxl + 1.2, cxl + 5.4, yb + h / 2
    if tag == "L1":
        node((mx0, my + 1.9), 0.6, fc=GBLUE); node((mx1, my + 1.9), 0.6, fc=GBLUE)
        edge((mx0, my + 1.9), (mx1, my + 1.9), "act", 0.6, 0.6, lw=0.6)
        node((mx0, my - 1.9), 0.6, fc=GBLUE); node((mx1, my - 1.9), 0.6, fc=GBLUE)
        edge((mx0, my - 1.9), (mx1, my - 1.9), "rep", 0.6, 0.6, lw=0.6)
    elif tag == "L2":
        ax.plot([mx0, mx1], [my, my], color=GREY700, lw=0.6, zorder=3)
        ax.add_patch(Rectangle((mx0 + 1.2, my - 0.6), 1.8, 1.2, fc=GBLUE_L, ec=GREY700, lw=0.4, zorder=4))
        node((mx0, my), 0.6, fc=GBLUE); node((mx1, my), 0.6, fc=GBLUE)
    elif tag in ("L3", "L4"):
        node((mx0, my), 0.6, fc=GBLUE); node((mx1, my), 0.6, fc=GBLUE)
        edge((mx0, my), (mx1, my), "assoc", 0.6, 0.6, lw=0.7)
    else:
        for k, v in enumerate([0.5, -0.9, 1.0, -0.4]):
            ax.add_patch(Rectangle((mx0 + k * 1.1, my), 0.8, v, fc=GYELLOW if v > 0 else GPURPLE_L,
                                   ec="none", zorder=4))
        ax.plot([mx0 - 0.2, mx1 + 0.2], [my, my], color=GREY700, lw=0.4, zorder=3)
    for k, s in enumerate(lines):
        T(cxl + 7.0, yb + h - 0.7 - LH * (k + 0.5), s, ha="left", fs=FS, box=cb)
    yy = yb - 1.15
rx = x0 + 1.2 + CW + 1.2; rw = x0 + w - 0.8 - rx; rc = rx + rw / 2
cyl_y = ytop - 2.0
ax.add_patch(Rectangle((rc - 3.0, cyl_y - 5.0), 6.0, 5.0, fc=GBLUE_L, ec=GBLUE, lw=0.5, zorder=3))
ax.add_patch(Ellipse((rc, cyl_y - 5.0), 6.0, 1.8, fc=GBLUE_L, ec=GBLUE, lw=0.5, zorder=3))
ax.add_patch(Ellipse((rc, cyl_y), 6.0, 1.8, fc="#D2E3FC", ec=GBLUE, lw=0.5, zorder=4))
T(rc, cyl_y - 7.3, "databases", box=box1)
edge((rc, cyl_y - 9.1), (rc, cyl_y - 11.2), "flow", 0, 0)
for k, s in enumerate(["context-", "specific", "GRN"]):
    T(rc, 46.6 - k * P, s, box=box1)
nodes = [(0.12, 0.85), (0.88, 0.9), (0.5, 0.45), (0.15, 0.08), (0.85, 0.1)]
gw, gh = rw - 1.8, 8.0
T(rc, 36.0, "K562", w="bold", c=GBLUE, box=box1)
mini_grn(rc - gw / 2, 26.6, gw, gh, nodes, [(0, 2, "act"), (1, 2, "rep"), (2, 4, "act"), (0, 3, "assoc")])
T(rc, 23.0, "RPE1", w="bold", c=GBLUE, box=box1)
mini_grn(rc - gw / 2, 13.6, gw, gh, nodes, [(0, 2, "rep"), (2, 3, "act"), (1, 4, "assoc"), (1, 2, "act")])

# ---------------------------------------------------------------- 3 Hill propagation
x0, w = XS[2], widths[2]; cx = x0 + w / 2
box2 = (x0, BY0, w, BY1 - BY0)
ca, cb_ = x0 + 2.0, 50.8
cw_, ch_ = 10.0, 7.5
ax.plot([ca, ca], [cb_, cb_ + ch_], color=GREY700, lw=0.5, zorder=3)
ax.plot([ca, ca + cw_], [cb_, cb_], color=GREY700, lw=0.5, zorder=3)
xs = np.linspace(0, 1, 60); k_, n_ = 0.45, 3.0
ys = xs ** n_ / (k_ ** n_ + xs ** n_)
ax.plot(ca + xs * cw_, cb_ + 0.3 + ys * (ch_ - 0.8), color=GBLUE, lw=1.0, zorder=4)
kx = ca + k_ * cw_
ax.plot([kx, kx], [cb_, cb_ + 0.3 + 0.5 * (ch_ - 0.8)], color=GREY500, lw=0.4, ls=(0, (2, 1.5)), zorder=3)
T(ca + cw_ + 1.0, cb_ + ch_ * 0.55, "H(x;k)", ha="left", box=box2)
edge((cx, cb_ - 1.3), (cx, cb_ - 4.2), "flow", 0, 0)
mg = 2.3
mx, my0 = cx - 1.5 * mg, 39.0
shade = ["#D2E3FC", "white", GBLUE_L, "white", "#D2E3FC", "white", GBLUE_L, "white", "#D2E3FC"]
for r in range(3):
    for c in range(3):
        ax.add_patch(Rectangle((mx + c * mg, my0 + (2 - r) * mg), mg, mg, fc=shade[r * 3 + c],
                               ec=GREY700, lw=0.4, zorder=3))
for k, s in enumerate(["Chebyshev basis", "× sparse/dense", "matmul"]):
    T(cx, my0 - 2.2 - k * P, s, box=box2)
edge((cx, 28.2), (cx, 26.0), "flow", 0, 0)
lx = x0 + 6.8; ly = 21.0
ax.add_patch(Arc((lx, ly + 2.0), 2.4, 2.9, theta1=0, theta2=180, ec=GREY900, lw=0.8, zorder=4))
ax.add_patch(FancyBboxPatch((lx - 1.9 + 0.3, ly - 1.4 + 0.3), 3.2, 3.0, boxstyle="round,pad=0.3",
                            fc=GYELLOW, ec=GREY900, lw=0.5, zorder=5))
ax.add_patch(Circle((lx, ly + 0.1), 0.4, fc=GREY900, ec="none", zorder=6))
for k, s in enumerate(["persistent", "clamp"]):
    T(lx, 16.4 - k * P, s, box=box2)
px = x0 + w - 5.2
ax.add_patch(Arc((px, 21.7), 5.2, 5.2, theta1=-60, theta2=235, ec=GBLUE, lw=1.0, zorder=4))
ha = np.deg2rad(235); hp = (px + 2.6 * np.cos(ha), 21.7 + 2.6 * np.sin(ha))
tan = np.array([-np.sin(ha), np.cos(ha)])
ax.add_patch(FancyArrowPatch((hp[0] - tan[0] * 0.3, hp[1] - tan[1] * 0.3), (hp[0] + tan[0] * 0.3, hp[1] + tan[1] * 0.3),
                             **ARROW_KW, color=GBLUE, lw=LW_ARROW, shrinkA=0, shrinkB=0, zorder=5))
T(px, 21.7, "x*", w="bold", c=GBLUE, box=box2)
for k, s in enumerate(["fixed", "point"]):
    T(px, 16.4 - k * P, s, box=box2)

# ---------------------------------------------------------------- 4 Output
x0, w = XS[3], widths[3]; cx = x0 + w / 2
box3 = (x0, OY0, w, BY1 - OY0)
vals = [0.9, 0.55, 0.3, -0.15, -0.5, -0.95, 0.7, -0.3, 0.25, -0.6, 0.45, -0.2]
bw_, bg_ = 1.3, 0.45
xb0 = cx - (12 * bw_ + 11 * bg_) / 2
base = 57.9
for k, v in enumerate(vals):
    ax.add_patch(Rectangle((xb0 + k * (bw_ + bg_), base), bw_, v * 3.8, fc=GYELLOW if v > 0 else GPURPLE_L,
                           ec="none", zorder=3))
ax.plot([xb0 - 0.5, xb0 + 12 * bw_ + 11 * bg_ + 0.5], [base, base], color=GREY700, lw=0.5, zorder=4)
for k, s in enumerate(["predicted", "expression change"]):
    T(cx, 51.7 - k * P, s, box=box3)

# ---------------------------------------------------------------- 5 Explanation
x0, w = XS[3], widths[3]; cx = x0 + w / 2
box4 = (x0, BY0, w, EY1 - BY0)
pts = mini_grn(x0 + 3.5, 27.9, w - 7.0, 5.0,
               [(0.5, 0.9), (0.1, 0.5), (0.9, 0.5), (0.3, 0.05), (0.75, 0.05)],
               [(1, 0, "act"), (2, 0, "rep"), (3, 1, "act"), (4, 2, "assoc")],
               hl={0: GBLUE, 1: GBLUE_L, 2: GBLUE_L, 3: GREY500, 4: GREY500})
ax.add_patch(Circle(pts[0], 1.3, fc="none", ec=GBLUE, lw=0.6, zorder=5))
lines5 = ["top affected genes", "+ supporting", "neighbours/paths,", "validated by edge/", "neighbour ablation"]
for k, s in enumerate(lines5):
    T(cx, 23.9 - k * P, s, box=box4)

# ---------------------------------------------------------------- legend strip
LY = 4.2
rbox(0.6, 0.7, W - 1.2, 7.0, fc="white", ec=GREY300, lw=0.5, r=1.0)
lx = 4.0
edge((lx, LY), (lx + 5.5, LY), "act", 0, 0, lw=0.8)
edge((lx + 7.2, LY), (lx + 12.7, LY), "rep", 0, 0, lw=0.8)
T(lx + 14.5, LY, "signed regulation", ha="left")
lx2 = 56.5
edge((lx2, LY), (lx2 + 6.5, LY), "assoc", 0, 0, lw=0.9)
T(lx2 + 8.3, LY, "association (undirected)", ha="left")
lx3 = 99.0
edge((lx3, LY), (lx3 + 5.5, LY), "flow", 0, 0)
T(lx3 + 7.3, LY, "data flow", ha="left")

# ---------------------------------------------------------------- QC (mechanical)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bbs, bad = [], []
for t, box in TEXTS:
    bb = t.get_window_extent(rend); (a, b), (c, d) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    bbs.append((t.get_text().replace("\n", "/"), a, b, c, d))
    if a < 0.3 or c > W - 0.3 or b < 0.3 or d > H - 0.3:
        bad.append(("off-canvas", t.get_text()))
    if box is not None:
        bx, by, bw_, bh_ = box
        if a < bx + 0.4 or c > bx + bw_ - 0.4 or b < by + 0.3 or d > by + bh_ - 0.3:
            bad.append(("outside container", t.get_text(), round(a - bx, 2), round(bx + bw_ - c, 2)))
    if t.get_fontsize() < 7.0:
        bad.append(("font<7", t.get_text()))
for (n1, a1, b1, c1, d1), (n2, a2, b2, c2, d2) in itertools.combinations(bbs, 2):
    if a1 < c2 - 0.05 and a2 < c1 - 0.05 and b1 + 0.3 < d2 - 0.3 and b2 + 0.3 < d1 - 0.3:
        bad.append(("text overlap", n1, n2))
print("QC problems:", bad if bad else "none")
print("texts:", len(TEXTS), " min fontsize:", min(t.get_fontsize() for t, _ in TEXTS))
print("fonts:", {t.get_fontname() for t, _ in TEXTS})

fig.savefig(os.path.join(OUT, "fig1_overview.pdf"), dpi=600, bbox_inches=None, pad_inches=0)
fig.savefig(os.path.join(OUT, "fig1_overview.png"), dpi=600, bbox_inches=None, pad_inches=0)
