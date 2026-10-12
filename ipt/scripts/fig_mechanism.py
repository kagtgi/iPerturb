"""Fig. mechanism: clamped steady-state propagation explained (a), sign accuracy by hop vs controls (b),
a mechanism report (c), hop-wise concordance and reached-gene Pearson (d).  Run from ipt/ with PYTHONPATH=."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Arc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from ipt.viz.style import *
from ipt.explain.mechanism import build_W, clamped_response, reachable
apply_style()
ART, OUT = ROOT / "artifacts", ROOT / "paper" / "figures"
MM = 1 / 25.4
LINES = [("K562_essential", "K562"), ("rpe1", "RPE1")]
CMAP = LinearSegmentedColormap.from_list("bwr_g", [GBLUE, "white", GRED])
FS = 7.0


def wilson(h, n, z=1.96):
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - w, c + w


def lock(ax, x, y, s=0.45, col=GREY900):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s * 0.8, fc=col, ec=col, lw=0.5, zorder=6))
    ax.add_patch(Arc((x, y + s * 0.3), s * 0.62, s * 0.8, theta1=0, theta2=180, ec=col, lw=1.1, zorder=6))


def edge(ax, p, q, sign, rad=0.0, shrink=11.5, lw=LW_ARROW):
    col = GGREEN if sign > 0 else GRED
    style, ms = (ARROW_STYLE, ARROW_MS) if sign > 0 else ("-[,widthB=0.45,lengthB=0.0", ARROW_MS)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink,
                                 color=col, lw=lw, mutation_scale=ms, zorder=2))


# ------------------------------------------------------------------ data
sc = {}
for ln, _ in LINES:
    sc[ln] = pd.read_csv(ART / "scores" / f"g13_{ln}.tsv", sep="\t")


def acc_table(d, hops):
    a = d[(d.model == "steady") & (d.hop.isin(hops))].copy()
    a["kind"] = a.world.str.replace(r"\d+$", "", regex=True)
    return a


# ------------------------------------------------------------------ toy (a): real code
src = np.array([0, 1, 1, 3]); dst = np.array([1, 2, 3, 1]); sgn = np.array([1, -1, 1, -1]); conf = np.ones(4)
W, _, _ = build_W(src, dst, sgn, conf, 4, rho=0.8)
nodes, steady, _ = clamped_response(W, 0, -1.0, reachable(W, 0))
toy = dict(zip(nodes.tolist(), steady.tolist()))
print("toy dx (B,C,D) =", [round(toy[i], 3) for i in (1, 2, 3)])
assert toy[1] < 0 and toy[2] > 0 and toy[3] < 0

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(122 * MM, 108 * MM))

# ---- (a)
axa = fig.add_axes([0.005, 0.60, 0.355, 0.33]); axa.axis("off"); axa.set_xlim(0, 10); axa.set_ylim(0, 10)
P = dict(A=(1.3, 5.6), B=(4.7, 5.6), C=(8.6, 8.1), D=(8.6, 3.1))
m = max(abs(v) for v in toy.values())
col = {"A": "white", "B": CMAP(0.5 + 0.5 * toy[1] / m), "C": CMAP(0.5 + 0.5 * toy[2] / m), "D": CMAP(0.5 + 0.5 * toy[3] / m)}
SA = 6.5   # node radius (pt) of the toy nodes, same size as the toy GRN of the propagation figure
UX, UY = 10 / (0.355 * 122), 10 / (0.33 * 108)      # data units per mm (x, y)
RN = (SA - 1.3) / 72 * 25.4 + 0.35                   # node radius + gap, mm


def on(k, deg):
    """Point on the rim of node k at angle deg (screen angle, mm-isotropic)."""
    a = np.deg2rad(deg)
    return (P[k][0] + RN * np.cos(a) * UX, P[k][1] + RN * np.sin(a) * UY)


def edge2(p, q, sign, rad=0.0):
    col = GGREEN if sign > 0 else GRED
    style = ARROW_STYLE if sign > 0 else "-[,widthB=0.45,lengthB=0.0"
    axa.add_patch(FancyArrowPatch(p, q, arrowstyle=style, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                  color=col, lw=LW_ARROW, mutation_scale=ARROW_MS, zorder=2))


edge2(on("A", 0), on("B", 180), +1)
edge2(on("B", 62), on("C", 205), -1)                  # leaves B toward the upper right
edge2(on("B", -105), on("D", 190), +1, rad=0.3)        # leaves B downward, enters D from the left
edge2(on("D", 130), on("B", -32), -1, rad=0.2)        # feedback arrives at B's right side
for k, (x, y) in P.items():
    axa.scatter([x], [y], s=(2 * SA - 2.6) ** 2, c=[col[k]], edgecolors=GREY900 if k == "A" else GREY700, linewidths=1.0 if k == "A" else LW, zorder=4)
    axa.text(x, y, k, ha="center", va="center", fontsize=FS, zorder=5)
lock(axa, P["A"][0], P["A"][1] + 1.25)
axa.text(P["A"][0] + 0.4, 9.2, "CRISPRi\n(clamped)", ha="center", va="center", fontsize=FS)
axa.text(0.1, 1.2, "B down, C up, D down", ha="left", va="center", fontsize=FS, color=GREY900)
axa.text(7.2, 5.75, "feedback", ha="left", va="center", fontsize=FS, color=GREY700)
edge2((0.2, 2.9), (1.6, 2.9), +1); axa.text(1.85, 2.9, "act.", va="center", fontsize=FS, color=GREY700)
edge2((3.6, 2.9), (5.0, 2.9), -1); axa.text(5.25, 2.9, "rep.", va="center", fontsize=FS, color=GREY700)

# ---- (b)
KC = {"real": (GBLUE, "real signs"), "signperm": (GGREY, "permuted signs"), "rewired": (GYELLOW, "rewired edges")}
HOPS = ["1", "2", "3", "4+"]; HL = ["1", "2", "3", "$\\geq$4"]
axb = []
b_res = {}
for j, (ln, nm) in enumerate(LINES):
    ax = fig.add_axes([0.435 + j * 0.285, 0.665, 0.255, 0.27]); axb.append(ax)
    a = acc_table(sc[ln], HOPS)
    for k, off in zip(KC, (-0.24, 0, 0.24)):
        g = a[a.kind == k].groupby("hop")[["n", "hit"]].sum()
        for h, hop in enumerate(HOPS):
            p, lo, hi = wilson(g.loc[hop, "hit"], g.loc[hop, "n"])
            ax.errorbar(h + off, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=3.2, color=KC[k][0], elinewidth=LW, capsize=0, zorder=3)
            b_res[(ln, k, hop)] = (p, lo, hi, int(g.loc[hop, "n"]))
    ax.axhline(0.5, color=GREY900, lw=LW_THIN, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(HL); ax.set_xlim(-0.55, 3.55); ax.set_ylim(0.4, 1.0)
    ax.set_title(nm, fontsize=FS_TITLE, pad=3, color=GREY900)
    ax.set_xlabel("hop distance", fontsize=FS, labelpad=1)
    ax.tick_params(labelsize=FS, pad=1.5)
    n1 = b_res[(ln, "real", "1")][3]
    ax.text(-0.5, 0.43, f"n={n1}", fontsize=FS, color=GBLUE, va="bottom", ha="left")
    if j == 0:
        ax.set_ylabel("sign accuracy", fontsize=FS, labelpad=1)
    else:
        ax.set_yticklabels([])
axb[1].legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, color=KC[k][0], label=KC[k][1]) for k in KC], loc="upper right",
              fontsize=FS, handletextpad=0.1, labelspacing=0.15, borderaxespad=0.1)

# ---- (c)
txt = (ART / "reports" / "mechanism_K562_essential.md").read_text(encoding="utf8")
sec = re.split(r"\n\n(?=## )", txt)[0]
tgt = re.match(r"## (\S+) ", sec).group(1)
recs = []
for blk in re.split(r"\n- ", sec)[1:]:
    ls = blk.split("\n")
    mm = re.match(r"(\S+): observed z=([+-][\d.]+), predicted sign ([+-]) \(hop (\d+)\)", ls[0])
    path = next(l for l in ls if l.strip().startswith("path "))
    chain = re.match(r"\s*path (.+?) \(net", path).group(1).split(" -> ")
    eds = re.findall(r"(\w+)(->|-\|)(\w+) \[([^\]]+)\]", path)
    recs.append(dict(gene=mm.group(1), z=float(mm.group(2)), pred=mm.group(3), hop=int(mm.group(4)), chain=chain, edges=eds))
recs = recs[:5]
print("case study:", tgt, [(r["gene"], r["z"], r["pred"], r["chain"]) for r in recs])
axc = fig.add_axes([0.005, 0.15, 0.55, 0.43]); axc.axis("off"); axc.set_xlim(0, 11.4); axc.set_ylim(0, 10)
pos = {tgt: (0.8, 5.0)}
mids = sorted({r["chain"][1] for r in recs if len(r["chain"]) > 2})
for mid in mids:
    pos[mid] = (3.3, 5.0)
ys = np.linspace(9.2, 0.8, len(recs))
for r, y in zip(recs, ys):
    pos[r["gene"]] = (7.7, y)
norm = Normalize(-26, 26)
drawn = set()
edb = {}
for r in recs:
    for (a_, s_, b_, db) in r["edges"]:
        sg = +1 if s_ == "->" else -1
        if (a_, b_) in drawn:
            continue
        drawn.add((a_, b_))
        edge(axc, pos[a_], pos[b_], sg)
        edb[(a_, b_)] = db
axc.scatter([pos[tgt][0]], [pos[tgt][1]], s=330, c="white", edgecolors=GREY900, linewidths=1.0, zorder=4)
lock(axc, pos[tgt][0], pos[tgt][1] + 1.55)
axc.text(pos[tgt][0], pos[tgt][1] - 1.35, tgt, ha="center", va="top", fontsize=FS, fontweight="bold")
for mid in mids:
    axc.scatter([pos[mid][0]], [pos[mid][1]], s=330, c=GREY200, edgecolors=GREY500, linewidths=LW, zorder=4)
    axc.text(*pos[mid], mid, ha="center", va="center", fontsize=FS, zorder=5)
nok = 0
for r in recs:
    x, y = pos[r["gene"]]
    ok = (r["z"] < 0) == (r["pred"] == "-"); nok += ok
    axc.scatter([x], [y], s=330, c=[CMAP(norm(r["z"]))], edgecolors=GGREEN if ok else GRED, linewidths=1.5, zorder=4)
    axc.scatter([x], [y], s=26, marker="v" if r["pred"] == "-" else "^", c=GREY900, zorder=5)
    axc.text(x + 0.85, y + 0.5, r["gene"], ha="left", va="center", fontsize=FS, fontweight="bold")
    axc.text(x + 0.85, y - 0.5, f"z={r['z']:+.1f}", ha="left", va="center", fontsize=FS, color=GGREEN if ok else GRED)
print("sign-correct", nok, "of", len(recs))
grp = {}
for (a_, b_), db in edb.items():
    if a_ != tgt:
        grp.setdefault(db.replace(",", "+"), []).append(b_)
ev = f"{tgt}->{mids[0]}: " + edb[(tgt, mids[0])].replace(",", "+") + "; " + "; ".join(f"{mids[0]}->" + "/".join(v) + ": " + k for k, v in grp.items())
print("evidence:", ev)
import textwrap
fig.text(0.005, 0.115, "\n".join(textwrap.wrap(ev, 50)), fontsize=FS, color=GREY700, va="center", ha="left", linespacing=1.05)
fig.text(0.215, 0.04, "fill: observed z; marker: predicted sign\nring: green = sign correct, red = sign wrong", fontsize=FS, color=GREY700, va="center", linespacing=1.0)
cax = fig.add_axes([0.03, 0.045, 0.15, 0.014])
cb = fig.colorbar(plt.cm.ScalarMappable(norm, CMAP), cax=cax, orientation="horizontal")
cb.set_ticks([-25, 0, 25]); cb.ax.tick_params(labelsize=FS, length=2, pad=1); cb.outline.set_visible(False)

# ---- (d)
axd = fig.add_axes([0.62, 0.12, 0.2, 0.38]); axd2 = fig.add_axes([0.89, 0.12, 0.105, 0.38])
GX = 2.6; HX = 1.2   # pair offset, within-pair spacing   # x offset of the RPE1 pair
xs, labs = [], []
d_res = {}
for j, (ln, nm) in enumerate(LINES):
    a = acc_table(sc[ln], ["1", "2", "3", "4+"]); a = a[a.kind == "real"]
    for h, (name, hs) in enumerate((("1", ["1"]), ("$\\geq$2", ["2", "3", "4+"]))):
        n = int(a[a.hop.isin(hs)].n.sum()); hit = a[a.hop.isin(hs)].hit.sum()
        p, lo, hi = wilson(hit, n); x = j * GX + HX * h
        axd.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=3.2, color=GBLUE, elinewidth=LW, zorder=3)
        if h == 0:   # hop 1: above the error bar
            axd.text(x, hi + 0.015, f"n={n}", ha="center", va="bottom", fontsize=FS)
        else:        # hop >= 2: below the 0.5 reference line (the rightmost one right-aligned to stay inside)
            last = j == 1
            axd.text(x + (0.45 if last else 0), 0.488, f"n={n}", ha="right" if last else "center", va="top", fontsize=FS)
        d_res[(ln, name)] = (p, lo, hi, n)
        xs.append(x); labs.append(name)
axd.axhline(0.5, color=GREY900, lw=LW_THIN, ls=(0, (3, 2)), zorder=1)
axd.set_xticks(xs); axd.set_xticklabels(["hop 1", "\u2265 2"] * 2); axd.set_ylim(0.3, 1.0); axd.set_xlim(-0.6, GX + HX + 0.5)
axd.grid(axis="x", visible=False)
import matplotlib.transforms as mtrans
for j, (ln, nm) in enumerate(LINES):   # group label centred under each pair
    axd.annotate(nm, xy=(j * GX + HX / 2, 0), xycoords=mtrans.blended_transform_factory(axd.transData, axd.transAxes),
                 xytext=(0, -2.0 - 2.5 - FS - 2.5), textcoords="offset points", ha="center", va="top", fontsize=FS, color=GREY900)
axd.tick_params(labelsize=FS, pad=1.5); axd.set_ylabel("sign accuracy", fontsize=FS, labelpad=1)
rng = np.random.default_rng(0)
pc = {}
for j, (ln, nm) in enumerate(LINES):
    c = sc[ln]; c = c[c.hop == "corr"].hit.astype(float).to_numpy()
    pc[ln] = (c.mean(), len(c))
    axd2.scatter(j + rng.uniform(-0.22, 0.22, len(c)), c, s=4, color=GBLUE_L, lw=0, zorder=2)
    axd2.plot([j - 0.3, j + 0.3], [c.mean()] * 2, color=GBLUE, lw=1.5, zorder=3)
axd2.axhline(0, color=GREY900, lw=LW_THIN, ls=(0, (3, 2)), zorder=1)
axd2.set_xticks([0, 1]); axd2.set_xticklabels(["K562", "RPE1"]); axd2.set_xlim(-0.45, 1.45); axd2.grid(axis="x", visible=False)
axd2.set_ylim(-0.5, 0.5); axd2.tick_params(labelsize=FS, pad=1.5)
print("panel d", d_res, pc)
axd2.set_title("Pearson r", fontsize=FS_TITLE, pad=3, color=GREY900, loc="right")

# ---- panel letters
for lab, x, y in (("a", 0.0025, 0.992), ("b", 0.375, 0.992), ("c", 0.0025, 0.595), ("d", 0.57, 0.595)):
    fig.text(x, y, f"({lab})", fontsize=FS_LETTER, fontweight="bold", va="top", ha="left", color=GREY900)

plt.rcParams["savefig.bbox"] = "standard"
for ext in (".pdf", ".png"):
    fig.savefig(OUT / f"fig_mechanism{ext}", dpi=600, pad_inches=0)

# ---- caption generated from the plotted numbers
def f2(x): return f"{x:.2f}"
B = lambda ln, k, h: b_res[(ln, k, h)][0]
cap = (
    "Fig. X. Mechanism of the clamped steady-state propagation, and how far its signs can be trusted. "
    "(a) Toy network computed with the iPerturb functions build_W (spectral radius 0.8) and clamped_response: A activates B, B represses C, B activates D, "
    "D represses B. CRISPRi clamps A (lock) at dx = -1; the predicted responses are "
    f"B {toy[1]:+.2f}, C {toy[2]:+.2f} and D {toy[3]:+.2f} (blue = down, red = up; fill scaled to the largest magnitude), i.e. B down, C up, D down. "
    "(b) Sign accuracy of the steady-state prediction among responding genes (|z| > 3) on held-out targets, by shortest-path distance from the perturbed gene, "
    "for the real signed network (blue), permuted edge signs (grey) and degree-preserving rewired edges (yellow). Hits and counts are pooled over the five draws "
    "of each control kind; error bars are Wilson 95% intervals; the dashed line marks 0.5; the number at hop 1 is the count of responding genes for the real network "
    f"(K562 {b_res[('K562_essential','real','1')][3]}, RPE1 {b_res[('rpe1','real','1')][3]}). "
    "The controls lie above 0.5 because most responding genes move in the same direction, so a prediction dominated by one sign scores above chance; "
    "the controls, not 0.5, are the relevant reference. "
    f"At hop 1 the real signs reach {f2(B('K562_essential','real','1'))} (K562) and {f2(B('rpe1','real','1'))} (RPE1) against "
    f"{f2(B('K562_essential','signperm','1'))} and {f2(B('rpe1','signperm','1'))} for permuted signs and "
    f"{f2(B('K562_essential','rewired','1'))} and {f2(B('rpe1','rewired','1'))} for rewired edges. "
    "At hops 2 to 4 the real signs exceed the permuted-sign control by no more than 0.01 in any cell of either line. "
    f"(c) Mechanism report for {tgt} (K562, fold 0): the five reached genes with the largest |z|, and the strongest signed path to each "
    "(all run through MYC at hop 2; all edges are activations); the supporting databases of each edge are listed under the diagram. "
    "Node fill shows the observed z, the triangle the predicted sign (all five predicted down), and the ring marks whether the sign is correct (green) or wrong (red): "
    + ", ".join(r["gene"] for r in recs if (r["z"] < 0) == (r["pred"] == "-")) + " correct; "
    + ", ".join(r["gene"] for r in recs if (r["z"] < 0) != (r["pred"] == "-")) + " wrong. "
    "(d) Real-network sign accuracy at hop 1 and at hops 2 and above with Wilson 95% intervals "
    f"(K562 {f2(d_res[('K562_essential','1')][0])}, n = {d_res[('K562_essential','1')][3]}, and {f2(d_res[('K562_essential','$\\geq$2')][0])}, n = {d_res[('K562_essential','$\\geq$2')][3]}; "
    f"RPE1 {f2(d_res[('rpe1','1')][0])}, n = {d_res[('rpe1','1')][3]}, and {f2(d_res[('rpe1','$\\geq$2')][0])}, n = {d_res[('rpe1','$\\geq$2')][3]}). "
    "Right: Pearson correlation between predicted and observed change over reached genes for each held-out target (dots) with the mean as a bar "
    f"(K562 mean {pc['K562_essential'][0]:.3f} over {pc['K562_essential'][1]} targets; RPE1 mean {pc['rpe1'][0]:.3f} over {pc['rpe1'][1]} targets). "
    "Only signed curated edges are used; association edges are excluded.\n")
(OUT / "fig_mechanism.caption.txt").write_text(cap, encoding="utf8")
print("panel b:")
for k_, v in b_res.items():
    print(k_, [round(x, 3) if isinstance(x, float) else x for x in v])
