"""Fig. propagation: (a) clamped signed propagation on a toy GRN, (b) random walk with restart heat strip,
(c) before/after propagation on the real K562 association network. Run from repo root with PYTHONPATH=."""
import itertools, json, pickle, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc
import networkx as nx

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from ipt.viz.style import *
from ipt.explain.mechanism import build_W, clamped_response
from ipt.models.diffusion import normalise, kernel_scores
from ipt.graph.l1 import load_l1
apply_style()
mpl.rcParams["axes.grid"] = False
OUT = ROOT / "paper" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("D:/iPerturb/lnbi/cache")
FW, FH = 122.0, 99.0  # mm
MM = 1 / 25.4
fig = plt.figure(figsize=(FW * MM, FH * MM))


def ax_mm(l, t, w, h, **kw):
    return fig.add_axes([l / FW, 1 - (t + h) / FH, w / FW, h / FH], **kw)


def letter(l, t, s):
    fig.text(l / FW, 1 - t / FH, s, fontsize=9, fontweight="bold", va="top", ha="left")


# ======================= (a) toy signed network, clamped model =======================
names = ["P", "A", "B", "C", "D", "E", "F"]
ix = {n: i for i, n in enumerate(names)}
edges = [("P", "A", +1), ("P", "B", -1), ("A", "C", +1), ("B", "C", -1), ("B", "D", -1),
         ("C", "E", +1), ("D", "F", +1), ("E", "F", -1), ("F", "B", +1)]
src = np.array([ix[a] for a, b, s in edges]); dst = np.array([ix[b] for a, b, s in edges])
sgn = np.array([s for a, b, s in edges], float); conf = np.full(len(edges), 0.9)
RHO = 0.9
W, r_raw, scale = build_W(src, dst, sgn, conf, 7, rho=RHO)
DXG = -1.0   # CRISPRi: perturbed gene held at a reduced level
nodes, steady, hops = clamped_response(W, 0, DXG, hops=3)


def full(v):
    out = np.zeros(7); out[nodes] = v; out[0] = DXG; return out


states = [full(np.zeros(len(nodes)))]
cum = np.zeros(len(nodes))
for k in range(3):
    cum = cum + hops[k]; states.append(full(cum))
states.append(full(steady))
mags = [np.abs(s) for s in states]
VMAX = 0.5   # colour scale saturates at 0.5 |dx|; P itself is |dx|=1 (clamped)
cmapA = LinearSegmentedColormap.from_list("sig", ["#FFFFFF", GBLUE_L, GBLUE])
pos = {"P": (3.2, 12), "A": (8.7, 18), "B": (8.7, 6), "C": (14.2, 18), "D": (14.2, 6), "E": (19.7, 18), "F": (19.7, 6)}
R = 2.3
titles = ["t = 0", "t = 1", "t = 2", "t = 3", "t = \u221e"]
MW, MH, MT = 23.0, 24.0, 4.0
for k in range(5):
    ax = ax_mm(1.0 + k * 24.0, MT, MW, MH); ax.set_xlim(0, MW); ax.set_ylim(0, MH)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(MW / 2, MH - 0.3, titles[k], ha="center", va="top", fontsize=8, fontweight="bold")
    for a, b, s in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        col = GGREEN if s > 0 else GRED
        if (a, b) == ("F", "B"):
            ax.add_patch(FancyArrowPatch((x0, y0 - R * .6), (x1, y1 - R * .6), connectionstyle="arc3,rad=-0.3",
                         arrowstyle="-|>,head_length=2.4,head_width=1.2", mutation_scale=1, color=col, lw=0.8,
                         shrinkA=0, shrinkB=0, zorder=1))
            continue
        d = np.array([x1 - x0, y1 - y0]); L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]])
        p0_ = np.array([x0, y0]) + u * R; p1_ = np.array([x1, y1]) - u * (R + 0.15)
        if s > 0:
            ax.add_patch(FancyArrowPatch(p0_, p1_, arrowstyle="-|>,head_length=2.4,head_width=1.2", mutation_scale=1,
                                         color=col, lw=0.8, shrinkA=0, shrinkB=0, zorder=1))
        else:
            ax.plot([p0_[0], p1_[0]], [p0_[1], p1_[1]], color=col, lw=0.8, zorder=1, solid_capstyle="butt")
            b0, b1 = p1_ + nrm * 1.2, p1_ - nrm * 1.2
            ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color=col, lw=1.1, zorder=1, solid_capstyle="butt")
    for nme in names:
        v = mags[k][ix[nme]]
        c = GBLUE if nme == "P" else cmapA(float(min(v / VMAX, 1.0)))
        ax.add_patch(Circle(pos[nme], R, fc=c, ec=GREY700, lw=0.8, zorder=3))
        ax.text(*pos[nme], nme, ha="center", va="center", fontsize=7, zorder=4,
                color="white" if (nme == "P" or v / VMAX > 0.6) else GREY900)
    lx, ly = pos["P"][0], pos["P"][1] + R + 0.5   # lock marker above P
    ax.add_patch(Rectangle((lx - 1.2, ly), 2.4, 1.8, fc=GGREY, ec=GREY900, lw=0.5, zorder=5))
    ax.add_patch(Arc((lx, ly + 1.8), 1.6, 2.0, theta1=0, theta2=180, ec=GREY900, lw=0.8, zorder=5))
# colour bar + legend strip for (a)
cax = ax_mm(5, 32.5, 20, 2.2)
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmapA, extent=[0, VMAX, 0, 1]); cax.set_yticks([])
cax.set_xticks([0, VMAX]); cax.set_xticklabels(["0", "\u22650.5"], fontsize=7)
cax.tick_params(length=2, pad=1)
for s_ in cax.spines.values(): s_.set_visible(False)
fig.text(26.5 / FW, 1 - 33.6 / FH, "|\u0394x|", fontsize=7, va="center", ha="left")
lg = ax_mm(38, 31.5, 82, 5.0); lg.set_xlim(0, 82); lg.set_ylim(0, 5); lg.axis("off")
lg.add_patch(FancyArrowPatch((0.5, 2.5), (6.5, 2.5), arrowstyle="-|>,head_length=2.4,head_width=1.2", mutation_scale=1, color=GGREEN, lw=0.8))
lg.text(7.5, 2.5, "activates", fontsize=7, va="center")
lg.plot([22, 28], [2.5, 2.5], color=GRED, lw=0.8); lg.plot([28, 28], [1.3, 3.7], color=GRED, lw=1.1)
lg.text(29.5, 2.5, "represses", fontsize=7, va="center")
lg.add_patch(Rectangle((46, 1.2), 2.4, 1.8, fc=GGREY, ec=GREY900, lw=0.5)); lg.add_patch(Arc((47.2, 3.0), 1.6, 2.0, theta1=0, theta2=180, ec=GREY900, lw=0.8))
lg.text(50, 2.5, "clamped P (\u0394x = \u22121)", fontsize=7, va="center")
letter(0.2, 0.2, "(a)")

# ======================= (b) RWR heat strip on the unsigned toy graph =======================
n = 7
Au = np.zeros((n, n))
for a, b, s in edges: Au[ix[a], ix[b]] = Au[ix[b], ix[a]] = 1
Wn = normalise(sp.csr_matrix(Au), "rw").toarray()
alpha = 0.3
p0 = np.zeros(n); p0[0] = 1
P = [p0]
for k in range(1, 60): P.append(alpha * p0 + (1 - alpha) * Wn @ P[-1])
conv = next(k for k in range(1, 60) if np.abs(P[k] - P[k - 1]).max() < 1e-3)
H = np.array(P[:10]).T
axb = ax_mm(10.5, 50, 29, 27)
cmb = LinearSegmentedColormap.from_list("rwr", ["#FFFFFF", GBLUE_L, GBLUE])
axb.imshow(H, aspect="auto", cmap=cmb, vmin=0, vmax=1)
axb.set_xticks(range(10)); axb.set_yticks(range(n)); axb.set_yticklabels(names)
axb.tick_params(length=1.5, pad=1.5, labelsize=7)
axb.set_xticks(np.arange(-.5, 10, 1), minor=True); axb.set_yticks(np.arange(-.5, n, 1), minor=True)
axb.grid(which="minor", color="white", lw=0.6); axb.tick_params(which="minor", length=0)
for s_ in axb.spines.values(): s_.set_visible(False)
axb.set_xlabel("iteration k", fontsize=7, labelpad=1.5)
cb = ax_mm(41.5, 50, 2.4, 27)
cb.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap=cmb, extent=[0, 1, 0, 1])
cb.set_xticks([]); cb.yaxis.tick_right(); cb.set_yticks([0, 1]); cb.set_yticklabels(["0", "1"], fontsize=7)
cb.tick_params(length=1.5, pad=1)
for s_ in cb.spines.values(): s_.set_visible(False)
fig.text(47.8 / FW, 1 - 63.5 / FH, "p", fontsize=7, rotation=90, va="center", ha="center")
fig.text(25.5 / FW, 1 - 46.0 / FH, "unsigned RWR, \u03b1 = 0.3", fontsize=7, ha="center", va="center")
fig.text(25.5 / FW, 1 - 86.5 / FH, f"max change < 10$^{{-3}}$\nfirst at k = {conv}",
         fontsize=7, ha="center", va="top", linespacing=1.1)
letter(0.2, 39.0, "(b)")

# ======================= (c) real K562 association network =======================
genes = np.char.upper(np.load(ROOT / "artifacts/data/K562_essential.npz", allow_pickle=True)["genes"]).astype(str)
gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
sp_ = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
l1 = load_l1(CACHE)
rows, cols = [], []
for a, b in list(zip(sp_.a.str.upper(), sp_.b.str.upper())) + list(zip(l1.source.str.upper(), l1.target.str.upper())):
    if a in gi and b in gi and a != b: rows.append(gi[a]); cols.append(gi[b])
A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(G, G)).tocsr()
A = ((A + A.T) > 0).astype(float).tocsr()
deg = np.asarray(A.sum(1)).ravel()
test = [t.upper() for t in json.load(open(ROOT / "artifacts/folds/K562_essential.json"))[0]["test"]]
T = [t for t in test if t in gi and deg[gi[t]] >= 15]
A2 = ((A @ A + A) > 0).tocsr()
nb2 = {t: set(A2[gi[t]].indices) - {gi[t]} for t in T}
best = max(((len(nb2[x] & nb2[y]) / len(nb2[x] | nb2[y]), x, y) for x, y in itertools.combinations(T, 2) if x[:3] != y[:3]))
jac, s1n, s2n = best
s1, s2 = gi[s1n], gi[s2n]
S = kernel_scores(A, [s1, s2], "rwr", alpha=0.5, norm="rw")
mx = S.max(0).copy(); mx[[s1, s2]] = -np.inf
sub = np.array([s1, s2] + [int(i) for i in np.argsort(-mx) if i not in (s1, s2)][:45])
assert len(sub) == 47
Asub = A[sub][:, sub].toarray()
Ssub = S[:, sub]
ns = np.arange(len(sub)) >= 2
from scipy.sparse.csgraph import shortest_path
HOP = shortest_path(A, unweighted=True, indices=[s1, s2], directed=False)   # (2, G) hop distance from each seed
stats = []
for r_, sd in enumerate((s1, s2)):
    h = HOP[r_]; sc = S[r_]; K = int((h == 1).sum())
    tot = sc.sum(); nsm = tot - sc[sd]
    top = np.argsort(-np.where(np.arange(G) == sd, -np.inf, sc))[:2 * K]
    stats.append(dict(K=K, rec=int((h[top] >= 2).sum()), seed=sc[sd] / tot,
                      m1=sc[h == 1].sum() / nsm, m2=sc[h == 2].sum() / nsm, m3=sc[h >= 3].sum() / nsm,
                      unreach=int(np.isinf(h).sum()), tot=tot))
K1, K2 = stats[0]["K"], stats[1]["K"]
nbr1 = HOP[0][sub] == 1; nbr2 = HOP[1][sub] == 1
ns = np.arange(len(sub)) >= 2
lr = np.log(Ssub[0] + 1e-12) - np.log(Ssub[1] + 1e-12)
lt = np.log(np.minimum(Ssub[0], Ssub[1]) + 1e-12)
def mm(v):
    o = np.zeros(len(v)); o[2:] = np.argsort(np.argsort(v[2:])) / (len(v) - 3.0); return o
XY = np.column_stack([0.08 + 0.84 * mm(lr), 0.03 + 0.78 * mm(lt)])
XY[0] = (0.0, 1.0); XY[1] = (1.0, 1.0)
shared = np.minimum(Ssub[0], Ssub[1]); shared[:2] = -1
top_sh = [int(i) for i in np.argsort(-shared)[:5]]
CB, CY, CG = GBLUE, GYELLOW, GGREEN


def colours(f1, f2):
    return [CG if (a and b) else CB if a else CY if b else "white" for a, b in zip(f1, f2)]


from matplotlib.colors import LogNorm
cmc = LinearSegmentedColormap.from_list("rwrc", ["#FFFFFF", GBLUE_L, GBLUE, "#1A3F8F"])
smax = np.maximum(Ssub[0], Ssub[1])
vlo, vhi = smax[2:].min(), smax[2:].max()
norm = LogNorm(vmin=vlo, vmax=vhi)
colb = colours(nbr1, nbr2)
for j, title in enumerate(["before propagation", "after propagation"]):
    ax = ax_mm(55 + j * 34, 50, 33, 29); ax.set_xlim(-.1, 1.1); ax.set_ylim(-.05, 1.08); ax.axis("off")
    ax.text(0.5, 1.14, title, ha="center", va="bottom", fontsize=8, fontweight="bold", transform=ax.transAxes)
    for a, b in zip(*np.nonzero(np.triu(Asub))):
        if a < 2: ax.plot(XY[[a, b], 0], XY[[a, b], 1], color=GREY300, lw=0.25, zorder=1)
    for i in range(len(sub)):
        seed = i < 2
        if seed: c = CB if i == 0 else CY
        elif j == 0: c = colb[i]
        else: c = cmc(norm(smax[i]))
        ax.scatter(*XY[i], s=34 if seed else 14, c=[c], edgecolors=GREY900 if seed else GREY700,
                   linewidths=0.9 if seed else 0.4, zorder=3)
    for i, t, ha, dx, dy in [(0, s1n, "left", .04, -.06), (1, s2n, "right", -.04, -.06)]:
        ax.text(XY[i, 0] + dx, XY[i, 1] + dy, t, fontsize=7, ha=ha, va="center", zorder=5, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.8))
    for k, i in enumerate(sorted(top_sh, key=lambda q: XY[q, 0])):
        ly = 0.36 - 0.085 * k
        ax.plot([XY[i, 0], -0.1], [XY[i, 1], ly], color=GREY700, lw=0.4, zorder=4)
        ax.text(-0.065, ly, genes[sub[i]], fontsize=7, ha="left", va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.7))
cbx = ax_mm(93, 80.8, 19, 1.8)
cbx.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmc, extent=[np.log10(vlo), np.log10(vhi), 0, 1])
cbx.set_yticks([]); cbx.set_xticks([np.log10(vlo), np.log10(vhi)])
cbx.set_xticklabels([f"{vlo:.0e}".replace("e-0", "e-"), f"{vhi:.0e}".replace("e-0", "e-")], fontsize=7)
cbx.tick_params(length=1.5, pad=1)
for s_ in cbx.spines.values(): s_.set_visible(False)
fig.text(113.5 / FW, 1 - 81.7 / FH, "score", fontsize=7, va="center", ha="left")
lgc = ax_mm(42, 86.5, 79, 10); lgc.axis("off"); lgc.set_xlim(0, 79); lgc.set_ylim(0, 10)
for k, (c, t) in enumerate([(CB, f"{s1n} neighbour"), (CY, f"{s2n} neighbour"), (CG, "both"), ("white", "other")]):
    x = [0, 22, 45, 58][k]
    lgc.scatter(x + 1.2, 8.6, s=16, c=[c], edgecolors=GREY700, linewidths=0.4, zorder=3)
    lgc.text(x + 3, 8.6, t, fontsize=7, va="center")
for k, (nm, st) in enumerate([(s1n, stats[0]), (s2n, stats[1])]):
    lgc.text(0, 5.0 - 3.3 * k, f"{nm}: {st['rec']}/{2*st['K']} of top-2K beyond hop 1; mass hop 1/2/3+ "
             f"{100*st['m1']:.0f}/{100*st['m2']:.0f}/{100*st['m3']:.0f}%", fontsize=7, va="center")
letter(52.5, 39.0, "(c)")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_propagation.{ext}", dpi=600, bbox_inches=None, pad_inches=0)

# ---------- caption (numbers computed above) ----------
cap = (f"Network propagation, from a toy regulatory network to a real association network. "
       f"(a) Signed directed toy network of seven nodes (green arrows activate, red bars repress; all edge confidences 0.9) with one "
       f"CRISPRi-perturbed gene P held (clamped, lock) at a reduced level (\u0394x = \u22121) at every step. Panels show the predicted "
       f"|\u0394x| of the six other nodes after t = 0, 1, 2, 3 hops (cumulated hop contributions) and at the steady state (t = \u221e) of the clamped "
       f"model of ipt/explain/mechanism.py, with the weight matrix rescaled to spectral radius \u03c1 = {RHO} (raw radius {r_raw:.2f}); "
       f"node colour is |\u0394x| on a white-to-blue scale saturating at 0.5, and P is drawn in solid blue. "
       f"(b) The same toy graph with unsigned undirected edges: random walk with restart from P (restart probability \u03b1 = 0.3, "
       f"p_k = \u03b1 p_0 + (1 \u2212 \u03b1) W p_(k\u22121), W = A D^(\u22121)) for iterations 0\u20139; rows are nodes, columns iterations, colour is the probability p. "
       f"The largest per-node change between consecutive iterations first falls below 1e-3 at k = {conv}. "
       f"(c) K562 association network (STRING plus curated pairs, undirected, unsigned; {G} genes). Seeds are the fold-0 test targets {s1n} (blue) and {s2n} (yellow), "
       f"chosen as the pair with different three-letter gene-name prefixes, each with at least 15 neighbours, and the largest Jaccard overlap of two-hop neighbourhoods ({jac:.2f}). "
       f"The subnetwork is the two seeds plus the 45 genes with the highest maximum random-walk-with-restart score (\u03b1 = 0.5). "
       f"Left: genes that are direct (one-hop) neighbours of {s1n} (blue), of {s2n} (yellow) or of both (green); all other genes white. "
       f"Right: the same genes coloured by the larger of the two seeds' propagated scores (random walk with restart, \u03b1 = 0.5, W = A D^(\u22121)) on a logarithmic scale. "
       f"Direct neighbours: {K1} for {s1n}, {K2} for {s2n}. "
       f"Over all {G} genes, {stats[0]['rec']} of the {2*K1} highest-scoring genes of {s1n} (seed excluded) are not direct neighbours (reached at hop 2 or more), and {stats[1]['rec']} of the {2*K2} highest-scoring genes of {s2n}. "
       f"Share of the propagated mass on non-seed genes at hop 1, hop 2 and hop 3 or more: "
       f"{s1n} {100*stats[0]['m1']:.1f}%, {100*stats[0]['m2']:.1f}%, {100*stats[0]['m3']:.1f}%; "
       f"{s2n} {100*stats[1]['m1']:.1f}%, {100*stats[1]['m2']:.1f}%, {100*stats[1]['m3']:.1f}%. "
       f"The seed itself holds {100*stats[0]['seed']:.1f}% ({s1n}) and {100*stats[1]['seed']:.1f}% ({s2n}) of the total mass. "
       f"Node positions are a layout, not geometry: horizontal position is the rank of log(score from {s1n} / score from {s2n}) and vertical position the rank of the smaller of the two scores; only edges incident to a seed are drawn. "
       f"Seeds are outlined; labelled are the seeds and the five non-seed nodes with the highest minimum of the two scores. "
       "")
(OUT / "fig_propagation.caption.txt").write_text(cap, encoding="utf-8")
print("pair", s1n, s2n, jac, "conv", conv, "raw rho", r_raw)
print(stats)
