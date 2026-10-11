"""Fig. applications: iPerturb counterpart of Cowen et al. 2017 Fig. 4.
(a) a ribosomal module found by propagation + its shared response matrix, (b) a similarity network of fold-0 test
perturbations from propagated predictions, (c) integration of tier graphs (schematic + fitted NNLS weights).
Run from repo root with PYTHONPATH=."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from ipt.viz.style import *
from ipt.models.diffusion import kernel_scores, propagate_responses
from ipt.graph.l1 import load_l1
apply_style()
mpl.rcParams["savefig.bbox"] = "standard"
mpl.rcParams["axes.grid"] = False
OUT = ROOT / "paper" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("D:/iPerturb/lnbi/cache")
FW, FH = 122.0, 104.0
MM = 1 / 25.4
fig = plt.figure(figsize=(FW * MM, FH * MM))
CAP = []


def ax_mm(l, t, w, h, **kw):
    return fig.add_axes([l / FW, 1 - (t + h) / FH, w / FW, h / FH], **kw)


def letter(l, t, s):
    fig.text(l / FW, 1 - t / FH, s, fontsize=9, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------- data
z = np.load(ROOT / "artifacts/data/K562_essential.npz", allow_pickle=True)
genes = np.char.upper(z["genes"].astype(str)); G = len(genes)
perts = np.char.upper(z["perts"].astype(str)); D = z["delta"].astype(np.float64)
gi = {g: i for i, g in enumerate(genes)}
fold = json.load(open(ROOT / "artifacts/folds/K562_essential.json"))[0]
train = [t.upper() for t in fold["train"]]; test = [t.upper() for t in fold["test"]]
prow = {p: i for i, p in enumerate(perts)}
sp_ = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
l1 = load_l1(CACHE)
pairs = list(zip(sp_.a.str.upper(), sp_.b.str.upper())) + list(zip(l1.source.str.upper(), l1.target.str.upper()))
r, c = [], []
for a, b in pairs:
    if a in gi and b in gi and a != b:
        r += [gi[a], gi[b]]; c += [gi[b], gi[a]]
A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(G, G)).tocsr(); A.data[:] = 1.0
deg = np.asarray(A.sum(1)).ravel()


def isrib(s): return s.startswith("RPL") or s.startswith("RPS")


# ---------------------------------------------------------------- (a) module
seed = next(t for t in test if isrib(t) and t in gi and deg[gi[t]] >= 10)
S1 = kernel_scores(A, [gi[seed]], "rwr", alpha=0.5, norm="rw")[0]
order = [j for j in np.argsort(-S1) if j != gi[seed]][:30]
sc = S1[order]
nodes = [gi[seed]] + order
isr = np.array([isrib(genes[n]) for n in nodes])
n_rib_nb = int(isr[1:].sum())
ra = ax_mm(2, 11, 50, 44); ra.axis("off")
NN = len(order)
ang = np.array([((i * 7) % NN) / NN * 2 * np.pi + np.pi / 2 for i in range(NN)])   # stride 7 spreads the ranks
xy = np.column_stack([np.cos(ang), np.sin(ang)])
for k in range(NN):
    ra.plot([0, xy[k, 0]], [0, xy[k, 1]], color=GREY300, lw=0.4, zorder=1)
area = 48 * sc / sc.max()
ra.scatter(xy[:, 0], xy[:, 1], s=area, c=[GBLUE if x else GGREY_L for x in isr[1:]], edgecolors="white", linewidths=0.3, zorder=2)
ra.scatter([0], [0], s=170, c=GBLUE, edgecolors=GRED, linewidths=1.2, zorder=3)
ra.text(0, -0.17, seed, fontsize=7, ha="center", va="top", color=GREY900, fontweight="bold", zorder=4)
for k in range(8):   # label the eight highest-scoring neighbours
    c_, s_ = np.cos(ang[k]), np.sin(ang[k])
    ra.text(1.13 * c_, 1.13 * s_, genes[order[k]], fontsize=7, color=GREY900, va="center",
            ha="left" if c_ > 0.2 else ("right" if c_ < -0.2 else "center"), zorder=4)
ra.set_xlim(-1.75, 1.75); ra.set_ylim(-1.3, 1.3); ra.set_aspect("equal")
fig.text(7 / FW, 1 - 3.2 / FH, "Propagated neighbourhood", fontsize=8, ha="left", va="center", color=GREY900)
fig.text(7 / FW, 1 - 7.2 / FH, "(top 30 genes by RWR score)", fontsize=7, ha="left", va="center", color=GREY700)

# matrix
train_rib = [t for t in train if isrib(t) and t in prow]
ncol = min(8, len(train_rib))
rng = np.random.RandomState(0)
cols_sel = sorted(rng.choice(train_rib, ncol, replace=False)) if len(train_rib) > ncol else train_rib
Dr = np.stack([D[prow[t]] for t in train_rib])  # (n_rib, G)
top = np.argsort(-np.abs(Dr).mean(0))[:20]
M = np.stack([D[prow[t]][top] for t in cols_sel], 1)
cc = np.corrcoef(Dr); iu = np.triu_indices(len(train_rib), 1); pear = float(cc[iu].mean())
rb = ax_mm(78, 15, 31, 41)
vm = np.percentile(np.abs(M), 98)
cmap = mpl.colors.LinearSegmentedColormap.from_list("bwr_g", [GBLUE, "white", GRED])
im = rb.imshow(M, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto", interpolation="nearest")
rb.set_yticks(range(20)); rb.set_yticklabels(genes[top], fontsize=7)
rb.set_xticks(range(ncol)); rb.set_xticklabels(cols_sel, rotation=90, fontsize=7)
rb.xaxis.tick_top(); rb.tick_params(length=1.5, pad=1)
for s in rb.spines.values(): s.set_visible(False)
cb = fig.add_axes([(111.5) / FW, 1 - 56 / FH, 2.2 / FW, 41 / FH])
cbar = fig.colorbar(im, cax=cb); cbar.outline.set_visible(False)
cbar.set_ticks([-vm, 0, vm]); cbar.set_ticklabels([f"{-vm:.1f}", "0", f"{vm:.1f}"]); cbar.ax.tick_params(labelsize=7, length=1.5, pad=1)
fig.text(56 / FW, 1 - 3.2 / FH, "Observed Δ, training RPL/RPS perturbations", fontsize=8, ha="left", va="center", color=GREY900)
letter(0.5, 0.5, "(a)")

# ---------------------------------------------------------------- (b) similarity network
fam_rules = [("MRPL", "mito-ribosome", GGREEN), ("MRPS", "mito-ribosome", GGREEN), ("RPL", "ribosome", GBLUE),
             ("RPS", "ribosome", GBLUE), ("PSM", "proteasome", GRED), ("POLR", "RNA polymerase", GYELLOW),
             ("SNRP", "spliceosome", GPURPLE), ("SF3", "spliceosome", GPURPLE), ("PRP", "spliceosome", GPURPLE),
             ("EIF", "translation init.", "#FA7B17")]
famcol = {n: c_ for _, n, c_ in fam_rules}


def family(s):
    for p, n, _ in fam_rules:
        if s.startswith(p): return n
    return None


tr_idx = np.array([gi[t] for t in train if t in gi and t in prow]); tr_rows = np.array([prow[t] for t in train if t in gi and t in prow])
Ytr = D[tr_rows]
te = [t for t in test if t in gi and t in prow]
te_idx = np.array([gi[t] for t in te])
S = kernel_scores(A, list(te_idx), "rwr", alpha=0.5, norm="rw")
Ph, ok = propagate_responses(S, te_idx, tr_idx, Ytr, gamma=0.0, deg=deg, exclude_self=True)
Ph = Ph.astype(np.float64)
keep = np.where(ok)[0]
P = Ph[keep]; P = P - P.mean(0, keepdims=True)
names = [te[i] for i in keep]
nrm = np.linalg.norm(P, axis=1, keepdims=True); Pn = P / np.maximum(nrm, 1e-12)
Cs = Pn @ Pn.T; np.fill_diagonal(Cs, -np.inf)
nn = np.argsort(-Cs, 1)[:, :3]
Gb = nx.Graph(); Gb.add_nodes_from(range(len(names)))
for i in range(len(names)):
    for j in nn[i]: Gb.add_edge(i, int(j))
fams = [family(n) for n in names]
E = list(Gb.edges())
EF = [(u, v) for u, v in E if fams[u] is not None and fams[v] is not None]
same = np.mean([fams[u] == fams[v] for u, v in EF]); same_all = np.mean([(fams[u] is not None and fams[u] == fams[v]) for u, v in E])
fl = [f for f in fams if f is not None]
fr = pd.Series(fl).value_counts(normalize=True)
chance = float((fr ** 2).sum())
# note: chance computed among nodes with a family, per the spec: P(two random family nodes share family)
posb = nx.spring_layout(Gb, seed=1, k=0.35, iterations=300)
rbx = ax_mm(1, 61, 52, 24); rbx.axis("off")
for u, v in E:
    rbx.plot([posb[u][0], posb[v][0]], [posb[u][1], posb[v][1]], color=GREY300, lw=0.35, zorder=1)
X = np.array([posb[i] for i in range(len(names))])
for nmx in [None] + list(dict.fromkeys(f for _, f, _ in fam_rules)):
    idx = [i for i, f in enumerate(fams) if f == nmx]
    if not idx: continue
    rbx.scatter(X[idx, 0], X[idx, 1], s=7 if nmx is None else 13, c=GGREY_L if nmx is None else famcol[nmx],
                edgecolors="white", linewidths=0.25, zorder=2 if nmx is None else 3)
rbx.set_aspect("equal"); rbx.margins(0.03)
# legend (outside network, below title)
leg = [(famcol[n], n) for n in dict.fromkeys(f for _, f, _ in fam_rules)] + [(GGREY_L, "other")]
handles = [plt.Line2D([], [], marker="o", ls="", color=c_, markersize=3.6, markeredgewidth=0) for c_, _ in leg]
fig.legend(handles, [t for _, t in leg], loc="upper left", bbox_to_anchor=(0.0, 1 - 87 / FH), fontsize=7,
           ncol=2, handletextpad=0.1, columnspacing=0.8, labelspacing=0.15, borderpad=0, frameon=False,
           bbox_transform=fig.transFigure)
fig.text(7 / FW, 1 - 59 / FH, "Test-perturbation similarity", fontsize=8, ha="left", va="center", color=GREY900)
letter(0.5, 57, "(b)")

# ---------------------------------------------------------------- (c) integration
DISP = {"L1": "L1 curated", "L3": "L3 STRING", "L4": "L4 COXPRES", "signor": "SIGNOR", "corum": "CORUM",
        "biogrid": "BioGRID", "gtex_coex": "GTEx co-expr.", "chipatlas": "ChIP-Atlas\n(cell line)",
        "chipatlas_rpe1": "ChIP-Atlas\nRPE-1"}
lines = [("K562", "K562_essential", GBLUE), ("RPE1", "rpe1", GRED)]


def read_w(line):
    ws, used = [], []
    for f in range(10):
        p = ROOT / f"artifacts/scores/g12cfg_{line}_p{f}_f{f}.tsv"
        if not p.exists():
            p = ROOT / f"artifacts/scores/g12cfg_{line}_f{f}.tsv"
        if not p.exists(): continue
        d = pd.read_csv(p, sep="\t"); d = d[d.model == "prop[integrated]|nnls"]
        if not len(d): continue
        cm = str(d.iloc[0].comps).split(","); w = [float(x) for x in str(d.iloc[0].w).split(",")]
        ws.append(dict(zip(cm, w))); used.append(f)
    return ws, used


W = {}
for lab_, ln, _ in lines:
    ws, used = read_w(ln)
    if ws:
        df = pd.DataFrame(ws).drop(columns=["B2"], errors="ignore")
        W[lab_] = (df.mean(axis=0), len(ws), used)
tiers = [t for t in DISP if any(t in W[k][0].index and (W[k][0][t] > 0.005 or t == "L1") for k in W)]
# schematic
BX, BW, BH = 56, 28, 7.2
boxes = [("Tier graphs", "L1, L3, L4, ..."), ("Separate\npropagation", "one response per tier"),
         ("Non-negative\nweights", "fitted on validation"), ("Integrated\nresponse", "")]
sa = ax_mm(55, 62, 23, 39); sa.axis("off"); sa.set_xlim(0, 23); sa.set_ylim(41, 0)
ys = [0.5, 10.7, 20.9, 31.1]; bh = 8.6
fc = [GBLUE_L, "#E8F0FE", "#E8F0FE", GBLUE]
for (t1, _), y0, f_ in zip(boxes, ys, fc):
    sa.add_patch(FancyBboxPatch((0.5, y0), 22, bh, boxstyle="round,pad=0,rounding_size=1.2", fc=f_, ec=GBLUE, lw=0.8))
    sa.text(11.5, y0 + bh / 2, t1, ha="center", va="center", fontsize=7, color="white" if f_ == GBLUE else GREY900, linespacing=0.95)
for y0 in ys[:-1]:
    sa.add_patch(FancyArrowPatch((11.5, y0 + bh + 0.1), (11.5, y0 + 10.2 - 0.1), arrowstyle="-|>", mutation_scale=5, color=GREY700, lw=0.8))
sa.set_xlim(0, 23); sa.set_ylim(41, 0)
fig.text(61 / FW, 1 - 59 / FH, "Integration", fontsize=8, ha="left", va="center", color=GREY900)
letter(54, 57, "(c)")
# bars
bx = ax_mm(98, 62, 23, 32)
nt = len(tiers); yy = np.arange(nt)
h = 0.38
for k, (lab_, ln, col_) in enumerate(lines):
    if lab_ not in W: continue
    vals = [W[lab_][0].get(t, np.nan) for t in tiers]
    off = (-h / 2 if k == 0 else h / 2) if len(W) > 1 else 0
    bx.barh(yy + off, vals, height=h if len(W) > 1 else 0.6, color=col_, label=lab_, zorder=3)
bx.set_yticks(yy); bx.set_yticklabels([DISP[t] for t in tiers], fontsize=7)
bx.invert_yaxis(); bx.grid(axis="x", color=GREY200, lw=0.6, zorder=0); bx.set_axisbelow(True)
bx.set_xlabel("NNLS weight", fontsize=7.5, labelpad=1); bx.tick_params(axis="x", labelsize=7, length=2, pad=1); bx.tick_params(axis="y", length=0, pad=2)
bx.legend(loc="upper right", ncol=2, columnspacing=0.8, fontsize=7, handlelength=0.9, handletextpad=0.4, borderaxespad=0.2)
bx.spines["left"].set_visible(False)
bx.spines["bottom"].set_color(GREY300)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_applications.{ext}", dpi=600, bbox_inches=None, pad_inches=0)

# ---------------------------------------------------------------- caption + report
w_txt = "; ".join(f"{k}: " + ", ".join(f"{DISP[t]} {W[k][0].get(t, float('nan')):.2f}" for t in tiers if W[k][0].get(t, 0) > 0.005)
                  for k in W)
missing = [k for k, _, _ in lines if k not in W]
nb_edges = len(E)
cap = (f"Applications of network propagation in iPerturb on the K562 essential-gene screen (fold-0 split; association network = STRING plus curated pairs, "
       f"undirected, {G} genes). (a) Left: random-walk-with-restart (α = 0.5) neighbourhood of the held-out ribosomal-protein seed {seed}: the seed and its "
       f"30 highest-scoring genes; node area of the 30 neighbours is proportional to their propagation score (the seed, which holds most of the mass, is drawn at a fixed size, red outline; only edges from the seed are drawn and nodes sit on a ring, so position carries no information; the eight highest-scoring genes are labelled); {n_rib_nb} of the 30 genes are ribosomal proteins (RPL/RPS, blue), and none are grey. Right: observed "
       f"perturbation responses (Δ) of the 20 genes with the largest mean |Δ| over the {len(train_rib)} training RPL/RPS perturbations (rows), for "
       f"{ncol} of them (columns, randomly chosen, fixed seed); the mean pairwise Pearson correlation of the Δ profiles of all {len(train_rib)} training "
       f"ribosomal perturbations is {pear:.2f}. (b) Each of the {len(names)} fold-0 test perturbations with a propagated prediction is a node; "
       f"predictions come from propagating training responses over the network (α = 0.5, own response excluded), centred on the mean over nodes; each node is "
       f"joined to its 3 nearest neighbours by cosine similarity ({nb_edges} distinct edges). Colours: gene family from the symbol prefix. "
       f"{100*same:.1f}% of the {len(EF)} edges between two nodes that both have a family join nodes of the same family, against {100*chance:.1f}% expected by chance (sum of squared family frequencies among the "
       f"{len(fl)} nodes with a family); counting all {nb_edges} edges, {100*same_all:.1f}% are same-family. Layout: spring layout, fixed seed. (c) Left: schematic of the integration scheme. Right: fitted "
       f"non-negative-least-squares weights of the integrated model per tier graph, mean over the folds present ("
       + "; ".join(f"{k}: {W[k][1]} fold{'s' if W[k][1] > 1 else ''} ({', '.join(map(str, W[k][2]))})" for k in W) + ")"
       + (f"; no weight file exists for {', '.join(missing)}, which is therefore not drawn" if missing else "")
       + f". Weights above 0.005: {w_txt}.")
(OUT / "fig_applications.caption.txt").write_text(cap, encoding="utf-8")
print("seed", seed, "nrib_nb", n_rib_nb, "train_rib", len(train_rib), "pearson", round(pear, 3), "cols", cols_sel)
print("nodes", len(names), "edges", nb_edges, "same", same, "chance", chance, "nfam", len(fl))
print({k: (W[k][1], W[k][2]) for k in W}); print(W["K562"][0].round(3).to_dict())


