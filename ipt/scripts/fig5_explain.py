"""Fig. 5: faithfulness of neighbour attributions (a) and three predefined-rule case studies on K562 fold 0 (b-d)."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from ipt.viz.style import *
from ipt.metrics import paired_bootstrap
from ipt.graph.l1 import load_l1
apply_style()
ART = ROOT / "artifacts"; CACHE = Path("D:/iPerturb/lnbi/cache"); OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LINE = "K562_essential"

# ---------- (a) faithfulness
g6 = pd.read_csv(ART / "scores" / "g6_explain.tsv", sep="\t")
Ks = (1, 2, 3, 5)
resa = {}
for line in ("K562_essential", "rpe1"):
    resa[line] = []
    for K in Ks:
        d = g6[(g6.line == line) & (g6.K == K)].groupby("target")[["drop_top", "drop_rand"]].mean()
        r = paired_bootstrap(d.drop_top.to_numpy(), d.drop_rand.to_numpy(), 10000, 0)
        resa[line].append((r["mean_diff"], r["ci_low"], r["ci_high"], len(d)))

# ---------- data for (b-d)
z = np.load(ART / "data" / f"{LINE}.npz", allow_pickle=True)
genes = [str(g).upper() for g in z["genes"]]; gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
perts = [str(p).upper() for p in z["perts"]]; prow = {p: i for i, p in enumerate(perts)}
Yall, tix = z["delta"], z["target_idx"]
folds = json.load(open(ART / "folds" / f"{LINE}.json")); f = folds[0]
test = [p.upper() for p in f["test"]]
sc = pd.read_csv(ART / "scores" / f"g4_{LINE}.tsv", sep="\t")
s0 = sc[(sc.fold == 0)]
sm = s0[(s0.model == "smooth_L1+L3") & (s0.graph == "real") & (s0.svd == "self")].set_index("target").pcc_c
b2 = s0[s0.model == "B2_linear"].set_index("target").pcc_c
gain = (sm - b2.reindex(sm.index)).dropna()
gain.index = [t.upper() for t in gain.index]
l1 = load_l1(CACHE); st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
def pairs(a_, b_):
    m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
    return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)
e = np.unique(np.sort(np.concatenate([pairs(l1.source.str.upper(), l1.target.str.upper()),
                                      pairs(st.a.str.upper(), st.b.str.upper())]), 1), axis=0)
A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G)); A = (A + A.T).tocsr(); A.data[:] = 1
nbrs = lambda t: A.indices[A.indptr[t]:A.indptr[t + 1]]
tgt_idx = {p: int(tix[prow[p]]) for p in test}
succ = gain.idxmax(); fail = gain.idxmin()
nonb = sorted(p for p in test if len(nbrs(tgt_idx[p])) == 0)
print("n test", len(test), "n no-neighbour", len(nonb), "first", nonb[:3])
cases = [("b", "success", succ), ("c", "failure", fail), ("d", "no neighbours", nonb[0])]

# model rebuild (as g6_explain.py)
cfg = pd.read_csv(ART / "scores" / f"g4cfg_{LINE}.tsv", sep="\t")
c = cfg[(cfg.fold == 0) & (cfg.variant == "smooth_L1+L3|real|self")].iloc[0]
k, al = int(c.k), float(c.alpha)
idx = {s: np.array([prow[p.upper()] for p in f[s]]) for s in ("train", "test")}
Ytr = Yall[idx["train"]].astype(np.float64); Ttr = tix[idx["train"]]
tm = Ytr.mean(0)
_, _, Vt = np.linalg.svd(Ytr - tm, full_matrices=False); Gk = Vt[:k].T
deg = np.asarray(A.sum(1)).ravel(); Ahat = sp.diags(1 / np.maximum(deg, 1)) @ A
emb = (1 - al) * Gk + al * (Ahat @ Gk); P = emb[Ttr]
W = np.linalg.solve(P.T @ P + 1e-2 * np.eye(k), P.T @ ((Ytr - tm) @ Gk))
def pred_from(t, nb):
    e_t = (1 - al) * Gk[t] + (al * Gk[nb].mean(0) if len(nb) else 0 * Gk[t])
    return tm + e_t @ W @ Gk.T
preds = np.load(ART / "preds" / f"g4_{LINE}.npz")
PB2 = preds["f0|B2_linear"].astype(np.float64); PSM = preds["f0|smooth_L1+L3|real|self"].astype(np.float64)
obsT = Yall[idx["test"]].astype(np.float64)

info = {}
for L, kind, name in cases:
    r = test.index(name); t = tgt_idx[name]; obs = obsT[r]
    order = np.argsort(-np.abs(obs)); sel = [j for j in order if j != t][:40]
    o, pb, ps = obs[sel], PB2[r][sel], PSM[r][sel]
    nb = nbrs(t); top = []
    if len(nb):
        full = pred_from(t, nb)
        attr = np.array([(full - pred_from(t, np.delete(nb, j))) @ (full - tm) for j in range(len(nb))])
        top = [genes[nb[j]] for j in np.argsort(-attr)[:3]]
    info[L] = dict(name=name, kind=kind, o=o, pb=pb, ps=ps, rb=np.corrcoef(o, pb)[0, 1], rs=np.corrcoef(o, ps)[0, 1],
                   nn=len(nb), top=top, gain=float(gain[name]))
    print(L, name, "gain %.4f" % gain[name], "nn", len(nb), top, "r_B2 %.3f r_sm %.3f" % (info[L]["rb"], info[L]["rs"]))

# ---------- figure (122 mm = LNCS text width; positions in mm)
FW, FH = 122.0, 105.0
fig = plt.figure(figsize=(FW * MM, FH * MM))
def ax_mm(x0, top, w, h):
    return fig.add_axes([x0 / FW, 1 - (top + h) / FH, w / FW, h / FH])
AXH, TOP1, TOP2 = 28.0, 11.0, 59.5
XL, WL, XR, WR = 18.0, 40.0, 79.0, 41.0
axa = ax_mm(XL, TOP1, WL, AXH)
for line, col, lab, dx in (("K562_essential", GBLUE, "K562", -0.07), ("rpe1", GGREEN, "RPE1", 0.07)):
    m, lo, hi, n = zip(*resa[line]); m, lo, hi = map(np.array, (m, lo, hi))
    x = np.array(Ks) + dx
    axa.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=3.2, lw=LW, elinewidth=LW, capsize=1.5,
                 capthick=LW, label=f"{lab} (n={n[0]})")
axa.axhline(0, color=GGREY, lw=LW_THIN, ls="--")
axa.set_xticks(Ks); axa.set_xlabel("K removed neighbours")
axa.set_ylabel("drop in centred Pearson:\ntop-K minus random-K\nremoval", linespacing=1.1)
axa.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1)
axa.set_title("Faithfulness (mean, 95% CI)", fontsize=FS_TITLE, loc="left", pad=3)
for ax_, L, x0, top in ((ax_mm(XR, TOP1, WR, AXH), "b", XR, TOP1), (ax_mm(XL, TOP2, WL, AXH), "c", XL, TOP2),
                        (ax_mm(XR, TOP2, WR, AXH), "d", XR, TOP2)):
    d = info[L]
    ax_.scatter(d["o"], d["pb"], s=6, color=GRED_L, alpha=0.9, lw=0, label="B2 linear")
    ax_.scatter(d["o"], d["ps"], s=6, color=GBLUE, alpha=0.9, lw=0, label="smooth L1+L3")
    lim = np.abs(np.concatenate([d["o"], d["pb"], d["ps"]])).max() * 1.05
    ax_.plot([-lim, lim], [-lim, lim], color=GGREY_L, lw=LW_THIN, zorder=0)
    ax_.set_xlim(-lim, lim); ax_.set_ylim(-lim, lim)
    ax_.set_xlabel("observed delta"); ax_.set_ylabel("predicted delta")
    t1 = f"{d['name']} ({d['kind']}), {d['nn']} nbr" + ("" if d["nn"] == 1 else "s")
    t2 = f"r: B2 {d['rb']:.2f}, L1+L3 {d['rs']:.2f}"
    t3 = ("top nbrs: " + ", ".join(d["top"])) if d["top"] else "no association neighbours"
    # title line (8 pt) above two 7-pt detail lines
    ax_.annotate(f"{t2}\n{t3}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                 fontsize=FS_BODY, ha="left", va="bottom", color=GREY900, linespacing=1.1)
    ax_.annotate(t1, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3 + 2 * FS_BODY * 1.1 + 2.0),
                 textcoords="offset points", fontsize=FS_TITLE, ha="left", va="bottom", color=GREY900)
    globals()["ax" + L] = ax_
axb.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.1, markerscale=1.2)
for L, x, top in (("a", 0.3, TOP1), ("b", 64.0, TOP1), ("c", 0.3, TOP2), ("d", 64.0, TOP2)):
    fig_letter(fig, x, top - 9.6, f"({L})", FH)
fig.text(0.3 / FW, 0.3 / FH, "Case rule (K562, fold 0): gain = Pearson(L1+L3) - Pearson(B2); (b) max gain, (c) min gain,\n"
         "(d) alphabetically first test target without neighbours. \nScatter: 40 genes with largest |observed| delta, perturbed gene excluded.",
         fontsize=FS_BODY, ha="left", va="bottom", color=GREY700, linespacing=1.15)
fig.savefig(OUT / "fig5_explain.pdf"); fig.savefig(OUT / "fig5_explain.png", dpi=600)
json.dump({L: dict(name=d["name"], gain=d["gain"], n_nb=d["nn"], top=d["top"], r_b2=d["rb"], r_sm=d["rs"]) for L, d in info.items()},
          open(OUT / "fig5_explain.provenance.json", "w"), indent=1)
