"""Neighbour-ablation faithfulness for the association-smoothing model (L1+L3).

Attribution a_j = <pred(N) - pred(N\\j), pred(N) - base>: how much neighbour j contributes to the predicted shift.
Faithfulness: removing the top-K attributed neighbours should hurt agreement with the OBSERVED profile more than
removing K random neighbours (50 draws). This is an association explanation, not a causal path claim.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
rng = np.random.default_rng(0)
rows = []
for line in ("K562_essential", "rpe1"):
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    genes = [g.upper() for g in z["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    perts = [p.upper() for p in z["perts"]]
    prow = {p: i for i, p in enumerate(perts)}
    Yall, tix = z["delta"], z["target_idx"]
    folds = json.load(open(ART / "folds" / f"{line}.json"))
    cfg = pd.read_csv(ART / "scores" / f"g4cfg_{line}.tsv", sep="\t")
    l1 = load_l1(CACHE)
    st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))

    def pairs(a_, b_):
        m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
        return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)

    e = np.unique(np.sort(np.concatenate([pairs(l1.source, l1.target), pairs(st.a.str.upper(), st.b.str.upper())]), 1), axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    for fo in sorted(cfg.fold.unique()):
        c = cfg[(cfg.fold == fo) & (cfg.variant == "smooth_L1+L3|real|self")].iloc[0]
        k, al = int(c.k), float(c.alpha)
        f = folds[fo]
        idx = {s: np.array([prow[p.upper()] for p in f[s]]) for s in ("train", "valid", "test")}
        Ytr = Yall[idx["train"]].astype(np.float64)
        T = {s: tix[idx[s]] for s in idx}
        tm = Ytr.mean(0)
        _, _, Vt = np.linalg.svd(Ytr - tm, full_matrices=False)
        Gk = Vt[:k].T

        def emb_rows(nb_sets):
            return None

        # ridge map W fitted with the real smoothing (same as g4)
        deg = np.asarray(A.sum(1)).ravel()
        Ahat = sp.diags(1 / np.maximum(deg, 1)) @ A
        emb = (1 - al) * Gk + al * (Ahat @ Gk)
        P = emb[T["train"]]
        W = np.linalg.solve(P.T @ P + 1e-2 * np.eye(k), P.T @ ((Ytr - tm) @ Gk))
        obs = Yall[idx["test"]]

        def pred_from(t, nb):
            e_t = (1 - al) * Gk[t] + (al * Gk[nb].mean(0) if len(nb) else 0 * Gk[t])
            return tm + e_t @ W @ Gk.T

        for r, t in enumerate(T["test"]):
            nb = A.indices[A.indptr[t]:A.indptr[t + 1]]
            if len(nb) < 6 or al == 0:
                continue
            full = pred_from(t, nb)
            attr = np.array([(full - pred_from(t, np.delete(nb, j))) @ (full - tm) for j in range(len(nb))])
            order = np.argsort(-attr)
            obs_r, tt = obs[r:r + 1], np.array([t])

            def score(pred):
                return M.pcc_delta_centered(pred[None, :], obs_r, tm, tt)[0]

            base = score(full)
            for K in (1, 2, 3, 5):
                top = score(pred_from(t, np.delete(nb, order[:K])))
                rnd = np.mean([score(pred_from(t, np.delete(nb, rng.choice(len(nb), K, replace=False)))) for _ in range(50)])
                rows.append(dict(line=line, fold=fo, target=f["test"][r], n_nb=len(nb), K=K, base=base,
                                 drop_top=base - top, drop_rand=base - rnd))
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / "g6_explain.tsv", sep="\t", index=False)
from ipt.metrics import paired_bootstrap
for line in df.line.unique():
    for K in (1, 2, 3, 5):
        d = df[(df.line == line) & (df.K == K)].groupby("target")[["drop_top", "drop_rand"]].mean()
        r = paired_bootstrap(d.drop_top.to_numpy(), d.drop_rand.to_numpy(), 10000, 0)
        print(f"{line} K={K}: drop_top-drop_rand {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] n={r['n']}")
