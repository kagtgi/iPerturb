"""Addendum I: biologically structured variants (TF-activity state, module bridge, Kuhlman-form logic, gating, program metric)."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1
from ipt.models import baselines as B
from ipt.models.hill import hill4

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.set_num_threads(4)
ALPHAS, TAUS = (0.0, 0.25, 0.5, 0.75, 1.0), (None, 0.1, 0.3, 0.6)

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n_c = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n_c[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
cm = z["ctrl_mean"]
folds = json.load(open(ART / "folds" / f"{a.line}.json"))

# ---- regulon matrix from signed curated edges
l1 = load_l1(CACHE)
l1 = l1[(l1.sign != 0) & l1.source.isin(gi) & l1.target.isin(gi)]
cnt = l1.groupby("source").size()
tfs = [t for t in cnt.index if cnt[t] >= 5]
tix_of = {t: i for i, t in enumerate(tfs)}
T = len(tfs)
R0 = np.zeros((G, T), np.float32)
for s_, t_, sg in zip(l1.source, l1.target, l1.sign):
    if s_ in tix_of:
        R0[gi[t_], tix_of[s_]] += sg / np.sqrt(cnt[s_])
tf_gene = np.array([gi[t] for t in tfs])
tf_expr = cm[tf_gene]

# ---- association graph (L1 + STRING)
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))


def pairs(a_, b_):
    m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
    return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)


e = np.concatenate([pairs(load_l1(CACHE).source, load_l1(CACHE).target), pairs(st.a.str.upper(), st.b.str.upper())])
Aadj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G)).tocsr()
Aadj = ((Aadj + Aadj.T) > 0).astype(np.float32).tocsr()


def projector(tau):
    keep = np.ones(T, bool) if tau is None else tf_expr > tau
    R = R0 * keep[None, :]
    Pm = np.linalg.solve(R.T @ R + np.eye(T, dtype=np.float32), R.T).astype(np.float32)   # (T, G)
    return R, Pm, keep


def masked(Y, tgt):
    Z = Y.copy()
    Z[np.arange(len(tgt)), tgt] = 0.0
    return Z


class Readout(torch.nn.Module):
    """Kuhlman-form combinatorial readout: activators in the numerator, repressors in the denominator."""
    def __init__(self, Rm):
        super().__init__()
        self.register_buffer("Rp", torch.as_tensor(np.clip(Rm, 0, None)))
        self.register_buffer("Rn", torch.as_tensor(np.clip(-Rm, 0, None)))
        self.s0 = torch.nn.Parameter(torch.tensor(0.0))
        self.u = torch.nn.Parameter(torch.zeros(G))
        self.kz = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, dA):
        k = 0.05 + 0.95 * torch.sigmoid(self.kz)
        x = torch.sigmoid(dA)
        H = hill4(x, k, 2.0)
        H0 = hill4(torch.tensor(0.5), k, 2.0)
        h, r = H @ self.Rp.T, H @ self.Rn.T
        h0, r0 = H0 * self.Rp.sum(1), H0 * self.Rn.sum(1)
        g = torch.nn.functional.softplus(self.s0 + self.u)
        return g * ((torch.log1p(h) - torch.log1p(r)) - (torch.log1p(h0) - torch.log1p(r0)))


def fit_readout(Rm, dA_tr, resid_tr, tgt_tr, epochs=250, lam=1e-2):
    torch.manual_seed(0)
    net = Readout(Rm)
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    dA, Yt, tt = torch.as_tensor(dA_tr), torch.as_tensor(resid_tr), torch.as_tensor(tgt_tr)
    mask = torch.ones_like(Yt, dtype=torch.bool)
    mask[torch.arange(len(tt)), tt] = False
    for _ in range(epochs):
        opt.zero_grad()
        loss = (((net(dA) - Yt) ** 2) * mask).sum() / mask.sum() + lam * (net.u ** 2).mean()
        loss.backward()
        opt.step()
    return net


rows, cfgs = [], []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float32) for k in idx}
    Tg = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    b2 = B.fit_linear_select(Y["train"], Tg["train"], Y["valid"], Tg["valid"])
    kk = b2.G.shape[1]
    cv = B.cross_fit_linear(Y["train"], Tg["train"], kk)
    base = {"train": cv, "valid": b2.predict(Tg["valid"]), "test": b2.predict(Tg["test"])}
    mval = np.ones_like(Y["valid"], bool)
    mval[np.arange(len(Tg["valid"])), Tg["valid"]] = False
    nbm = {}
    for i, t in enumerate(Tg["train"]):
        nbm.setdefault(int(t), i)
    variants = {}  # name -> (readout, tau)
    best = {}
    for tau in TAUS:
        R, Pm, keep = projector(tau)
        A_tr = masked(Y["train"], Tg["train"]) @ Pm.T                       # observed training activities
        dA_tr = masked(Y["train"] - base["train"], Tg["train"]) @ Pm.T       # activity of the training residuals
        nets = {"linear": None}
        net = fit_readout(R, dA_tr.astype(np.float32), (Y["train"] - base["train"]).astype(np.float32), Tg["train"])
        nets["kuhlman"] = net

        def predict(split, alpha, readout):
            P0 = base[split].copy()
            a_base = masked(base[split], Tg[split]) @ Pm.T                    # activity of B2 prediction
            nb = np.zeros_like(a_base)
            has = np.zeros(len(P0), bool)
            for r_, t in enumerate(Tg[split]):
                js = [nbm[j] for j in Aadj.indices[Aadj.indptr[t]:Aadj.indptr[t + 1]] if j in nbm]
                if js:
                    nb[r_] = A_tr[js].mean(0)
                    has[r_] = True
            dA = alpha * (nb - a_base) * has[:, None]
            if readout == "linear":
                return P0 + dA @ R.T
            with torch.no_grad():
                return P0 + nets["kuhlman"](torch.as_tensor(dA.astype(np.float32))).numpy()

        for readout in ("linear", "kuhlman"):
            for alpha in ALPHAS:
                pv = predict("valid", alpha, readout)
                ev = float((((pv - Y["valid"]) ** 2) * mval).sum() / mval.sum())
                key = (tau, readout)
                if key not in best or ev < best[key][0]:
                    best[key] = (ev, alpha, (lambda s, al=alpha, ro=readout, pr=predict: pr(s, al, ro)))
        for readout in ("linear", "kuhlman"):
            ev, alpha, fn = best[(tau, readout)]
            variants[f"bio_{readout}|tau={tau}"] = (fn, ev, alpha)
            cfgs.append(dict(fold=fo, variant=f"bio_{readout}|tau={tau}", alpha=alpha, val_mse=ev, n_tf=int(keep.sum())))
        print(f"fold {fo} tau={tau}: TFs kept {int(keep.sum())}/{T}", flush=True)
    # choose tau per readout on validation (gated model); 'ungated' = tau None
    chosen = {}
    for readout in ("linear", "kuhlman"):
        cand = [(variants[f"bio_{readout}|tau={t}"][1], t) for t in TAUS]
        chosen[readout] = min(cand, key=lambda x: x[0])[1]
    obs, tt = Y["test"].astype(np.float64), Tg["test"]
    deg = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    R_all, P_all, _ = projector(None)
    A_tm = masked(Y["train"], Tg["train"]) @ P_all.T
    a_tm = A_tm.mean(0)
    A_obs = masked(obs.astype(np.float32), tt) @ P_all.T - a_tm
    preds = {"B2_linear": base["test"]}
    for readout in ("linear", "kuhlman"):
        preds[f"bio_{readout}_ungated"] = variants[f"bio_{readout}|tau=None"][0]("test")
        preds[f"bio_{readout}_gated"] = variants[f"bio_{readout}|tau={chosen[readout]}"][0]("test")
        cfgs.append(dict(fold=fo, variant=f"chosen_tau_{readout}", alpha=-1, val_mse=np.nan, n_tf=-1 if chosen[readout] is None else int(chosen[readout] * 100)))
    g4 = np.load(ART / "preds" / f"g4_{a.line}.npz")
    preds["iPerturb_smoothing_head"] = g4[f"f{fo}|smooth_L1+L3|real|self"].astype(np.float32)
    preds["iPerturb_nbrmean_head"] = g4[f"f{fo}|nbrmean_L3|real|self"].astype(np.float32)
    for m_, p in preds.items():
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pres = M.pcc_delta_resid(p, obs, tm, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        A_pred = masked(p.astype(np.float32), tt) @ P_all.T - a_tm
        # program-level: Pearson across TF activities (perturbed gene excluded) and sign accuracy on the 20 largest |A_obs|
        prog, sgn = [], []
        for i in range(len(tt)):
            msk = np.ones(T, bool)
            hit = np.where(tf_gene == tt[i])[0]
            msk[hit] = False
            x, y = A_pred[i, msk], A_obs[i, msk]
            prog.append(0.0 if x.std() < 1e-12 or y.std() < 1e-12 else float(np.corrcoef(x, y)[0, 1]))
            top = np.argsort(-np.abs(y))[:20]
            sgn.append(float(np.mean(np.sign(x[top]) == np.sign(y[top]))))
        for i in range(len(tt)):
            rows.append(dict(line=a.line, fold=fo, model=m_, target=f["test"][i], pcc_c=pc[i], pcc_resid=pres[i], top20_c=t20[i], cent_rank=cn[i],
                             prog_r=prog[i], prog_sign=sgn[i]))
    print(f"fold {fo} done; chosen tau {chosen}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / f"g9_{a.line}{a.tag}.tsv", sep="\t", index=False)
pd.DataFrame(cfgs).to_csv(ART / "scores" / f"g9cfg_{a.line}{a.tag}.tsv", sep="\t", index=False)
u = df.groupby(["model", "target"], as_index=False)[["pcc_c", "pcc_resid", "top20_c", "cent_rank", "prog_r", "prog_sign"]].mean()
print(u.groupby("model").mean(numeric_only=True).round(4).to_string())
