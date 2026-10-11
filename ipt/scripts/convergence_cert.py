"""Convergence certificates on the real signed graphs (appendix table): rho(|W|), ||W||_inf, contraction constants, iterations, uniqueness."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.explain.mechanism import abs_spectral_radius, build_W, clamped_response, inf_norm
from ipt.graph.l1 import load_l1
from ipt.models.diffusion import kernel_scores
from ipt.models.graphnet2 import GraphNet2

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
out = {}
for line in ("K562_essential", "rpe1"):
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    genes = [g.upper() for g in z["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    cm = z["ctrl_mean"]
    x0 = 1 / (1 + np.exp(-(cm - cm.mean()) / cm.std()))
    e = load_l1(CACHE)
    e = e[(e.sign != 0) & e.source.isin(gi) & e.target.isin(gi)]
    src, dst = e.source.map(gi).to_numpy(), e.target.map(gi).to_numpy()
    sign = e.sign.to_numpy(np.float32)
    conf = (np.minimum(e.n_db.to_numpy(), 3) / 3).astype(np.float32)
    rec = {}
    # (a) linear mechanism model W (rho(|W|) scaled to 0.9)
    Wraw, r_before, scale = build_W(src, dst, sign, conf, G, rho=10.0)       # no rescaling: report the raw radius
    Wsc, _, sc = build_W(src, dst, sign, conf, G, rho=0.9)
    rec["abs_rho_raw"] = float(r_before)
    rec["abs_rho_scaled"] = float(abs_spectral_radius(Wsc))
    rec["inf_norm_scaled"] = inf_norm(Wsc)
    rec["inf_norm_raw"] = inf_norm(Wraw)
    # Neumann convergence for the clamped response of reachable genes: terms until max |hop contribution| < 1e-6
    has_out = sorted(set(src.tolist()))[:40]
    hops_needed = []
    for g in has_out:
        nodes, steady, per_hop = clamped_response(Wsc, g, -1.0, hops=60)
        if len(nodes) == 0:
            continue
        mx = [float(np.abs(h).max()) for h in per_hop]
        hops_needed.append(next((i + 1 for i, v in enumerate(mx) if v < 1e-6), 60))
    rec["neumann_terms_1e6_median"] = float(np.median(hops_needed))
    rec["neumann_terms_1e6_max"] = float(np.max(hops_needed))
    # weighted sup-norm certificate: v = (I - |W|)^-1 1 >= 1 gives |W| v = v - 1 <= (1 - 1/max v) v, hence ||W_RR||_v <= q_v < 1
    A_abs = abs(Wsc).tocsc()
    v = splu((sp.identity(G, format="csc") - A_abs).tocsc()).solve(np.ones(G))
    q_v = 1.0 - 1.0 / float(v.max())
    rec["q_v"] = q_v
    rec["v_max"] = float(v.max())
    ratios = []
    for g in has_out:
        nodes, steady, per_hop = clamped_response(Wsc, g, -1.0, hops=4)
        if len(nodes) == 0:
            continue
        b = np.asarray(Wsc[nodes][:, [g]].todense()).ravel() * -1.0
        vv = v[nodes]
        nb = float(np.max(np.abs(b) / vv))
        for K in (1, 2, 3, 4):
            err = float(np.max(np.abs(steady - sum(per_hop[:K])) / vv))
            bound = q_v ** K / (1 - q_v) * nb
            ratios.append(err / max(bound, 1e-300))
    rec["truncation_bound_max_ratio"] = float(max(ratios))     # <= 1 means the bound holds on every checked perturbation
    # (b) certified nonlinear operators at extreme and default parameters
    cert = []
    for transfer in ("hill4", "linear"):
        for k_init, rho_p in ((0.06, 8.0), (0.3, 0.0), (0.99, 3.0)):
            m = GraphNet2(src, dst, sign, conf, x0, transfer=transfer, k_init=k_init, gamma_max=0.5, damping=0.5)
            with torch.no_grad():
                m.rho.fill_(rho_p)
            c = m.certificate()
            tgt = torch.as_tensor([int(has_out[i]) for i in (0, 5, 10)])
            sols, its = [], []
            for seed in range(4):
                z0 = torch.randn(3, G, generator=torch.Generator().manual_seed(seed)) * 2.0
                zz, _, _ = m.equilibrium(tgt, 0.8, z_init=z0, tol=1e-6)
                sols.append(zz)
                its.append(m.last_iters)
            spread = max(float((sols[0] - s_).abs().max()) for s_ in sols[1:])
            # fixed-point residual at the returned solution
            zz, z0b, M = m.equilibrium(tgt, 0.8, tol=1e-6)
            m.steps = 1
            z1, _, _ = m.equilibrium(tgt, 0.8, z_init=zz)
            resid = float((z1 - zz).abs().max())
            cert.append(dict(transfer=transfer, k=float(m.k()), q=c["q"], q_eta=c["q_eta"], L_T=c["L_T"], W_inf=c["W_inf"],
                             iters=int(np.max(its)), spread=spread, residual=resid))
    rec["certified"] = cert
    rec["q_max"] = max(c["q"] for c in cert)
    rec["q_eta_max"] = max(c["q_eta"] for c in cert)
    rec["iters_max"] = max(c["iters"] for c in cert)
    rec["spread_max"] = max(c["spread"] for c in cert)
    rec["residual_max"] = max(c["residual"] for c in cert)
    # (c) restart-walk kernels on the L1+L3 association graph
    import pickle, scipy.sparse as sp
    st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
    m_ = st.a.str.upper().isin(gi) & st.b.str.upper().isin(gi)
    a_, b_ = st.a[m_].str.upper().map(gi).to_numpy(), st.b[m_].str.upper().map(gi).to_numpy()
    A = sp.coo_matrix((np.ones(len(a_)), (a_, b_)), shape=(G, G)).tocsr()
    A = ((A + A.T) > 0).astype(float).tocsr()
    deg = np.asarray(A.sum(0)).ravel()
    seeds = np.where(deg > 0)[0][:50]
    cs = {}
    for al in (0.1, 0.5, 0.9):
        S = kernel_scores(A, seeds, "rwr", alpha=al, norm="rw")
        cs[str(al)] = dict(col_sum_dev=float(np.abs(S.sum(1) - 1).max()), rate=1 - al)
    rec["rwr"] = cs
    out[line] = rec
    print(line, {k: v for k, v in rec.items() if k not in ("certified", "rwr")}, flush=True)
json.dump(out, open(ART / "certificate.json", "w"), indent=1)
