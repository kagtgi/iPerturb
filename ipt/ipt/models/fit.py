"""Fit graph residual models (B3 linear / H2 Hill4) on training reachable targets with val early stopping."""
from __future__ import annotations
import copy
import numpy as np
import torch
from ipt.models.graphnet import GraphNet


def rewire(src, dst, sign, conf, seed, n_swap_mult=10):
    """Degree-preserving target swap (keeps out/in-degree and each edge's sign/conf)."""
    rng = np.random.default_rng(seed)
    dst = dst.copy()
    E = len(dst)
    pairs = set(zip(src.tolist(), dst.tolist()))
    for _ in range(n_swap_mult * E):
        i, j = rng.integers(0, E, 2)
        if i == j:
            continue
        a, b = (src[i], dst[j]), (src[j], dst[i])
        if a[0] == a[1] or b[0] == b[1] or a in pairs or b in pairs:
            continue
        pairs.discard((src[i], dst[i]))
        pairs.discard((src[j], dst[j]))
        pairs.add(a)
        pairs.add(b)
        dst[i], dst[j] = dst[j], dst[i]
    return src, dst, sign, conf


def masked_mse(pred, y, tgt):
    m = torch.ones_like(y, dtype=torch.bool)
    m[torch.arange(len(tgt)), tgt] = False
    return (((pred - y) ** 2) * m).sum() / m.sum()


def fit_graph(graph, x0, transfer, k_mode, R_tr, t_tr, R_va, t_va, zeta, lr, lam_u, seed,
              epochs=120, batch=32, freeze_k=False, patience=20):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = GraphNet(*graph, x0, transfer=transfer, k_mode=k_mode)
    if freeze_k or transfer == "linear":
        net.kz.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    Rt, tt = torch.as_tensor(R_tr, dtype=torch.float32), torch.as_tensor(t_tr)
    Rv, tv = torch.as_tensor(R_va, dtype=torch.float32), torch.as_tensor(t_va)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        perm = rng.permutation(len(tt))
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            pred, _ = net(tt[b], zeta)
            loss = masked_mse(pred, Rt[b], tt[b]) + lam_u * (net.u ** 2).mean()
            if k_mode == "regulator":
                loss = loss + lam_u * ((net.k() - net.k_bar) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            v = float(masked_mse(net(tv, zeta)[0], Rv, tv)) if len(tv) else float(loss)
        hist.append(v)
        if v < best - 1e-9:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, dict(val_mse=best, best_epoch=int(np.argmin(hist)), epochs=len(hist))
