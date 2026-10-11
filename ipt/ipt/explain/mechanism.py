"""Mechanism-faithful explanation of CRISPRi propagation over a signed, directed GRN.

Model (steady state, perturbed gene g clamped, R = all other genes):
    dx_R = (I - W_RR)^-1  W_Rg  dx_g ,      W_ij = signed effect of gene j on gene i (activation > 0, repression < 0)
with  dx_g = log-expression change of the clamped gene implied by the (partial) knockdown efficiency.
Valid when rho(W_RR) < 1 (checked and enforced by a global gain).  Hop decomposition: dx^(k) = W_RR^(k-1) W_Rg dx_g.
Paths: strongest signed walks g -> ... -> t of length <= L with their edge evidence; competing (opposite-sign) and reinforcing
contributions are reported separately so cancellation is visible.  Paths are candidate mechanisms, not proven causal routes.
"""
from __future__ import annotations
from collections import deque
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, eigs


def abs_spectral_radius(W, iters=200, tol=1e-9):
    """rho(|W|) by power iteration on the nonnegative matrix |W| (Perron-Frobenius: rho(|W_RR|) <= rho(|W|) for every principal block).
    The SIGNED spectral radius is NOT used: rho(W) < 1 does not imply rho(W_RR) < 1 (counterexample in the appendix)."""
    A = abs(W).tocsr()
    v = np.ones(A.shape[0]) / np.sqrt(A.shape[0])
    r = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        if abs(nw - r) < tol * max(nw, 1.0):
            r = nw
            break
        r = nw
    return float(r)


def inf_norm(W):
    """||W||_inf = maximum absolute row sum (total incoming weight of the most regulated gene)."""
    return float(abs(W).sum(1).max())


def build_W(src, dst, sign, conf, G, rho=0.9):
    """W[dst, src] = sign*conf (row i = target, column j = regulator), rescaled so that rho(|W|) = `rho` (< 1).
    Then every principal block satisfies rho(|W_RR|) <= rho, so the Neumann series converges absolutely (Theorem 2).
    Returns (W, rho_abs_before, scale)."""
    W = sp.coo_matrix((np.asarray(sign, float) * np.asarray(conf, float), (dst, src)), shape=(G, G)).tocsr()
    r = abs_spectral_radius(W)
    scale = rho / r if r > rho else 1.0
    return (W * scale).tocsr(), float(r), float(scale)


def reachable(W, g, max_depth=None):
    """BFS distances from g following regulator -> target edges (column j -> row i)."""
    Wc = W.tocsc()
    dist = {g: 0}
    q = deque([g])
    while q:
        u = q.popleft()
        if max_depth is not None and dist[u] >= max_depth:
            continue
        for v in Wc.indices[Wc.indptr[u]:Wc.indptr[u + 1]]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def clamped_response(W, g, dxg, dist=None, hops=4):
    """Return (nodes, dx_steady, [dx_hop1..dx_hopK]) restricted to the genes reachable from g."""
    dist = dist or reachable(W, g)
    nodes = np.array([n for n in dist if n != g])
    if len(nodes) == 0:
        return nodes, np.zeros(0), [np.zeros(0)] * hops
    Wrr = W[nodes][:, nodes].tocsc()
    b = np.asarray(W[nodes][:, [g]].todense()).ravel() * dxg
    M = (sp.identity(len(nodes), format="csc") - Wrr).tocsc()
    steady = splu(M).solve(b)
    per_hop, cur = [], b.copy()
    for _ in range(hops):
        per_hop.append(cur.copy())
        cur = Wrr @ cur
    return nodes, steady, per_hop


def top_paths(W, g, t, max_len=3, top=5, edge_info=None):
    """Strongest signed walks g -> t (length <= max_len) by |product of edge effects|; returns list of dicts."""
    Wc = W.tocsc()
    paths = []

    def rec(u, path, prod):
        if len(path) - 1 >= max_len:
            return
        lo, hi = Wc.indptr[u], Wc.indptr[u + 1]
        for v, w in zip(Wc.indices[lo:hi], Wc.data[lo:hi]):
            if v in path:
                continue
            p2 = prod * w
            if abs(p2) < 1e-6:
                continue
            if v == t:
                paths.append((p2, path + [v]))
            else:
                rec(v, path + [v], p2)

    rec(g, [g], 1.0)
    paths.sort(key=lambda x: -abs(x[0]))
    out = []
    for prod, p in paths[:top]:
        edges = []
        for a, b in zip(p[:-1], p[1:]):
            info = (edge_info or {}).get((a, b), {})
            edges.append(dict(src=a, dst=b, effect=float(W[b, a]), **info))
        out.append(dict(path=p, effect=float(prod), edges=edges))
    pos = float(sum(x[0] for x in paths if x[0] > 0))
    neg = float(sum(x[0] for x in paths if x[0] < 0))
    return out, dict(reinforcing=pos, opposing=neg, n_paths=len(paths), net=pos + neg)
