"""SCC-aware exact solver for the baseline-centered fixed point (single perturbation, numpy float64).

Same equations as GraphNet.equilibrium; only the execution order differs:
condense strongly connected components, topologically order them, restrict to nodes reachable from the
target, settle singleton acyclic nodes with one forward update and feedback components with a local damped
fixed-point loop whose external inputs are already settled. Predictions must match the global solver.
"""
from __future__ import annotations
import numpy as np
import networkx as nx


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(x):
    return np.log(x) - np.log1p(-x)


class Problem:
    def __init__(self, src, dst, w, x0, T):
        self.src, self.dst, self.w, self.x0, self.T = np.asarray(src), np.asarray(dst), np.asarray(w, float), np.asarray(x0, float), T
        self.G = len(x0)
        self.h0 = T(self.x0)
        self.base = _logit(self.x0)
        self.pred = [[] for _ in range(self.G)]  # incoming (src, w) lists
        for s, d, ww in zip(self.src, self.dst, self.w):
            self.pred[d].append((s, ww))
        self.pred = [(np.array([a for a, _ in p], int), np.array([b for _, b in p], float)) for p in self.pred]
        g = nx.DiGraph()
        g.add_nodes_from(range(self.G))
        g.add_edges_from(zip(self.src.tolist(), self.dst.tolist()))
        self.graph = g
        self.cond = nx.condensation(g)
        self.order = list(nx.topological_sort(self.cond))
        self.node_comp = self.cond.graph["mapping"]
        sizes = [len(self.cond.nodes[c]["members"]) for c in self.cond.nodes]
        self.largest_scc = max(sizes)

    def update(self, x, i, hx):
        s, w = self.pred[i]
        z = self.base[i] + (w * (hx[s] - self.h0[s])).sum() if len(s) else self.base[i]
        return _sigmoid(z)


def solve_global(P, target, clamp, damping=0.5, tol=1e-13, max_iter=200000):
    x = P.x0.copy()
    x[target] = clamp
    for it in range(max_iter):
        hx = P.T(x)
        new = np.array([P.update(x, i, hx) for i in range(P.G)])
        new[target] = clamp
        new = (1 - damping) * x + damping * new
        new[target] = clamp
        d = np.abs(new - x).max()
        x = new
        if d < tol:
            return x, it + 1
    return x, max_iter


def solve_scc(P, target, clamp, damping=0.5, tol=1e-13, max_iter=200000):
    reach = nx.descendants(P.graph, target) | {target}
    comps = [c for c in P.order if any(m in reach for m in P.cond.nodes[c]["members"])]
    x = P.x0.copy()
    x[target] = clamp
    n_updates = 0
    for c in comps:
        mem = [m for m in P.cond.nodes[c]["members"] if m in reach and m != target]
        if not mem:
            continue
        cyc = len(P.cond.nodes[c]["members"]) > 1 or any(P.graph.has_edge(m, m) for m in mem)
        if not cyc:
            hx = P.T(x)
            for m in mem:
                x[m] = P.update(x, m, hx)
                n_updates += 1
            continue
        for it in range(max_iter):
            hx = P.T(x)
            new = x.copy()
            for m in mem:
                new[m] = (1 - damping) * x[m] + damping * P.update(x, m, hx)
            n_updates += len(mem)
            d = np.abs(new - x).max()
            x = new
            if d < tol:
                break
    return x, n_updates


def solve_global_reach(P, target, clamp, damping=0.5, tol=1e-13, max_iter=200000):
    """Global synchronous iteration restricted to the genes reachable from the target (no SCC ordering)."""
    nodes = [i for i in sorted(nx.descendants(P.graph, target)) if i != target]
    x = P.x0.copy()
    x[target] = clamp
    for it in range(max_iter):
        hx = P.T(x)
        new = x.copy()
        for m in nodes:
            new[m] = (1 - damping) * x[m] + damping * P.update(x, m, hx)
        new[target] = clamp
        d = np.abs(new - x).max()
        x = new
        if d < tol:
            return x, it + 1
    return x, max_iter
