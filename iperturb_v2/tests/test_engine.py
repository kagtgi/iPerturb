"""CPU unit tests for the iPerturb Tier-1 network-propagation engine (no GPU/torch)."""
import numpy as np

from iperturb_v2 import _iperturb_prop as prop


def _toy_graph(n=10, seed=0):
    rng = np.random.default_rng(seed)
    edges, signs, levels = [], [], []
    for _ in range(n * 3):
        s, t = int(rng.integers(0, n)), int(rng.integers(0, n))
        if s != t:
            edges.append((s, t)); signs.append(int(rng.choice([-1, 0, 1])))
            levels.append(int(rng.integers(1, 5)))
    return edges, signs, levels


def test_operator_is_contraction():
    """|What| must have spectral radius <= 1 so RWR's inverse exists / converges."""
    edges, signs, levels = _toy_graph()
    for norm in ("sym", "rw"):
        What = prop.build_operator(edges, signs, levels, 10, norm=norm, signed=True)
        rho = float(np.max(np.abs(np.linalg.eigvals(abs(What).toarray()))))
        assert rho <= 1.0 + 1e-6, (norm, rho)


def test_rwr_matches_power_series():
    """Closed-form RWR == truncated geometric power series alpha*sum (1-alpha)^k What^k s."""
    edges, signs, levels = _toy_graph(n=12, seed=1)
    What = prop.build_operator(edges, signs, levels, 12, norm="sym", signed=False)
    s = np.zeros(12); s[3] = 1.0
    alpha = 0.3
    p = prop.propagate(What, s, alpha, "rwr")[:, 0]
    Wd = What.toarray(); acc = np.zeros(12); term = s.copy()
    for k in range(300):
        acc += (1 - alpha) ** k * term
        term = Wd @ term
    assert np.allclose(p, alpha * acc, atol=1e-4)


def test_propagation_linear_so_combos_are_additive():
    """propagate(sA+sB) == propagate(sA)+propagate(sB): combos compose by construction."""
    edges, signs, levels = _toy_graph(n=14, seed=2)
    What = prop.build_operator(edges, signs, levels, 14, signed=False)
    sA = np.zeros(14); sA[1] = 1.0
    sB = np.zeros(14); sB[6] = 1.0
    pA = prop.propagate(What, sA, 0.2, "rwr")[:, 0]
    pB = prop.propagate(What, sB, 0.2, "rwr")[:, 0]
    pAB = prop.propagate(What, sA + sB, 0.2, "rwr")[:, 0]
    assert np.allclose(pAB, pA + pB, atol=1e-6)


def test_diag_readout_recovers_planted_response():
    """Closed-form diagonal ridge recovers a planted per-gene response r_j."""
    rng = np.random.default_rng(3)
    N, n = 18, 50
    P = rng.normal(size=(N, n))
    r_true = rng.normal(size=N)
    D = r_true[:, None] * P + 0.01 * rng.normal(size=(N, n))
    ro = prop.fit_readout(P, D, lam=1e-3, rank=None)
    assert ro["kind"] == "diag"
    assert np.corrcoef(ro["r"], r_true)[0, 1] > 0.99


def test_lowrank_readout_predicts():
    """Low-rank readout reconstructs train deltas it was fit on."""
    rng = np.random.default_rng(4)
    N, n = 20, 30
    P = rng.normal(size=(N, n))
    M = rng.normal(size=(N, N))
    D = M @ P
    ro = prop.fit_readout(P, D, lam=1e-6, rank=N)
    Dhat = prop.apply_readout(ro, P)
    # mean per-column correlation high
    cc = np.mean([np.corrcoef(Dhat[:, j], D[:, j])[0, 1] for j in range(n)])
    assert cc > 0.95
