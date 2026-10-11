"""Numerical certificates for the convergence statements of the appendix (Lemma 1, Theorems 1-2, Propositions 3-4)."""
import numpy as np
import scipy.sparse as sp
import torch

from ipt.explain.mechanism import abs_spectral_radius, build_W, clamped_response, inf_norm
from ipt.models.diffusion import kernel_scores, normalise
from ipt.models.graphnet2 import GraphNet2, hill_lipschitz
from ipt.models.hill import hill4, hill4_lipschitz


def rand_graph(G=60, E=240, seed=0):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < E:
        a, b = rng.integers(0, G, 2)
        if a != b:
            pairs.add((int(a), int(b)))
    s, d = map(np.array, zip(*sorted(pairs)))
    return s, d, rng.choice([-1.0, 1.0], E), rng.uniform(0.3, 1.0, E), rng.uniform(0.2, 0.8, G)


def test_lemma1_hill_slope_closed_form():
    x = torch.linspace(0, 1 - 1e-6, 200001, dtype=torch.float64, requires_grad=True)
    for k in (0.05, 0.1, 0.3, 0.7, 1.0):
        y = hill4(x, torch.tensor(k, dtype=torch.float64), 2.0)
        g, = torch.autograd.grad(y.sum(), x)
        assert abs(float(g.abs().max()) - hill_lipschitz(k)) / hill_lipschitz(k) < 1e-3
        assert abs(hill4_lipschitz(k) - hill_lipschitz(k)) / hill_lipschitz(k) < 5e-3


def test_theorem1_contraction_unique_fixed_point_and_posteriori_bound():
    s, d, sg, cf, x0 = rand_graph()
    for k_init, a, rho in ((0.06, 5.0, 8.0), (0.3, 0.0, 0.0), (0.99, -3.0, 3.0)):        # extreme parameter values
        for transfer in ("hill4", "linear"):
            m = GraphNet2(s, d, sg, cf, x0, transfer=transfer, k_init=k_init, gamma_max=0.5, damping=0.5)
            with torch.no_grad():
                m.a.fill_(a); m.rho.fill_(rho)
            cert = m.certificate()
            assert cert["q"] <= 0.5 + 1e-6 and cert["q_eta"] < 1.0
            tgt = torch.tensor([3, 11])
            sols = []
            for seed in range(4):
                z0 = torch.randn(2, m.G, generator=torch.Generator().manual_seed(seed)) * 2.0
                z, _, M = m.equilibrium(tgt, 0.8, z_init=z0, tol=1e-6)
                sols.append(z)
            assert max(float((sols[0] - s_).abs().max()) for s_ in sols[1:]) < 5e-6          # unique fixed point from any start (float32)
            # a-posteriori bound  ||z_k - z*|| <= q_eta/(1-q_eta) ||z_k - z_{k-1}||
            zstar = sols[0]
            m.steps = 7
            zk, _, _ = m.equilibrium(tgt, 0.8, z_init=torch.zeros(2, m.G))
            m.steps = 6
            zk1, _, _ = m.equilibrium(tgt, 0.8, z_init=torch.zeros(2, m.G))
            qe = cert["q_eta"]
            assert float((zk - zstar).abs().max()) <= qe / (1 - qe) * float((zk - zk1).abs().max()) + 5e-6


def test_theorem2_neumann_truncation_bound_and_abs_scaling():
    s, d, sg, cf, _ = rand_graph(80, 320, seed=2)
    W, r_before, scale = build_W(s, d, sg, cf, 80, rho=0.9)
    assert abs(abs_spectral_radius(W) - 0.9) < 1e-3
    rng = np.random.default_rng(0)
    for _ in range(5):                                           # every principal block keeps rho(|W_RR|) <= rho(|W|)
        R = np.sort(rng.choice(80, 50, replace=False))
        assert abs_spectral_radius(W[R][:, R]) <= abs_spectral_radius(W) + 1e-6
    # truncation bound with the sup norm (rescale to ||W||_inf < 1 for the bound)
    Wn = (W / (1.2 * inf_norm(W))).tocsr()
    b = rng.normal(size=80)
    exact = np.linalg.solve(np.eye(80) - Wn.toarray(), b)
    acc, cur = np.zeros(80), b.copy()
    nrm = inf_norm(Wn)
    for K in range(1, 12):
        acc += cur
        cur = Wn @ cur
        assert np.abs(exact - acc).max() <= nrm ** K / (1 - nrm) * np.abs(b).max() + 1e-12


def test_signed_spectral_radius_counterexample():
    W = sp.csr_matrix(np.array([[1.0, 1.0], [-1.0, -1.0]]))
    assert np.max(np.abs(np.linalg.eigvals(W.toarray()))) < 1e-8          # signed rho(W) = 0 ...
    assert abs(abs_spectral_radius(W[[0]][:, [0]]) - 1.0) < 1e-9           # ... but the block W_RR = [1] has rho = 1 (I - W_RR singular)
    W2, _, _ = build_W([0, 1, 0, 1], [0, 0, 1, 1], [1, -1, 1, -1], [1, 1, 1, 1], 2, rho=0.9)
    assert abs_spectral_radius(W2[[0]][:, [0]]) < 1.0                      # the |W| scaling repairs it


def test_proposition4_kernels():
    s, d, _, _, _ = rand_graph(70, 300, seed=3)
    A = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(70, 70)).tocsr()
    A = ((A + A.T) > 0).astype(float).tocsr()
    seeds = np.arange(0, 70, 7)
    for al in (0.1, 0.5, 0.9):
        S = kernel_scores(A, seeds, "rwr", alpha=al, norm="rw")           # rows = seeds
        deg = np.asarray(A.sum(0)).ravel()
        keep = deg[seeds] > 0
        assert np.allclose(S[keep].sum(1), 1.0, atol=1e-9) and (S >= -1e-12).all()   # probabilities that sum to one
        Ws = normalise(A, "sym").toarray()
        assert np.abs(np.linalg.eigvalsh(Ws)).max() <= 1 + 1e-9
    Wrw = normalise(A, "rw").toarray()
    assert np.max(np.abs(np.linalg.eigvals((1 - 0.3) * Wrw))) <= 0.7 + 1e-9    # spectral radius of the restart walk is 1 - alpha
