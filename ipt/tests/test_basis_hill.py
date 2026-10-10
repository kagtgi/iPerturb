import numpy as np, torch
from ipt.models.hill import hill4
from ipt.models.basis_hill import FastGraphNet, fit_table, coef_from_k, cheb_clenshaw, cheb_stack, K_GRID


def test_cheb_approximation_error_decreases_with_degree():
    x = torch.linspace(0, 1, 2001)
    errs = {}
    for r in (4, 8, 16, 32):
        tab = torch.as_tensor(fit_table(r), dtype=torch.float32)
        e = 0
        for k in (0.05, 0.12, 0.3, 0.7, 1.0):
            coef = coef_from_k(torch.full((1,), k), tab)
            e = max(e, float((cheb_clenshaw(x[None, :], coef) - hill4(x, torch.tensor(k), 2.0)[None, :]).abs().max()))
        errs[r] = e
    assert errs[32] < errs[8] < errs[4] and errs[32] < 5e-3


def test_clenshaw_equals_stack():
    tab = torch.as_tensor(fit_table(12), dtype=torch.float32); k = torch.rand(30) * 0.9 + 0.06
    x = torch.rand(5, 30); coef = coef_from_k(k, tab)
    assert torch.allclose(cheb_clenshaw(x, coef), cheb_stack(x, coef), atol=1e-4)


def rand_graph(G=60, E=200, seed=0):
    rng = np.random.default_rng(seed); pairs = set()
    while len(pairs) < E:
        a, b = rng.integers(0, G, 2)
        if a != b: pairs.add((int(a), int(b)))
    s, d = map(np.array, zip(*sorted(pairs)))
    return s, d, rng.choice([-1.0, 1.0], E), rng.uniform(0.3, 1, E), rng.uniform(0.2, 0.8, G)


def test_implementations_agree():
    s, d, sg, cf, x0 = rand_graph(); tgt = torch.tensor([3, 10, 40])
    ref = FastGraphNet(s, d, sg, cf, x0, transfer="hill4", steps=30)
    xr, _, _ = ref.equilibrium(tgt, 0.8)
    for hi, pi, tol in (("exact", "spmm", 1e-5), ("exact", "dense", 1e-5), ("cheb", "edge", 2e-3), ("cheb", "spmm", 2e-3), ("cheb", "dense", 2e-3)):
        m = FastGraphNet(s, d, sg, cf, x0, transfer="hill4", steps=30, hill_impl=hi, prop_impl=pi, r=24)
        m.load_state_dict({k: v for k, v in ref.state_dict().items() if k in m.state_dict() and m.state_dict()[k].shape == v.shape}, strict=False)
        x, _, _ = m.equilibrium(tgt, 0.8)
        assert float((x - xr).abs().max()) < tol, (hi, pi, float((x - xr).abs().max()))


def test_gradients_through_cheb_spmm():
    s, d, sg, cf, x0 = rand_graph()
    m = FastGraphNet(s, d, sg, cf, x0, transfer="hill4", steps=8, hill_impl="cheb", prop_impl="spmm", r=16, k_mode="regulator")
    out, _ = m(torch.tensor([1, 2]), 0.8); out.sum().backward()
    assert torch.isfinite(m.kz.grad).all() and m.a.grad is not None and torch.isfinite(m.a.grad)
