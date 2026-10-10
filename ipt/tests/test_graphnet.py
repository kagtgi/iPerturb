import numpy as np, torch
from ipt.models.hill import hill4
from ipt.models.graphnet import GraphNet


def test_hill_endpoints_monotone_grad():
    x = torch.linspace(0, 1, 101, dtype=torch.float64)
    for k in (0.05, 0.3, 1.0):
        h = hill4(x, torch.tensor(k, dtype=torch.float64), 2.0)
        assert abs(h[0]) < 1e-12 and abs(h[-1] - 1) < 1e-12 and (h[1:] >= h[:-1]).all()
    xx = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    kk = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: hill4(a, b, 2.0), (xx, kk))


def chain(transfer="hill4", sign=(1, -1)):
    # 0 -> 1 -> 2, isolated 3
    return GraphNet([0, 1], [1, 2], list(sign), [1, 1], np.full(4, 0.5), transfer=transfer, steps=40)


def test_zero_when_no_perturbation_effect_and_clamp_persists():
    m = chain(); tgt = torch.tensor([3]); out, r = m(tgt, 0.8)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)          # isolated gene -> no downstream change
    x, _, _ = m.equilibrium(torch.tensor([0]), 0.8)
    assert abs(x[0, 0].item() - 0.2 * 0.5 - 0.0) < 1e-6                   # target clamped at (1-zeta)*x0


def test_sign_propagation_and_convergence():
    for tr in ("hill4", "linear"):
        m = chain(tr); out, r = m(torch.tensor([0]), 0.8)
        assert out[0, 1] < 0 and out[0, 2] > 0 and r.item() < 1e-4       # KD of activator lowers gene1; inhibitor chain flips gene2
        assert out[0, 0].item() == 0.0 and abs(out[0, 3].item()) < 1e-7


def test_gradients_flow():
    m = chain(); out, _ = m(torch.tensor([0]), 0.8); out.sum().backward()
    assert m.a.grad is not None and torch.isfinite(m.a.grad) and m.kz.grad is not None
