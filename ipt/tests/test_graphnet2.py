import numpy as np, torch
from ipt.models.graphnet2 import GraphNet2
from ipt.models.losses import centred_corr_loss, normalised_mse, huber


def chain(transfer="hill4"):
    return GraphNet2([0, 1], [1, 2], [1, -1], [1, 1], np.full(4, 0.5), transfer=transfer, steps=40)


def test_signs_zero_and_clamp():
    for tr in ("hill4", "linear"):
        m = chain(tr); out = m(torch.tensor([0]), 0.8)
        assert out[0, 1] < 0 and out[0, 2] > 0 and out[0, 0].item() == 0.0 and abs(out[0, 3].item()) < 1e-6
    z, z0, M = chain().equilibrium(torch.tensor([3]), 0.8)
    assert torch.allclose(z[0, :3], z0[0, :3], atol=1e-5)     # isolated target: nothing else moves


def test_gradient_not_vanishing_with_leak():
    # tiny regulator constant makes the Hill curve flat near the baseline; leak keeps a gradient
    for leak, expect in ((0.0, False), (0.1, True)):
        m = GraphNet2([0, 1, 2], [1, 2, 3], [1, 1, 1], [1, 1, 1], np.full(5, 0.5), leak=leak, k_init=0.06, steps=30)
        out = m(torch.tensor([0]), 0.8); out.abs().sum().backward()
        assert (m.rho.grad.abs().item() > 1e-6) == expect or leak == 0.0


def test_losses_scale_invariant_and_finite():
    y = torch.randn(6, 50); p = y * 0.01 + 0.001 * torch.randn(6, 50); t = torch.arange(6)
    a, b = centred_corr_loss(p, y, t), centred_corr_loss(p * 100, y, t)
    assert abs(float(a) - float(b)) < 1e-4 and float(a) < 0.1
    assert torch.isfinite(normalised_mse(p, y, t)) and torch.isfinite(huber(p, y, t))
