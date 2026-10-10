import numpy as np, torch
from ipt.models.peredge import PerEdgeCheb


def test_edge_equals_spmm_and_grads():
    rng = np.random.default_rng(0); G, E = 50, 160; pairs = set()
    while len(pairs) < E:
        a, b = rng.integers(0, G, 2)
        if a != b: pairs.add((int(a), int(b)))
    s, d = map(np.array, zip(*sorted(pairs))); x0 = rng.uniform(0.2, 0.8, G)
    a = PerEdgeCheb(s, d, x0, r=8, impl="edge", steps=10); b = PerEdgeCheb(s, d, x0, r=8, impl="spmm", steps=10)
    b.load_state_dict(a.state_dict())
    tgt = torch.tensor([1, 7, 30])
    xa, xb = a(tgt), b(tgt)
    assert float((xa - xb).abs().max()) < 1e-5
    xb.pow(2).mean().backward(); xa.pow(2).mean().backward()
    assert torch.allclose(a.coef.grad, b.coef.grad, atol=1e-6)
