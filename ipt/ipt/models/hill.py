"""TF4 modified Hill transfer (Eduati 2017 / CNORode2017): H(x;k,n)=k^n[1-(1-x)^n]/((1-x)^n+k^n).
H(0)=0, H(1)=1 for k>0; monotone increasing in x on [0,1]."""
import torch


def hill4(x, k, n=2.0):
    kn = k ** n
    om = (1.0 - x).clamp_min(0.0) ** n
    return kn * (1.0 - om) / (om + kn)


def hill4_lipschitz(k, n=2.0):
    """Upper bound on sup_x |H'| over [0,1] by dense grid (used for the contraction check)."""
    x = torch.linspace(0, 1 - 1e-4, 4001, dtype=torch.float64, requires_grad=True)
    kk = torch.as_tensor(k, dtype=torch.float64)
    if kk.ndim == 0:
        y = hill4(x, kk, n); g, = torch.autograd.grad(y.sum(), x); return float(g.abs().max())
    return max(hill4_lipschitz(float(v), n) for v in kk.flatten())
