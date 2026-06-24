"""Phase F (optional) -- contraction check + convergence hop count for iPerturb.

Run in the Colab/notebook session after a model is trained. It verifies the
claim that the message-passing map F is contractive (R2.A): if the maximum
absolute Jacobian row-sum  max_i sum_k |dF_i/dx_k| < 1  on the fitted model,
the iteration converges geometrically (Banach). It also reports the median
number of hops to reach the tolerance, to fill [k] in the Section 2.3 sentence.

If the row-sum is < 1, upgrade the Section 2.3 close to the contraction wording
left as a TODO comment in main.tex. If it is >= 1, keep the pure-empirical
phrasing ('reached the tolerance within k hops in all experiments').
"""
import torch

# ---- hooks -----------------------------------------------------------------
# TODO 1: a trained model (GRNN) and a baseline expression vector x0 (1D tensor,
#         length = number of genes), on the same device.
model = None          # e.g. the K562 model after train_grnn(...)
x0    = None          # e.g. model baseline state used as the iteration start
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The model is assumed to expose one synchronous update step that maps the full
# expression vector to the next one (Eqs 1-2). In iperturb.py this is the inner
# update of GRNN; expose it as model.step(x) -> x_next, or adapt STEP below.
def STEP(x):
    # TODO 2: return one message-passing update F(x) for a non-perturbed forward
    # pass (no clamped gene), e.g.  return model._step(x)
    raise NotImplementedError("return one synchronous Hill-kinetics update F(x)")
# ---------------------------------------------------------------------------

@torch.no_grad()
def converge(x, eps=1e-5, max_iter=100):
    for t in range(max_iter):
        xn = STEP(x)
        if torch.linalg.norm(xn - x) < eps:
            return t + 1, xn
        x = xn
    return max_iter, x

def max_abs_rowsum(x_star):
    x = x_star.clone().detach().to(DEVICE).requires_grad_(False)
    J = torch.autograd.functional.jacobian(STEP, x)      # [n, n]
    rowsum = J.abs().sum(dim=1)
    return rowsum.max().item(), rowsum.mean().item()

if __name__ == "__main__":
    model.eval()
    hops, x_star = converge(x0.clone().to(DEVICE))
    mx, mean = max_abs_rowsum(x_star)
    print(f"hops to tolerance (this run): {hops}")
    print(f"max_i sum_k |dF_i/dx_k| = {mx:.4f}   (mean row-sum = {mean:.4f})")
    print("CONTRACTIVE (row-sum < 1): upgrade Section 2.3 to the Banach wording."
          if mx < 1.0 else
          "NOT globally contractive: keep the empirical convergence phrasing.")
    # For the paper, run converge() over all held-out perturbations and report the
    # median 'hops' as k, and the max 'mx' across trained models for the claim.
