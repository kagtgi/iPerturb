"""Phase F (optional) -- loss ablation + lambda-5 sensitivity sweep for iPerturb.

Run this INSIDE the Colab/notebook session where iperturb.py has already been
executed, so that the model factory, datasets, training and evaluation helpers
are in scope. It answers reviewers R1.1 / R2.B / R2.C (per-term contribution and
robustness of the lambda weights) on K562 with directional accuracy + MSE.

It does NOT retrain from this repo (no data/GPU here); it is the harness to drop
into your Colab. Wire the four hooks marked TODO to your existing objects, then
run. It prints a console table and paste-ready LaTeX rows for Table~\\ref{tab:ablation}.
"""
import numpy as np
import torch

# ---- hooks into your existing iperturb.py session -------------------------
# TODO 1: a fresh, untrained model for the K562 template GRN (called per seed).
def make_model():
    raise NotImplementedError("return a new GRNN(...) on the K562 template GRN")

# TODO 2: your train/val/test datasets for K562 (already built in the notebook).
TRAIN_DS, VAL_DS, TEST_DS = None, None, None          # e.g. K562_train, K562_val, K562_test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# train_grnn(...) and evaluate_all(...) are defined in iperturb.py; we call them.
# evaluate_all is expected to return a dict with 'directional_accuracy' and 'mse_delta'.
# -------------------------------------------------------------------------

BASE = dict(lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0, lam_balance=0.5, lam_deg=2.0)
DROP = {                                  # drop-one-term configs (set one weight to 0)
    "full":            dict(),
    "-WMSE":           dict(lam_wmse=0.0),
    "-AFDA":           dict(lam_afda=0.0),
    "-bal":            dict(lam_balance=0.0),
    "-Delta":          dict(lam_delta=0.0),
    "-DEG":            dict(lam_deg=0.0),
}
SEEDS = [0, 1, 2, 3, 4]

def run(weights):
    dirs, mses = [], []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        model = make_model().to(DEVICE)
        train_grnn(model, TRAIN_DS, VAL_DS, device=DEVICE, **weights)   # noqa: F821
        m = evaluate_all(model, TEST_DS, device=DEVICE)                 # noqa: F821
        dirs.append(m["directional_accuracy"]); mses.append(m["mse_delta"])
    return np.mean(dirs), np.std(dirs), np.mean(mses), np.std(mses)

print("\n=== Drop-one-term ablation (K562, 5 seeds) ===")
print(f"{'config':10} {'Dir.acc':>14} {'MSE':>14}")
rows = []
for name, override in DROP.items():
    w = {**BASE, **override}
    da, ds, ms, mss = run(w)
    print(f"{name:10} {da:.3f}+/-{ds:.3f}   {ms:.3f}+/-{mss:.3f}")
    rows.append((name, da, ds, ms, mss))

print("\n% paste-ready LaTeX rows (Dir up, MSE down):")
for name, da, ds, ms, mss in rows:
    tag = name.replace("-", "$-$")
    print(f"{tag:16} & ${da:.2f}{{\\pm}}{ds:.2f}$ & ${ms:.2f}{{\\pm}}{mss:.2f}$ \\\\")

print("\n=== lambda_5 (DEG) sweep ===")
for l5 in (0.5, 1.0, 2.0, 4.0):
    da, ds, ms, mss = run({**BASE, "lam_deg": l5})
    print(f"lambda5={l5:>3}  Dir={da:.3f}+/-{ds:.3f}  MSE={ms:.3f}+/-{mss:.3f}")
print("\nIf Dir/MSE are stable across the sweep, cite that as the robustness result in 2.3.")
