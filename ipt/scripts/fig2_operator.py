"""Fig. 2: the iPT operator -- transfer functions, regulator-shared Hill, persistent clamp, SCC execution.
All curves are computed from ipt code; panel (d) reads artifacts/scc_bench.json.
Run from the repo root:  PYTHONPATH=. python scripts/fig2_operator.py"""
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from ipt.viz.style import *          # noqa: F401,F403  (palette + apply_style)
from ipt.models.hill import hill4
from ipt.models.basis_hill import fit_table, coef_from_k, cheb_clenshaw

apply_style()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
D = torch.float64
BLUE_RAMP = ["#C6DAFC", GBLUE_L, GBLUE, "#2A62C9", "#16397A"]   # light -> dark


def H(x, k, n=2.0):
    return hill4(torch.as_tensor(x, dtype=D), torch.tensor(float(k), dtype=D), n).numpy()


def letter(ax, s, dx=-0.20, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom", color=GREY900)


fig = plt.figure(figsize=(122 * MM, 132 * MM))
gs = fig.add_gridspec(3, 1, left=0.095, right=0.985, top=0.965, bottom=0.125,
                      hspace=0.62, height_ratios=[1, 0.95, 1])
gs0 = gs[0].subgridspec(1, 3, width_ratios=[1.2, 0.8, 0.9], wspace=0.95)
gs2 = gs[2].subgridspec(1, 2, wspace=0.95)

# ------------------------------------------------------------------ (a)
axa = fig.add_subplot(gs0[0, 0])
x = np.linspace(0, 1, 401)
axa.plot(x, x, color=GGREY, ls="--", lw=1.0, label="linear")
KS = [0.05, 0.15, 0.3, 0.6, 1.0]
for k, c in zip(KS, BLUE_RAMP):
    axa.plot(x, H(x, k), color=c, lw=1.3, label=f"$k$={k:g}")
tab = fit_table(8)
coef = coef_from_k(torch.tensor([0.15], dtype=torch.float32), torch.as_tensor(tab, dtype=torch.float32))
cheb = cheb_clenshaw(torch.as_tensor(x, dtype=torch.float32)[:, None].expand(-1, 1), coef)[:, 0].numpy()
axa.plot(x, cheb, color=GRED, ls=(0, (2.5, 2)), lw=1.1, label="Cheb. $r$=8, $k$=0.15")
axa.set_xlim(0, 1); axa.set_ylim(0, 1.02)
axa.set_xlabel("input $x$"); axa.set_ylabel("$H(x;k)$")
hl = axa.get_legend_handles_labels()
axl = fig.add_subplot(gs0[0, 1]); axl.axis("off"); axl.grid(False)
axl.legend(*hl, loc="center left", bbox_to_anchor=(-0.2, 0.5), fontsize=7, handlelength=1.6,
           labelspacing=0.35, borderaxespad=0.0)
letter(axa, "(a)", dx=-0.27)

# inset: max error vs degree over a k grid
RS = [4, 8, 16, 32]
kgrid = np.exp(np.linspace(np.log(0.05), np.log(1.0), 25))   # off the fit nodes on purpose
xs = np.linspace(0, 1, 2001)
errs = []
for r in RS:
    t = torch.as_tensor(fit_table(r), dtype=torch.float32)
    cf = coef_from_k(torch.tensor(kgrid, dtype=torch.float32), t)
    xx = torch.as_tensor(xs, dtype=torch.float32)[:, None].expand(-1, len(kgrid)).contiguous()
    approx = cheb_clenshaw(xx, cf).numpy()
    exact = np.stack([H(xs, k) for k in kgrid], 1)
    errs.append(np.abs(approx - exact).max())
axi = fig.add_subplot(gs0[0, 2])
axi.plot(RS, errs, "o-", color=GRED, ms=3, lw=1.1)
axi.set_yscale("log"); axi.set_xscale("log", base=2)
axi.set_xticks(RS); axi.set_xticklabels([str(r) for r in RS], fontsize=7)
axi.minorticks_off()
axi.tick_params(labelsize=7.5, pad=1)
axi.set_ylabel("max |Hill $-$ Cheb.|", fontsize=7.5)
axi.set_xlabel("degree $r$", fontsize=7.5, labelpad=1)
axi.grid(True, lw=0.5)
letter(axi, "", 0, 0)
ERRS = errs

# ------------------------------------------------------------------ (b)
gsb = gs[1].subgridspec(1, 2, wspace=0.12)
axb1 = fig.add_subplot(gsb[0, 0])
axb2 = fig.add_subplot(gsb[0, 1], sharey=axb1)
W = [0.8, -0.5, 0.3]
TC = [GBLUE, GPURPLE, GGREEN]
x0 = 0.3
msg_all = []
for ax, ks, ttl in ((axb1, [0.3] * 3, "shared $k$\n1$k$+3$w$ = 4 params"),
                    (axb2, [0.1, 0.3, 0.9], "per-edge $k$\n3$k$+3$w$ = 6 params")):
    for i, (w, k, c) in enumerate(zip(W, ks, TC)):
        m = w * (H(x, k) - H(np.array([x0]), k))
        msg_all.append(m)
        lab = f"$w$={w:+.1f}" if ax is axb1 else f"$k$={k:g}"
        ax.plot(x, m, color=c, lw=1.3, label=lab)
    ax.axhline(0, color=GREY500, lw=0.6)
    ax.axvline(x0, color=GREY500, lw=0.6, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("regulator $x$")
    ax.set_title(ttl, fontsize=7, pad=2, color=GREY700)
    ax.set_xticks([0, 0.5, 1])
axb1.set_ylabel("message $w\\,[H(x)-H(x_0)]$")
axb2.tick_params(labelleft=False)
axb1.set_ylim(-0.55, 0.9)
axb1.legend(loc="upper left", fontsize=6.8, handlelength=1.2, labelspacing=0.2, borderaxespad=0.15)
axb2.legend(loc="upper left", fontsize=6.8, handlelength=1.2, labelspacing=0.2, borderaxespad=0.15)
letter(axb1, "(b)", dx=-0.16)

# ------------------------------------------------------------------ (c)
edges = [(0, 1, +0.9), (1, 2, +0.9), (2, 3, +0.9), (3, 1, -0.9)]
G, T, ZETA, K = 4, 24, 0.8, 0.3
x_base = torch.full((G,), 0.5, dtype=D)
logit0 = torch.log(x_base / (1 - x_base))
H0 = hill4(x_base, torch.tensor(K, dtype=D), 2.0)
Wm = torch.zeros(G, G, dtype=D)
for s, d, w in edges:
    Wm[s, d] = w


def run(persistent):
    xx = x_base.clone()
    xx[0] = (1 - ZETA) * x_base[0]
    traj = [xx.clone()]
    for _ in range(T):
        hd = hill4(xx, torch.tensor(K, dtype=D), 2.0) - H0
        xn = 0.5 * xx + 0.5 * torch.sigmoid(logit0 + hd @ Wm)
        if persistent:
            xn[0] = (1 - ZETA) * x_base[0]
        xx = xn
        traj.append(xx.clone())
    return torch.stack(traj).numpy()


tp, ts = run(True), run(False)
its = np.arange(T + 1)
axc = fig.add_subplot(gs2[0, 0])
axc2 = axc.twinx()
axc2.spines["right"].set_visible(True); axc2.spines["right"].set_color(GGREEN)
axc2.tick_params(axis="y", colors="#1E7B3A", labelsize=7.5); axc2.grid(False)
for g, ax, c in ((0, axc, GBLUE), (2, axc2, GGREEN)):
    ax.plot(its, tp[:, g], color=c, lw=1.5, ls="-")
    ax.plot(its, ts[:, g], color=c, lw=1.3, ls=(0, (3, 2)))
axc.set_xlim(0, T); axc.set_ylim(0.05, 0.85)
axc2.set_ylim(0.4875, 0.5075); axc2.set_yticks([0.49, 0.50])
axc.set_xlabel("iteration"); axc.set_ylabel("$x_0$ (target)", color=GBLUE)
axc2.set_ylabel("$x_2$ (downstream)", color="#1E7B3A")
from matplotlib.lines import Line2D
axc.legend(handles=[Line2D([], [], color=GGREY, lw=1.5, ls="-"),
                    Line2D([], [], color=GGREY, lw=1.3, ls=(0, (3, 2)))],
           labels=["persistent clamp (iPT)", "single injection"],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=2.0,
           labelspacing=0.2, borderaxespad=0.1)
axc.set_title("0$\\to$1 (+), 1$\\to$2 (+), 2$\\to$3 (+), 3$\\to$1 ($-$)", fontsize=7, pad=2, color=GREY700)
letter(axc, "(c)", dx=-0.27, dy=1.14)

# ------------------------------------------------------------------ (d)
with open(os.path.join(ROOT, "artifacts", "scc_bench.json")) as f:
    bench = json.load(f)
cells = [("K562_essential", "K562"), ("rpe1", "RPE1")]
axd = fig.add_subplot(gs2[0, 1])
xp = np.arange(2)
for i, (key, lab) in enumerate(cells):
    b = bench[key]
    axd.bar(i, b["median_speedup"], width=0.8, color=GBLUE, zorder=3)
    axd.text(i, b["median_speedup"] + 0.06, f"{b['median_speedup']:.2f}" + r"$\times$", ha="center",
             va="bottom", fontsize=7.5, color=GREY900, fontweight="bold")
    axd.text(i, 0.99, f"SCC {b['largest_scc']}" + "\n" + r"max|$\Delta x$|" + "\n" + f"{b['max_diff']:.1e}", ha="center",
             va="bottom", fontsize=7, color="white", zorder=4, linespacing=1.05)
axd.axhline(1.0, color=GRED, lw=1.0, ls="--", zorder=2, label=r"global solver (1$\times$)")
axd.legend(loc="upper right", fontsize=7, handlelength=1.6, borderaxespad=0.1)
axd.set_xticks(xp); axd.set_xticklabels([c[1] for c in cells])
axd.set_xlim(-0.6, 1.6); axd.set_ylim(0, 3.9)
axd.set_ylabel("median speed-up")
axd.grid(False)
letter(axd, "(d)", dx=-0.3, dy=1.14)
fig.text(0.10, 0.005, "Speed-up (SCC-aware vs global solver) is measured in a numpy/python\nreference implementation, not an optimized kernel.",
         fontsize=6.8, color=GREY700, ha="left", va="bottom", linespacing=1.15)

save(fig, "fig2_operator", OUT)
print("max Cheb err by r:", dict(zip(RS, [f"{e:.2e}" for e in ERRS])))
print("gene0/2 persistent final:", tp[-1, 0], tp[-1, 2], "single final:", ts[-1, 0], ts[-1, 2])
print("gene2 persistent min/max:", tp[:, 2].min(), tp[:, 2].max(), " single:", ts[:, 2].min(), ts[:, 2].max())
