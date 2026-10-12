"""Fig. 2: the iPT operator -- transfer functions, regulator-shared Hill, persistent clamp, SCC execution.
All curves are computed from ipt code; panel (d) reads artifacts/scc_bench.json.
Run from the repo root:  PYTHONPATH=. python scripts/fig2_operator.py"""
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ipt.viz.style import *          # noqa: F401,F403  (palette + apply_style)
from ipt.models.hill import hill4
from ipt.models.basis_hill import fit_table, coef_from_k, cheb_clenshaw

apply_style()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
D = torch.float64
BLUE_RAMP = ["#C6DAFC", GBLUE_L, GBLUE, "#2A62C9", "#16397A"]   # light -> dark
FW, FH = 122.0, 109.5   # figure size, mm (width = LNCS text width)


def H(x, k, n=2.0):
    return hill4(torch.as_tensor(x, dtype=D), torch.tensor(float(k), dtype=D), n).numpy()


def ax_mm(x0, top, w, h, **kw):
    """Axes placed in millimetres: x0 from the left edge, top from the top edge."""
    return fig.add_axes([x0 / FW, 1 - (top + h) / FH, w / FW, h / FH], **kw)


def letter(x, top, s):
    fig_letter(fig, x, top, s, FH)


fig = plt.figure(figsize=(FW * MM, FH * MM))

# ------------------------------------------------------------------ (a)
axa = ax_mm(12, 5.5, 33, 23)
x = np.linspace(0, 1, 401)
axa.plot(x, x, color=GGREY, ls="--", lw=LW, label="linear")
KS = [0.05, 0.15, 0.3, 0.6, 1.0]
for k, c in zip(KS, BLUE_RAMP):
    axa.plot(x, H(x, k), color=c, lw=LW, label=f"$k$={k:g}")
tab = fit_table(8)
coef = coef_from_k(torch.tensor([0.15], dtype=torch.float32), torch.as_tensor(tab, dtype=torch.float32))
cheb = cheb_clenshaw(torch.as_tensor(x, dtype=torch.float32)[:, None].expand(-1, 1), coef)[:, 0].numpy()
axa.plot(x, cheb, color=GRED, ls=(0, (2.5, 2)), lw=LW, label="Cheb. $r$=8, $k$=0.15")
axa.set_xlim(0, 1); axa.set_ylim(0, 1.02)
axa.set_xticks([0, 0.5, 1]); axa.set_yticks([0, 0.5, 1])
axa.set_xlabel("input $x$"); axa.set_ylabel("$H(x;k)$")
hl = axa.get_legend_handles_labels()
axl = ax_mm(49, 5.5, 30, 23); axl.axis("off"); axl.grid(False)
axl.legend(*hl, loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=1.6,
           labelspacing=0.3, borderaxespad=0.0)
letter(0.3, 0.8, "(a)")

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
axi = ax_mm(94, 5.5, 26, 23)
axi.plot(RS, errs, "o-", color=GRED, ms=3, lw=LW)
axi.set_yscale("log"); axi.set_xscale("log", base=2)
axi.set_xticks(RS); axi.set_xticklabels([str(r) for r in RS])
axi.minorticks_off()
axi.set_ylabel("max |Hill $-$ Cheb.|")
axi.set_xlabel("degree $r$")
axi.grid(True, lw=0.5)
ERRS = errs

# ------------------------------------------------------------------ (b)
axb1 = ax_mm(16, 41.5, 48.5, 19)
axb2 = ax_mm(71.5, 41.5, 48.5, 19, sharey=axb1)
W = [0.8, -0.5, 0.3]
TC = [GBLUE, GPURPLE, GGREEN]
x0 = 0.3
msg_all = []
for ax, ks, ttl in ((axb1, [0.3] * 3, "shared $k$: 1$k$+3$w$ = 4 params"),
                    (axb2, [0.1, 0.3, 0.9], "per-edge $k$: 3$k$+3$w$ = 6 params")):
    for i, (w, k, c) in enumerate(zip(W, ks, TC)):
        m = w * (H(x, k) - H(np.array([x0]), k))
        msg_all.append(m)
        lab = f"$w$={w:+.1f}" if ax is axb1 else f"$k$={k:g}"
        ax.plot(x, m, color=c, lw=LW, label=lab)
    ax.axhline(0, color=GREY500, lw=LW_THIN)
    ax.axvline(x0, color=GREY500, lw=LW_THIN, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("regulator $x$")
    ax.set_title(ttl, fontsize=FS_TITLE, pad=3, color=GREY700)
    ax.set_xticks([0, 0.5, 1])
axb1.set_ylabel("message\n$w\\,[H(x)-H(x_0)]$", linespacing=1.1)
axb2.tick_params(labelleft=False)
axb1.set_ylim(-0.55, 0.9)
axb1.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.1)
axb2.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.1)
letter(0.3, 35.5, "(b)")

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
axc = ax_mm(12, 75, 38, 19)
axc2 = axc.twinx()
axc2.spines["right"].set_visible(True); axc2.spines["right"].set_color(GGREEN)
axc2.tick_params(axis="y", colors="#1E7B3A"); axc2.grid(False)
for g, ax, c in ((0, axc, GBLUE), (2, axc2, GGREEN)):
    ax.plot(its, tp[:, g], color=c, lw=LW, ls="-")
    ax.plot(its, ts[:, g], color=c, lw=LW, ls=(0, (3, 2)))
axc.set_xlim(0, T); axc.set_ylim(0.05, 0.85)
axc2.set_ylim(0.4875, 0.5075); axc2.set_yticks([0.49, 0.50])
axc.set_xlabel("iteration"); axc.set_ylabel("$x_0$ (target)", color=GBLUE)
axc2.set_ylabel("$x_2$ (downstream)", color="#1E7B3A")
axc.legend(handles=[Line2D([], [], color=GGREY, lw=LW, ls="-"),
                    Line2D([], [], color=GGREY, lw=LW, ls=(0, (3, 2)))],
           labels=["persistent clamp (iPT)", "single injection"],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.0,
           labelspacing=0.2, borderaxespad=0.1)
axc.set_title("0$\\to$1 (+), 1$\\to$2 (+), 2$\\to$3 (+), 3$\\to$1 ($-$)", fontsize=FS_TITLE, pad=3,
              color=GREY700)
letter(0.3, 69, "(c)")

# ------------------------------------------------------------------ (d)
with open(os.path.join(ROOT, "artifacts", "scc_bench.json")) as f:
    bench = json.load(f)
cells = [("K562_essential", "K562"), ("rpe1", "RPE1")]
axd = ax_mm(82, 75, 38, 19)
xp = np.arange(2)
for i, (key, lab) in enumerate(cells):
    b = bench[key]
    axd.bar(i, b["median_speedup"], width=0.8, color=GBLUE, zorder=3)
    axd.text(i, b["median_speedup"] + 0.05, f"{b['median_speedup']:.2f}" + r"$\times$", ha="center",
             va="bottom", fontsize=FS_BODY, color=GREY900, fontweight="bold")
    axd.text(i, 0.12, f"SCC {b['largest_scc']}" + "\n" + r"max|$\Delta x$|" + "\n" + f"{b['max_diff']:.1e}",
             ha="center", va="bottom", fontsize=FS_BODY, color="white", zorder=4, linespacing=1.05)
axd.axhline(1.0, color=GRED, lw=LW, ls="--", zorder=2, label=r"global solver (1$\times$)")
axd.legend(loc="upper left", handlelength=1.6, borderaxespad=0.0)
axd.set_xticks(xp); axd.set_xticklabels([c[1] for c in cells])
axd.set_xlim(-0.6, 1.6); axd.set_ylim(0, 3.9); axd.set_yticks([0, 1, 2, 3])
axd.set_ylabel("median speed-up")
axd.grid(False)
letter(70, 69, "(d)")
fig.text(0.3 / FW, 0.3 / FH, "Speed-up (SCC-aware vs global solver) is measured in a numpy/python\n"
         "reference implementation, not an optimized kernel.",
         fontsize=FS_BODY, color=GREY700, ha="left", va="bottom", linespacing=1.15)

save(fig, "fig2_operator", OUT)
print("max Cheb err by r:", dict(zip(RS, [f"{e:.2e}" for e in ERRS])))
print("gene0/2 persistent final:", tp[-1, 0], tp[-1, 2], "single final:", ts[-1, 0], ts[-1, 2])
print("gene2 persistent min/max:", tp[:, 2].min(), tp[:, 2].max(), " single:", ts[:, 2].min(), ts[:, 2].max())
