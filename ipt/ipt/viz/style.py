"""Shared Google / Material figure style for the iPerturb paper.
Ports the design-guide tokens to matplotlib (guide section 9.2). Import and call
apply_style() at the top of every figure script."""
import matplotlib as mpl
from pathlib import Path as _Path
from matplotlib import font_manager as _fm

# the same bundled Google Sans files that the overview figure (Fig. 1) uses
FONT_DIR = _Path("D:/iPerturb/lnbi/figure_tools/assets/fonts/Google_Sans/static")
_REGISTERED = []
for _f in ("GoogleSans-Regular.ttf", "GoogleSans-Medium.ttf", "GoogleSans-Bold.ttf"):
    _p = FONT_DIR / _f
    if _p.exists():
        _fm.fontManager.addfont(str(_p))
        _REGISTERED.append(_p.name)

# Brand palette, semantic order (guide section 8.2)
GBLUE   = "#4285F4"   # primary / proposed (iPerturb)
GRED    = "#EA4335"   # baseline / competitor (STATE)
GYELLOW = "#FBBC04"   # caution / intermediate
GGREEN  = "#34A853"   # success / upper
GPURPLE = "#5E35B1"   # extra / new
GGREY   = "#5F6368"   # reference / other
# Neutrals
GREY300 = "#DADCE0"; GREY200 = "#E8EAED"; GREY500 = "#9AA0A6"; GREY700 = "#5F6368"
GREY900 = "#202124"
# Sequential ramp for "bad -> good" heatmaps (guide section 8.2): red -> amber -> green
SEQ_BAD_GOOD = ["#EA4335", "#FBBC04", "#34A853"]

# Lighter (300-level) tints of the brand hues -- softer fills for dense charts
GBLUE_L   = "#8AB4F8"   # blue 300
GRED_L    = "#F28B82"   # red 300
GYELLOW_L = "#FDD663"   # yellow 300
GGREEN_L  = "#81C995"   # green 300
GPURPLE_L = "#B39DDB"   # purple 300
GGREY_L   = "#BDC1C6"   # grey 400
SEQ_BAD_GOOD_L = ["#F6AEA6", "#FDD663", "#A8D5B5"]   # lighter bad -> good ramp
# Container tints
AMBER = "#B06000"

GOOGLE = [GBLUE, GRED, GYELLOW, GGREEN, GPURPLE, GGREY]

# Paper typography (LNCS, 122 mm text width; figures are built at exactly 122 mm
# wide and included at \linewidth, so 1 pt in the PDF = 1 pt on the page).
TEXT_W_MM = 122.0
FS_BODY = 7.0     # ticks, axis labels, legends, annotations, node labels
FS_TITLE = 8.0    # panel titles
FS_LETTER = 9.0   # panel letters "(a)"
LW = 0.8          # data lines
LW_THIN = 0.6     # axes, ticks, reference / grid lines
MM = 1 / 25.4
# One arrow style for every figure: filled triangular head ~1.9 mm long, ~1.6 mm wide.
# Use as FancyArrowPatch(p, q, **ARROW_KW, color=..., lw=LW_ARROW).
ARROW_STYLE = "-|>,head_length=0.6,head_width=0.25"
ARROW_MS = 9.0
ARROW_KW = dict(arrowstyle=ARROW_STYLE, mutation_scale=ARROW_MS)
LW_ARROW = 0.8


def _mathtext_scripts():
    """Mathtext shrinks sub/superscripts to 70% (7 pt -> 4.9 pt). Raise the factor so
    a 7 pt label gets >= 6 pt scripts."""
    try:
        import matplotlib._mathtext as _mt
        _mt.SHRINK_FACTOR = 0.86
    except Exception:
        pass


def apply_style():
    _mathtext_scripts()
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle":  mpl.cycler(color=GOOGLE),
        "axes.edgecolor":   GREY300, "axes.linewidth": LW_THIN,
        "axes.grid": True, "grid.color": GREY200, "grid.linewidth": 0.5,
        "axes.labelcolor":  GREY700, "text.color": GREY900,
        "xtick.color": GREY500, "ytick.color": GREY500,
        "xtick.labelcolor": GREY700, "ytick.labelcolor": GREY700,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Google Sans", "Google Sans Flex", "Roboto", "DejaVu Sans"],
        "font.monospace": ["Google Sans Code", "Roboto Mono", "DejaVu Sans Mono"],
        "mathtext.fontset": "custom", "mathtext.rm": "Google Sans", "mathtext.it": "Google Sans", "mathtext.bf": "Google Sans:bold",
        "font.size": FS_BODY, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_BODY,
        "xtick.labelsize": FS_BODY, "ytick.labelsize": FS_BODY, "legend.fontsize": FS_BODY,
        "legend.title_fontsize": FS_BODY, "figure.titlesize": FS_TITLE,
        "legend.frameon": False,
        "xtick.major.width": LW_THIN, "ytick.major.width": LW_THIN,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0,
        "axes.titlepad": 3.0,
        "lines.linewidth": LW, "lines.markersize": 3.5, "patch.linewidth": LW_THIN,
        "savefig.dpi": 600, "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def panel_letter(ax, s, dx=0.0, dy=1.0, transform=None, **kw):
    """Bold 9 pt panel letter. dx/dy in the given transform (default ax.transAxes)."""
    tr = transform if transform is not None else ax.transAxes
    kw.setdefault("ha", "left"); kw.setdefault("va", "bottom")
    return ax.text(dx, dy, s, transform=tr, fontsize=FS_LETTER, fontweight="bold",
                   color=GREY900, **kw)


def align_letters_to_ink(fig, pairs, min_x_mm=0.3):
    """Move each panel letter (created with panel_letter) horizontally so it starts at the
    left edge of its panel's ink (tick labels, axis label), like the mm-placed letters."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ink = {}
    for ax, t in pairs:
        t.set_visible(False)
        ink[id(ax)] = ax.get_tightbbox(r).x0
        t.set_visible(True)
    # panels in the same column share one letter x (the leftmost ink of that column)
    col = {}
    for ax, _ in pairs:
        k = round(ax.get_position().x0, 2)
        col[k] = min(col.get(k, 1e9), ink[id(ax)])
    for ax, t in pairs:
        x0 = col[round(ax.get_position().x0, 2)]
        x0 = max(x0, min_x_mm / 25.4 * fig.dpi)
        xa = ax.transAxes.inverted().transform((x0, 0))[0]
        t.set_x(xa)


def fig_letter(fig, x_mm, y_mm_from_top, s, fig_h_mm):
    """Panel letter placed in figure millimetres (top-left anchored)."""
    return fig.text(x_mm / TEXT_W_MM, 1 - y_mm_from_top / fig_h_mm, s, fontsize=FS_LETTER,
                    fontweight="bold", color=GREY900, ha="left", va="top")


def mono_ticks(ax):
    """Set numeric tick labels in the mono face (human/machine split)."""
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily("monospace")

def save(fig, stem, outdir="."):
    import os
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(outdir, stem + ext))
