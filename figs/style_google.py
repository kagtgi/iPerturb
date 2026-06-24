"""Shared Google / Material figure style for the iPerturb paper.
Ports the design-guide tokens to matplotlib (guide section 9.2). Import and call
apply_style() at the top of every figure script."""
import matplotlib as mpl

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
# Container tints
AMBER = "#B06000"

GOOGLE = [GBLUE, GRED, GYELLOW, GGREEN, GPURPLE, GGREY]

def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle":  mpl.cycler(color=GOOGLE),
        "axes.edgecolor":   GREY300, "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": GREY200, "grid.linewidth": 0.8,
        "axes.labelcolor":  GREY700, "text.color": GREY900,
        "xtick.color": GREY500, "ytick.color": GREY500,
        "xtick.labelcolor": GREY700, "ytick.labelcolor": GREY700,
        "axes.spines.top": False, "axes.spines.right": False,
        # fonts: sans for labels (>=7-8pt at final size), mono for numeric ticks
        "font.family": "sans-serif",
        "font.sans-serif": ["Google Sans Flex", "Roboto", "DejaVu Sans"],
        "font.monospace": ["Google Sans Code", "Roboto Mono", "DejaVu Sans Mono"],
        "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "legend.frameon": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })

def mono_ticks(ax):
    """Set numeric tick labels in the mono face (human/machine split)."""
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily("monospace")

def save(fig, stem, outdir="."):
    import os
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(outdir, stem + ext))
