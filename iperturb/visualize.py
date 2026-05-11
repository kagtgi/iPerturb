"""
Visualisation utilities for iPerturb.

Includes:
  - Publication-quality matplotlib figures (Q1 journal style)
  - Interactive Cytoscape.js HTML generation (for Jupyter / Colab)
  - Full-GRN Cytoscape rendering + GraphML export
  - Per-gene before/after/changed subgraph panels
  - Weight heatmaps
  - Loss curves
"""

from __future__ import annotations

import json
import html as html_module
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm


# ── Publication style ─────────────────────────────────────────────────────────
# Single-column = 3.46 in (88 mm), 1.5-col = 5.0 in, double = 7.20 in (183 mm)
COL_1, COL_15, COL_2 = 3.46, 5.0, 7.20

mpl.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset":    "dejavusans",
    "font.size":           7,
    "axes.titlesize":      8,
    "axes.labelsize":      7,
    "xtick.labelsize":     6.5,
    "ytick.labelsize":     6.5,
    "legend.fontsize":     6.5,
    "legend.frameon":      False,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.5,  "ytick.major.width":  0.5,
    "xtick.major.size":    2.5,  "ytick.major.size":   2.5,
    "xtick.minor.width":   0.4,  "ytick.minor.width":  0.4,
    "lines.linewidth":     1.0,
    "patch.linewidth":     0.5,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           False,
    "grid.linewidth":      0.3,
    "grid.alpha":          0.4,
    "figure.dpi":          150,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "savefig.transparent": False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "svg.fonttype":        "none",
})

# Colorblind-safe RdBu_r palette (Blue = repression, Red = activation)
CMAP_DIV = plt.get_cmap("RdBu_r")
ACT_C    = "#B2182B"   # deep red    — activation
REP_C    = "#2166AC"   # deep blue   — repression
UNK_C    = "#BDBDBD"   # neutral grey
NODE_CTR = "#F4A261"   # warm sand   — perturbed / centre gene
NODE_H1  = "#457B9D"   # slate blue  — direct neighbours
NODE_H2  = "#D9D9D9"   # cool grey   — second-hop
EDGE_LBL = "#3A3A3A"

NODE_C = {"center": NODE_CTR, "hop1": NODE_H1, "hop2": NODE_H2}
NODE_S = {"center": 360,      "hop1": 220,     "hop2": 110}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, plot_dir: str | Path, stem: str, *, also_pdf: bool = True):
    """Save figure as PNG (600 dpi) and optionally PDF (vector TrueType)."""
    d = Path(plot_dir)
    d.mkdir(exist_ok=True)
    png = d / f"{stem}.png"
    fig.savefig(png, dpi=600, facecolor="white")
    if also_pdf:
        fig.savefig(d / f"{stem}.pdf", facecolor="white")


# ── Colour helpers ────────────────────────────────────────────────────────────

def eff_color(w: float, thr: float = 0.02) -> str:
    """Red = activation (eff > 0), Blue = repression (eff < 0), Grey = near-zero."""
    return ACT_C if w > thr else REP_C if w < -thr else UNK_C


def before_edge_color(sign: int) -> str:
    return ACT_C if sign == 1 else REP_C if sign == -1 else UNK_C


def linestyle(sign: int) -> str:
    return "solid" if sign != 0 else "dashed"


# ── Subgraph helpers ──────────────────────────────────────────────────────────

def top_changed_subdf(
    df: pd.DataFrame,
    gene: str,
    eff_before: dict,
    eff_after: dict,
    top_k: int = 10,
) -> tuple[pd.DataFrame, set]:
    """
    Ego subgraph: edges where gene is source OR target,
    ranked by |eff_after − eff_before|, top-K kept.
    """
    direct = df[(df["source"] == gene) | (df["target"] == gene)].copy()
    if direct.empty:
        return pd.DataFrame(), set()
    edges  = [(r.source, r.target) for _, r in direct.iterrows()]
    deltas = {e: abs(eff_after.get(e, 0.0) - eff_before.get(e, 0.0)) for e in edges}
    top_set = set(sorted(deltas, key=lambda e: deltas[e], reverse=True)[:top_k])
    filtered = direct[direct.apply(lambda r: (r.source, r.target) in top_set, axis=1)]
    G    = nx.from_pandas_edgelist(filtered, "source", "target", create_using=nx.DiGraph())
    hop1 = (set(G.successors(gene)) | set(G.predecessors(gene))) - {gene}
    return filtered, hop1


def _node_hop(n: str, center: str, hop1: set) -> str:
    return "center" if n == center else "hop1" if n in hop1 else "hop2"


def _layout(G: nx.DiGraph, gene: str, hop1: set) -> dict:
    hop2   = set(G.nodes) - {gene} - hop1
    shells = [[gene], sorted(hop1 & set(G.nodes))]
    if hop2:
        shells.append(sorted(hop2 & set(G.nodes)))
    try:
        return nx.shell_layout(G, nlist=[s for s in shells if s])
    except Exception:
        return nx.kamada_kawai_layout(G)


# ── matplotlib per-gene publication figure ───────────────────────────────────

def _draw_network(
    G: nx.DiGraph,
    gene: str,
    hop1: set,
    *,
    sign_map: dict,
    edge_color_map: dict,
    edge_width_map: dict,
    edge_label_map: dict,
    node_sizes_override: list | None = None,
    panel_label: str,
    stem: str,
    plot_dir: str | Path,
):
    pos   = _layout(G, gene, hop1)
    nodes = list(G.nodes())
    sizes = (node_sizes_override if node_sizes_override is not None
             else [NODE_S[_node_hop(n, gene, hop1)] for n in nodes])
    size_map = dict(zip(nodes, sizes))

    fig, ax = plt.subplots(figsize=(COL_1, COL_1 * 0.95))

    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=nodes,
        node_color=[NODE_C[_node_hop(n, gene, hop1)] for n in nodes],
        node_size=sizes,
        edgecolors="white", linewidths=0.8, alpha=0.95,
    )
    label_artists = nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5,
                                             font_family="sans-serif", font_weight="bold")
    for txt in label_artists.values():
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    edges    = list(G.edges())
    solid_e  = [e for e in edges if sign_map.get(e, 0) != 0]
    dashed_e = [e for e in edges if sign_map.get(e, 0) == 0]
    common   = dict(ax=ax, arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.10",
                    node_size=[size_map[n] for n in nodes])
    if solid_e:
        nx.draw_networkx_edges(G, pos, edgelist=solid_e,
                                edge_color=[edge_color_map[e] for e in solid_e],
                                width=[edge_width_map[e] for e in solid_e],
                                style="solid", alpha=0.92, **common)
    if dashed_e:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_e,
                                edge_color=[edge_color_map[e] for e in dashed_e],
                                width=[edge_width_map[e] for e in dashed_e],
                                style=(0, (3, 2)), alpha=0.75, **common)

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_label_map, ax=ax,
        font_size=5.5, font_color=EDGE_LBL,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
        rotate=False,
    )
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.axis("off"); ax.margins(0.10)
    fig.tight_layout(pad=0.2)
    save_fig(fig, plot_dir, stem)
    plt.close(fig)


def save_before_figure(
    df: pd.DataFrame, gene: str, cell_line: str,
    eff_before: dict, eff_after: dict,
    plot_dir: str | Path, top_k: int = 10,
):
    sub, hop1 = top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty:
        return
    G        = nx.from_pandas_edgelist(sub, "source", "target", create_using=nx.DiGraph())
    edges    = list(G.edges())
    sign_map = {(r.source, r.target): int(r.get("sign", 0)) for _, r in sub.iterrows()}
    _draw_network(G, gene, hop1,
                  sign_map      ={e: sign_map.get(e, 0) for e in edges},
                  edge_color_map={e: before_edge_color(sign_map.get(e, 0)) for e in edges},
                  edge_width_map={e: 1.4 for e in edges},
                  edge_label_map={e: f"{eff_before.get(e, 0.0):+.2f}" for e in edges},
                  panel_label="a", stem=f"{cell_line}_{gene}_before", plot_dir=plot_dir)


def save_after_figure(
    df: pd.DataFrame, gene: str, cell_line: str,
    eff_before: dict, eff_after: dict,
    model, plot_dir: str | Path, top_k: int = 10,
):
    sub, hop1 = top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty:
        return
    G     = nx.from_pandas_edgelist(sub, "source", "target", create_using=nx.DiGraph())
    V_np  = model.V.detach().cpu().numpy()
    V_map = {g: float(v) for g, v in zip(model.gene_names, V_np)}
    V_max = max(V_map.values()) + 1e-9
    edges = list(G.edges())
    eff_m = {e: eff_after.get(e, 0.0) for e in edges}
    e_max = max(abs(v) for v in eff_m.values()) + 1e-9
    sign_map = {(r.source, r.target): int(r.get("sign", 0)) for _, r in sub.iterrows()}
    nodes    = list(G.nodes())
    sizes    = [380 if _node_hop(n, gene, hop1) == "center"
                else 80 + 320 * V_map.get(n, 1.0) / V_max
                for n in nodes]
    _draw_network(G, gene, hop1,
                  sign_map           ={e: sign_map.get(e, 0) for e in edges},
                  edge_color_map     ={e: eff_color(eff_m[e]) for e in edges},
                  edge_width_map     ={e: 0.6 + 3.0 * abs(eff_m[e]) / e_max for e in edges},
                  edge_label_map     ={e: f"{eff_m[e]:+.2f}" for e in edges},
                  node_sizes_override=sizes,
                  panel_label="b", stem=f"{cell_line}_{gene}_after", plot_dir=plot_dir)


def save_changed_figure(
    df: pd.DataFrame, gene: str, cell_line: str,
    eff_before: dict, eff_after: dict,
    plot_dir: str | Path, top_k: int = 10,
):
    sub, hop1 = top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty:
        return
    G       = nx.from_pandas_edgelist(sub, "source", "target", create_using=nx.DiGraph())
    edges   = list(G.edges())
    delta   = {e: eff_after.get(e, 0.0) - eff_before.get(e, 0.0) for e in edges}
    d_max   = max(abs(v) for v in delta.values()) + 1e-9
    sign_map = {(r.source, r.target): int(r.get("sign", 0)) for _, r in sub.iterrows()}
    _draw_network(G, gene, hop1,
                  sign_map      ={e: sign_map.get(e, 0) for e in edges},
                  edge_color_map={e: eff_color(eff_after.get(e, 0.0)) for e in edges},
                  edge_width_map={e: 0.6 + 3.6 * abs(delta[e]) / d_max for e in edges},
                  edge_label_map={e: f"Δ{delta[e]:+.2f}" for e in edges},
                  panel_label="c", stem=f"{cell_line}_{gene}_changed", plot_dir=plot_dir)


# ── Weight heatmap ────────────────────────────────────────────────────────────

def save_weight_heatmap(
    eff_weights: dict,
    label: str,
    cell_line: str,
    plot_dir: str | Path,
    max_label: int = 60,
):
    """
    Heatmap of GRN effective weights (RdBu_r, TwoSlopeNorm at 0).
    Saved as PNG (600 dpi) + PDF (vector).
    """
    rec = [(s, t, w) for (s, t), w in eff_weights.items()]
    df  = pd.DataFrame(rec, columns=["source", "target", "weight"])
    src = sorted(df["source"].unique())
    tgt = sorted(df["target"].unique())
    M   = (df.pivot_table(index="source", columns="target",
                           values="weight", aggfunc="first")
              .reindex(index=src, columns=tgt).fillna(0.0).values)
    n_s, n_t = len(src), len(tgt)

    vmax = float(np.quantile(np.abs(M), 0.98)) or 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    w   = min(COL_2, max(COL_1, 0.045 * n_t + 1.4))
    h   = min(COL_2, max(COL_1, 0.045 * n_s + 1.0))
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(M, aspect="auto", cmap=CMAP_DIV, norm=norm,
                   interpolation="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, aspect=18, pad=0.015)
    cb.set_label(r"effective weight  $\hat{w}\!\cdot\!\varphi(x_0)$",
                 fontsize=6.5, labelpad=4)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)

    if n_t <= max_label:
        ax.set_xticks(range(n_t))
        ax.set_xticklabels(tgt, rotation=90, fontsize=max(4, 380 // n_t))
    else:
        ax.set_xticks([])
        ax.text(0.5, -0.04, f"{n_t} target genes",
                transform=ax.transAxes, ha="center", va="top", fontsize=6)
    if n_s <= max_label:
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(src, fontsize=max(4, 380 // n_s))
    else:
        ax.set_yticks([])
        ax.text(-0.04, 0.5, f"{n_s} source genes",
                transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=6)

    ax.set_xlabel("Target gene", labelpad=3)
    ax.set_ylabel("Source gene", labelpad=3)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    fig.tight_layout(pad=0.3)
    save_fig(fig, plot_dir, f"{cell_line}_heatmap_{label}")
    plt.close(fig)


# ── Loss curves ───────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_h: list[float],
    val_h: list[float],
    cell_line: str,
    plot_dir: str | Path,
):
    """Training / validation loss curves (log-scale y axis)."""
    fig, ax = plt.subplots(figsize=(COL_1, COL_1 * 0.62))
    ep = np.arange(1, len(train_h) + 1)
    ax.plot(ep, train_h, color="#2A6F97", lw=1.1, label="train")
    ax.plot(ep, val_h,   color="#E07A5F", lw=1.1, label="validation")
    ax.set_xlabel("Epoch", labelpad=2)
    ax.set_ylabel("Loss",  labelpad=2)
    ax.set_yscale("log")
    ax.legend(loc="upper right", borderaxespad=0.2, handlelength=1.6)
    ax.grid(True, which="major", axis="y")
    fig.tight_layout(pad=0.2)
    save_fig(fig, plot_dir, f"{cell_line}_loss_curves")
    plt.close(fig)


# ── Full-GRN Cytoscape.js (interactive HTML) ─────────────────────────────────

def build_full_cytoscape_html(edges_df: pd.DataFrame, height_px: int = 800) -> str:
    """
    Render the full selected GRN as an interactive Cytoscape.js graph.
    Returns an HTML string suitable for display in Jupyter / Colab.
    """
    required = {"source", "target", "sign", "level"}
    missing  = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = edges_df.copy().reset_index(drop=True)
    if "db" not in df.columns:
        df["db"] = "NA"

    deg     = Counter(df["source"].tolist() + df["target"].tolist())
    max_deg = max(deg.values()) if deg else 1
    node_ids = sorted(set(df["source"]) | set(df["target"]))

    cy_nodes = [
        {"data": {"id": str(n), "label": str(n), "deg": int(deg[n]),
                  "size": float(10 + 35 * (deg[n] / max_deg))}}
        for n in node_ids
    ]

    edge_colour = {1: "#27ae60", -1: "#e74c3c", 0: "#95a5a6"}
    level_width = {1: 2.4, 2: 1.8, 3: 1.3, 4: 0.8}

    cy_edges = []
    for i, e in enumerate(df.itertuples(index=False)):
        sign  = int(getattr(e, "sign"))
        level = int(getattr(e, "level"))
        cy_edges.append({"data": {
            "id":     f"edge_{i}",
            "source": str(getattr(e, "source")),
            "target": str(getattr(e, "target")),
            "sign":   sign, "level": level,
            "db":     str(getattr(e, "db")),
            "color":  edge_colour.get(sign, "#95a5a6"),
            "width":  level_width.get(level, 0.8),
        }})

    elements_json = json.dumps(cy_nodes + cy_edges)

    return f"""
<div id="grn-wrap" style="font-family:sans-serif">
  <div style="display:flex;gap:16px;align-items:center;margin-bottom:8px;flex-wrap:wrap">
    <b style="font-size:15px">FULL GRN: {len(df)} edges / {len(node_ids)} genes</b>
    <label>Layout&nbsp;
      <select id="layoutSel" onchange="changeLayout()" style="padding:3px 6px">
        <option value="cose" selected>CoSE force</option>
        <option value="circle">Circle</option>
        <option value="grid">Grid</option>
        <option value="concentric">Concentric</option>
        <option value="breadthfirst">BFS tree</option>
      </select>
    </label>
    <label>Highlight gene&nbsp;
      <input id="geneBox" type="text" placeholder="e.g. MYC"
             style="width:110px;padding:3px"
             oninput="highlightGene(this.value.trim())">
    </label>
    <button onclick="cy.fit()" style="padding:4px 10px">Reset view</button>
  </div>
  <div style="display:flex;gap:12px;margin-bottom:6px;font-size:12px;flex-wrap:wrap">
    <span><span style="color:#27ae60;font-weight:bold">&#x2192;</span> Activation</span>
    <span><span style="color:#e74c3c;font-weight:bold">&#x2192;</span> Repression</span>
    <span><span style="color:#95a5a6;font-weight:bold">&#x2192;</span> Unsigned</span>
  </div>
  <div id="cy" style="width:100%;height:{height_px}px;border:1px solid #ddd;
       border-radius:6px;background:#fafafa"></div>
  <div id="tip" style="display:none;position:fixed;background:rgba(0,0,0,.78);
       color:#fff;padding:6px 10px;border-radius:5px;font-size:12px;
       pointer-events:none;z-index:9999"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<script>
const ALL_ELEMENTS = {elements_json};
var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: ALL_ELEMENTS,
  style: [
    {{ selector:'node', style:{{'label':'data(label)','width':'data(size)',
       'height':'data(size)','background-color':'#3498db','color':'#222',
       'font-size':7,'text-valign':'center','text-halign':'center',
       'text-wrap':'wrap','overlay-padding':'4px'}} }},
    {{ selector:'edge', style:{{'width':'data(width)','line-color':'data(color)',
       'target-arrow-color':'data(color)','target-arrow-shape':'triangle',
       'curve-style':'bezier','opacity':0.45}} }},
    {{ selector:'.faded', style:{{'opacity':0.04}} }},
    {{ selector:'.highlighted', style:{{'background-color':'#f39c12',
       'border-width':3,'border-color':'#e67e22','z-index':10,'font-size':13}} }},
    {{ selector:'.highlighted-edge', style:{{'opacity':1,'width':3.5}} }}
  ],
  layout:{{ name:'cose', animate:false, randomize:true, nodeRepulsion:12000,
            idealEdgeLength:60, edgeElasticity:80, numIter:2500 }},
  wheelSensitivity: 0.25
}});
const tip = document.getElementById('tip');
cy.on('mouseover','node', e => {{
  const d=e.target.data();
  tip.style.display='block';
  tip.innerHTML=`<b>${{d.label}}</b><br>degree: ${{d.deg}}`;
}});
cy.on('mouseover','edge', e => {{
  const d=e.target.data();
  const s=d.sign===1?'activation':d.sign===-1?'repression':'unsigned';
  tip.style.display='block';
  tip.innerHTML=`${{d.source}} &#x2192; ${{d.target}}<br>${{s}} | L${{d.level}}<br><small>${{d.db}}</small>`;
}});
cy.on('mouseout', () => {{ tip.style.display='none'; }});
document.addEventListener('mousemove', e => {{
  tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-10)+'px';
}});
function changeLayout() {{
  cy.layout({{ name: document.getElementById('layoutSel').value,
               animate:false, randomize:true, nodeRepulsion:12000 }}).run();
}}
function highlightGene(g) {{
  g=g.toUpperCase();
  cy.elements().removeClass('highlighted highlighted-edge faded');
  if(!g) return;
  const t=cy.nodes().filter(n=>n.data('label').toUpperCase()===g);
  if(!t.length) return;
  cy.elements().addClass('faded');
  const h=t.closedNeighborhood();
  h.removeClass('faded'); h.nodes().addClass('highlighted'); h.edges().addClass('highlighted-edge');
  t.addClass('highlighted'); cy.fit(h,60);
}}
</script>
"""


# ── Per-gene Cytoscape panels ─────────────────────────────────────────────────

_CDN_LOADED = False


def render_cytoscape(html_body: str):
    """Display an HTML fragment with Cytoscape.js in a Jupyter/Colab notebook."""
    global _CDN_LOADED
    from IPython.display import display, HTML
    cdn = ""
    if not _CDN_LOADED:
        cdn = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/'
               'cytoscape/3.28.1/cytoscape.min.js"></script>')
        _CDN_LOADED = True
    display(HTML(f"{cdn}<div style='margin-bottom:28px'>{html_body}</div>"))


def _cy_panel_html(gene: str, elements_json: str, cy_id: str,
                   title: str, height: int = 480) -> str:
    return f"""
<div style="font-family:Arial,Helvetica,sans-serif;margin-bottom:2px">
  <b style="font-size:12px">{html_module.escape(title)}</b>
</div>
<div id="{cy_id}" style="width:100%;height:{height}px;border:1px solid #ddd;
     border-radius:6px;background:#fafafa"></div>
<script>
(function(){{
  var cy=cytoscape({{
    container:document.getElementById('{cy_id}'),
    elements:{elements_json},
    layout:{{name:'cose',animate:false,nodeRepulsion:6000,idealEdgeLength:80}},
    style:[
      {{selector:'node',style:{{'label':'data(label)','font-size':9,
         'font-family':'Arial,Helvetica,sans-serif',
         'text-valign':'center','text-halign':'center',
         'width':'data(size)','height':'data(size)',
         'background-color':'data(color)'}}}},
      {{selector:'node[hop="center"]',style:{{'font-size':12,'font-weight':'bold'}}}},
      {{selector:'edge',style:{{'line-color':'data(color)',
         'target-arrow-color':'data(color)','target-arrow-shape':'triangle',
         'curve-style':'bezier','line-style':'data(linestyle)',
         'width':'data(width)','label':'data(label)',
         'font-size':7,'font-family':'Arial,Helvetica,sans-serif',
         'text-rotation':'autorotate','color':'#333','opacity':0.9}}}}
    ],
    wheelSensitivity:0.3
  }});
}})();
</script>"""


def before_cy_html(
    df: pd.DataFrame, gene: str,
    eff_before: dict, eff_after: dict,
    top_k: int = 10, height: int = 480,
) -> str:
    sub, hop1 = top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty:
        return f"<p><b>{html_module.escape(gene)}</b>: not in GRN or no changed edges.</p>"
    G     = nx.from_pandas_edgelist(sub, "source", "target", create_using=nx.DiGraph())
    nodes = [{"data": {"id": n, "label": n, "hop": _node_hop(n, gene, hop1),
                        "color": NODE_C[_node_hop(n, gene, hop1)],
                        "size": {"center": 44, "hop1": 30, "hop2": 22}[_node_hop(n, gene, hop1)]}}
             for n in G.nodes]
    edges = []
    for _, r in sub.iterrows():
        sign = int(r.get("sign", 0))
        w    = eff_before.get((r.source, r.target), 0.0)
        edges.append({"data": {"id": f"{r.source}__{r.target}",
                                "source": r.source, "target": r.target,
                                "color": before_edge_color(sign),
                                "linestyle": linestyle(sign),
                                "width": 2.5, "label": f"{w:+.2f}"}})
    return _cy_panel_html(gene, json.dumps(nodes + edges), f"cy_{gene}_pre",
                           f"{gene} — before (solid=DB-sign · dashed=learned · label=init eff)",
                           height)


def after_cy_html(
    df: pd.DataFrame, gene: str,
    eff_before: dict, eff_after: dict,
    model, top_k: int = 10, height: int = 480,
) -> str:
    sub, hop1 = top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty:
        return f"<p><b>{html_module.escape(gene)}</b>: not in GRN or no changed edges.</p>"
    G     = nx.from_pandas_edgelist(sub, "source", "target", create_using=nx.DiGraph())
    V_np  = model.V.detach().cpu().numpy()
    V_map = {g: float(v) for g, v in zip(model.gene_names, V_np)}
    V_max = max(V_map.values()) + 1e-9
    effs  = [abs(eff_after.get((r.source, r.target), 0.0)) for _, r in sub.iterrows()]
    max_eff = max(effs) + 1e-9
    nodes = [{"data": {"id": n, "label": n, "hop": _node_hop(n, gene, hop1),
                        "color": NODE_C[_node_hop(n, gene, hop1)],
                        "size": round(20 + 32 * V_map.get(n, 1.0) / V_max, 1)}}
             for n in G.nodes]
    edges = []
    for _, r in sub.iterrows():
        sign = int(r.get("sign", 0))
        eff  = eff_after.get((r.source, r.target), 0.0)
        edges.append({"data": {"id": f"{r.source}__{r.target}",
                                "source": r.source, "target": r.target,
                                "color": eff_color(eff),
                                "linestyle": linestyle(sign),
                                "width": round(1.0 + 7.0 * abs(eff) / max_eff, 2),
                                "label": f"{eff:+.2f}"}})
    return _cy_panel_html(gene, json.dumps(nodes + edges), f"cy_{gene}_post",
                           f"{gene} — after (solid=DB-sign · dashed=learned · label=eff)",
                           height)


# ── GraphML export ────────────────────────────────────────────────────────────

def export_graphml(edges_df: pd.DataFrame, path: str | Path):
    """Export full GRN as GraphML for Cytoscape desktop."""
    required = {"source", "target", "sign", "level"}
    missing  = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df    = edges_df.copy()
    if "db" not in df.columns:
        df["db"] = "NA"
    nodes = sorted(set(df["source"]) | set(df["target"]))

    def esc(x):
        return html_module.escape(str(x), quote=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
        '  <key id="sign"  for="edge" attr.name="sign"  attr.type="int"/>',
        '  <key id="level" for="edge" attr.name="level" attr.type="int"/>',
        '  <key id="db"    for="edge" attr.name="db"    attr.type="string"/>',
        '  <graph id="GRN" edgedefault="directed">',
    ]
    for n in nodes:
        lines.append(f'    <node id="{esc(n)}"/>')
    for i, e in enumerate(df.itertuples(index=False)):
        lines.extend([
            f'    <edge id="e{i}" source="{esc(getattr(e,"source"))}" target="{esc(getattr(e,"target"))}">',
            f'      <data key="sign">{int(getattr(e,"sign"))}</data>',
            f'      <data key="level">{int(getattr(e,"level"))}</data>',
            f'      <data key="db">{esc(getattr(e,"db"))}</data>',
            '    </edge>',
        ])
    lines.extend(["  </graph>", "</graphml>"])

    Path(path).write_text("\n".join(lines), encoding="utf-8")
