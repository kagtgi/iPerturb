"""Split results.tex into results.tex + ablation.tex (CIBB layout)."""
from pathlib import Path

root = Path(__file__).parents[1] / "paper" / "sections"
L = (root / "results.tex").read_text(encoding="utf8").split("\n")
g = lambda a, b: "\n".join(L[a - 1:b])  # 1-indexed inclusive

prep = L[9]
cut = prep.index("Weighting training")
prep_keep, prep_weight = prep[:cut].rstrip(), prep[cut:]

res = "\n".join([
    g(1, 7),
    "",
    prep_keep + " Weighting training perturbations and the other design choices that did not matter are examined in Section~\\ref{sec:ablation}.",
    "",
    g(12, 44),
    "",
    g(46, 59).replace("\\subsection{The Hill-form operator is faithful to exact Hill propagation and efficient as a matrix product}",
                      "\\subsection{The learnable Hill propagation is faithful to the exact Hill form and efficient as a matrix product}")
    + "\n\\paragraph{Predictive effect.} Whether the Hill form improves accuracy over a linear transfer is tested in Section~\\ref{sec:ablation_hill}; it does not, so the operator is reported as a faithful and efficient implementation of the regulatory model rather than as a source of predictive gain.",
    "",
    g(73, 97),
    "",
])
(root / "results.tex").write_text(res, encoding="utf8")

abl = "\n".join([
    "\\section{Ablation Study}",
    "\\label{sec:ablation}",
    "We ablate the two components of iPerturb, the context-specific GRN (evidence layers, cell-line tier, further databases, propagation kernels) and the learnable Hill propagation (transfer function, certified variant, loss), and the preprocessing. All differences are paired over unique held-out targets with 95\\% bootstrap intervals (Table~\\ref{tab:paired}).",
    "",
    "\\subsection{Evidence layers and controls}",
    "\\label{sec:ablation_layers}",
    "Figure~\\ref{fig:ablation}b summarises the layers of the GRN and the controls of Section~\\ref{sec:results_pred}: STRING alone, the curated tier plus STRING, a degree-preserving rewired version, a graph learned from the training data, and the real graph with own-target entries zeroed. Only the real association evidence changes $\\rho_c$ relative to no graph; the curated tier adds nothing beyond STRING, and controls with the same capacity and tuning grid gain nothing.",
    "",
    "\\subsection{Cell-line-specific tier}",
    "\\label{sec:ablation_context}",
    g(8, 8).replace("\\par\\medskip", "").replace("\\datanumif{g7_d_k_hep}{", "\\datanumif{g7_d_k_hep}{", 1),
    "",
    "\\subsection{Hill versus linear propagation}",
    "\\label{sec:ablation_hill}",
    g(60, 60).replace("\\paragraph{Predictive effect.} ", ""),
    "",
    g(62, 71),
    "",
    "\\subsection{Preprocessing}",
    "\\label{sec:ablation_prep}",
    prep_weight,
    "",
])
(root / "ablation.tex").write_text(abl, encoding="utf8")
print("ok", len(res), len(abl))
