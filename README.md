# iPerturb

**An interpretable framework for predicting gene-expression changes under genetic perturbations.**

iPerturb builds a *context-specific* gene regulatory network (GRN) from public
databases, fits signed Hill-kinetics parameters to each regulatory edge from
CRISPRi Perturb-seq data, and predicts the transcriptome of unseen perturbations
by multi-hop message passing on the GRN — a signed, nonlinear generalization of
network propagation. Every predicted change is attributable to specific signed,
directed edges, so predictions stay mechanistically interpretable.

This repository contains the end-to-end code and a one-click Colab notebook to
reproduce the results for the two human cell lines studied (K562 and RPE1).

> Code accompanying the paper *"iPerturb: An interpretable framework for
> predicting gene expression changes under perturbations"* (CIBB 2026).
> Khang Ta and Linh Huynh, AI Institute, Fulbright University Vietnam.

---

## Quickstart (Google Colab — recommended)

Open [`notebooks/iPerturb_Colab.ipynb`](notebooks/iPerturb_Colab.ipynb) in Colab
(use a **GPU runtime**). It runs everything from a clean clone:

1. Clone this repo and install dependencies.
2. Download the Replogle Perturb-seq expression data (auto, from figshare).
3. Point it at your **GeneHancer v5.26** files (see below — license-gated).
4. Run the full pipeline for K562 and RPE1.
5. Download a single `iperturb_results.zip` (figures + metrics + GRN tables).

Expected runtime: ~30–60 min on a Colab GPU (most of it is database download +
training for the two cell lines).

## What the pipeline does

For each cell line, [`iperturb.py`](iperturb.py) runs top-to-bottom:

1. **Data** — load the Perturb-seq matrix, normalise, select 2,000 HVGs.
2. **Template GRN** — four evidence tiers:
   - L1 curated signed TF–target edges (TRRUST, OmniPath, CollecTRI)
   - L2 TF–target via GeneHancer enhancers (tissue-filtered)
   - L3 PPI transitive closure (STRING, score ≥ 700)
   - L4 co-expression (COXPRESdb, top-5 partners)
3. **Edge selection** — a parameter budget (rule of ten) + a spanning tree and a
   diversity-penalised greedy fill (Algorithm 1 in the paper).
4. **Model** — a Hill-kinetics graph neural network (`GRNN`) with signed,
   range-constrained edge weights; trained with a five-term composite loss.
5. **Evaluation** — directional accuracy, MSE, Pearson (top-20 and genome-wide),
   and centroid accuracy on held-out perturbations, vs. five baselines.

## Data requirements

| Source | How it is obtained | In repo? |
|---|---|---|
| K562 / RPE1 `.h5ad` (Replogle et al. 2022) | auto-download (figshare `35773219`, `35775606`) | no |
| TRRUST v2, OmniPath, CollecTRI | auto (HTTP / `omnipath`) | no |
| STRING PPI | auto (STRING API) | no |
| COXPRESdb v8.1 | auto (Zenodo) | no |
| **GeneHancer v5.26** | **you provide** (license-gated) | **no** |

**GeneHancer is not redistributable.** Obtain `GeneHancer_v5.26.gff`,
`GeneHancer_TFBSs_v5.26.txt`, and `GeneHancer_Tissues_v5.26.txt` from GeneHancer
/ GeneCards and place them in `/content/` (the Colab notebook can copy them from
your Google Drive). Large raw data files and license-gated databases are
intentionally excluded from version control (see `.gitignore`).

## Local use

`iperturb.py` is a pure-Python script that expects its inputs under `/content/`
(Colab convention) and writes outputs there. To run outside Colab, install
[`requirements.txt`](requirements.txt), create `/content/` (or adjust the paths
near the top of the script), drop the data files in, and run `python iperturb.py`
on a CUDA machine.

```bash
pip install -r requirements.txt
python iperturb.py        # builds GRN, trains and evaluates K562 then RPE1
```

## Figures (`figs/`)

The paper figures are regenerated in a consistent Google/Material style from the
provided metric tables:

```bash
python figs/make_fig2_scatter.py   # Figure 2: TF cell-type-specific wiring scatter
python figs/make_fig3.py           # Figure 3: benchmark panels + Pearson heatmap
```

- `figs/style_google.py` — shared matplotlib style.
- `figs/fig2_points.csv`, `figs/fig3_values.csv` — figure data.
- `figs/ablation_lambda.py`, `figs/jacobian_rowsum.py` — optional analyses
  (loss-term ablation / λ sweep, and the message-passing contraction check).
- `figs/extract_*.py`, `figs/original/` — provenance: how the figure CSVs were
  reconstructed from the original renders (needs `pymupdf`).

## Repository layout

```
iperturb.py                 # end-to-end pipeline (K562 then RPE1)
requirements.txt
notebooks/iPerturb_Colab.ipynb
figs/                       # figure regeneration + analysis scripts and data
gene_list.txt               # example HVG list (K562)
LICENSE                     # MIT
```

## Citation

If you use this code, please cite the iPerturb paper (CIBB 2026). A BibTeX entry
will be added here on publication.

## License

[MIT](LICENSE) © 2026 Khang Ta, Linh Huynh — AI Institute, Fulbright University Vietnam.
