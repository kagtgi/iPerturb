# iPerturb house style (distilled from CIBB paper, lnbi/STATUS.md, figs/style_google.py)

## Philosophy
- The GRN is the computational architecture, not a post-hoc explanation (biologically informed NN).
- Gap framing: mechanistic kinetic models do not scale; deep models scale but are opaque; we test the middle.
- "Interpretable" is defined mechanistically: each prediction decomposes into signed edges / paths. Never overclaim causality.
- Honest reporting: name the metric where we lose; keep negative controls in the main text or appendix.
- Mean-response target (pseudobulk delta), single-cell source. Say this in abstract/methods/captions.

## Abstract / prose
- Order: problem -> gap -> method (First, Then, Finally) -> evaluation -> contribution/scope. ~180-220 words, no invented numbers.
- Intro in 3 short paragraphs: motivation + baselines, "we introduce iPerturb" with a three-step walk-through of Fig. 1, what we show.
- No "However"; intuitive flow (virtual cell -> topology -> propagation -> specificity -> controls). Dense but plain sentences.
- Results subsections titled as findings ("iPerturb identifies ...").
- Table captions state protocol; reduced-budget / different-protocol numbers are labelled and never merged.

## Figures
- Palette (figs/style_google.py): proposed=#4285F4, baseline/competitor=#EA4335, caution=#FBBC04, good=#34A853, extra=#5E35B1, reference grey=#5F6368; light tints for dense fills.
- Google Sans Flex (fallback Roboto/DejaVu), 8 pt body, 7.5 pt ticks, mono numeric ticks, no top/right spines, light grid, white bg.
- Export PDF (fonttype 42) + PNG 600 dpi (+SVG); LNBI figures via lnbi/figure_tools figkit (180 mm canvas, gate.py QC).
- Caption: bold one-sentence claim, then protocol, then (a)(b)(c) panel list. Every figure carries provenance JSON.
- STATE/scGPT/GEARS red-family, iPerturb blue; never draw STRING/coexpression as causal arrows.

## Addendum from the CIBB short paper (8398 final PDF)
Voice/structure to keep:
- Title = "iPerturb: <what it is> for <task>". Keywords line under the title. Abstract = ONE paragraph, ~150 words: grand-challenge opener -> "Mechanistic X ... whereas Y ..." gap -> "To bridge this gap, we propose iPerturb" -> first/then/finally pipeline -> evaluation (N perturbations, two cell lines) -> results sentence naming the metrics -> scope sentence ("while keeping every prediction interpretable at the level of ...").
- Intro: 2 paragraphs + one scope paragraph. Para 1: central challenge; needs predictive + scalable + mechanistically interpretable; kinetic models don't scale, deep models (GEARS, scGPT, STATE) opaque, biNNs predict phenotypes. Para 2: "In this paper, we introduce ..." + three-step walk through Fig. 1. Para 3: "We applied ... Our results show that ... (i) ... (ii) ..." and "Here, *interpretable* is meant mechanistically: ...".
- Methods: 2.1 Datasets (counts with exact numbers per source, Systema protocol, 100 cells/perturbation), 2.2 Template GRN (five tiers, Algorithm box, conflict rule: lowest-numbered tier wins, within-tier sign conflicts -> unsigned), 2.3 modelling + loss, 2.4 Evaluating.
- Results subsections are findings ("iPerturb identifies ...", "iPerturb correctly predicts ..."); each opens with the headline comparison, gives exact numbers with the direction of change (e.g. "+20.5%", "reduced by 58.3%"), names the one metric where we lose.
- Table/figure captions: bold lead sentence = the claim; then protocol (seeds, baselines at published defaults); reduced-budget protocols explicitly labelled "not the headline result".
- Discussion: ONE short paragraph restating the thesis ("makes the GRN the literal computational architecture"), the losing metric as a trade-off, one sentence of future work. Data & Software Availability names every source + versions + GitHub URL.
- Typographic: em-dashes, "(i)/(ii)/(iii)" enumerations, italic *interpretable*, \Delta_{20} notation for top-20 Pearson, Dir./MSE/centroid metrics reported as in Fig. 3 (a)-(d).
Claims policy (overrides the old paper where measurements differ): the CIBB Figure-3 numbers were transcribed from an earlier protocol and are NOT reproduced by the leak-free target-held-out protocol; new claims follow only measured results, with the five-tier evidence hierarchy language kept.
