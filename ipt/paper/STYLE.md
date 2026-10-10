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
