"""Reconstruct Figure 3 data from the original panel PDFs.
- panel c (Pearson): exact text grid.
- panels a,b (dots): colored markers -> means; vertical whiskers -> SD.
- panel d (centroid): tall colored bars -> means; whiskers -> SD.
Methods are assigned left-to-right per panel (CPA,GEARS,scGPT,STATE,Cell2Sentence,
iPerturb), which the x-axis labels confirm. Writes figs/fig3_values.csv."""
import fitz, re, csv, numpy as np

METHODS = ["CPA", "GEARS", "scGPT", "STATE", "Cell2Sentence", "iPerturb"]
NUM = re.compile(r"^\d\.\d+$")

def ycal(pg):
    pts = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                t = s["text"].strip(); x0,y0,x1,y1 = s["bbox"]
                if NUM.match(t) and (x0+x1)/2 < 45:          # left-axis numerals
                    pts.append((float(t), (y0+y1)/2))
    v = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    b = np.polyfit(y, v, 1)                                  # value = b0*py + b1
    return lambda py: float(np.polyval(b, py))

def is_black(c): return c is not None and max(c) < 0.18
def is_white(c): return c is not None and min(c) > 0.9
def split_gap(xs):
    """Return boundary x = midpoint of the largest gap between sorted x's."""
    s = sorted(xs); gaps = [(s[i+1]-s[i], (s[i+1]+s[i])/2) for i in range(len(s)-1)]
    return max(gaps)[1] if gaps else 0
def vsegs(pg):
    segs = []
    for dr in pg.get_drawings():
        for it in dr["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.x-p2.x) < 1.0 and abs(p1.y-p2.y) > 5:
                    segs.append(((p1.x+p2.x)/2, min(p1.y,p2.y), max(p1.y,p2.y),
                                 tuple(dr.get("color") or (0,0,0))))
    return segs

def _emit(pts, val, segs, metric, seg_tol):
    """pts: list of (x, y_for_value). Split into 2 panels by largest x-gap,
    assign methods left-to-right, attach SD from nearest vertical segment."""
    bnd = split_gap([p[0] for p in pts]); out = []
    for cell, sel in [("K562", lambda x: x < bnd), ("RPE1", lambda x: x >= bnd)]:
        grp = sorted([p for p in pts if sel(p[0])])
        if len(grp) != 6:
            print(f"  WARN {metric} {cell}: found {len(grp)} (expected 6)")
        for meth, (mx, my) in zip(METHODS, grp):
            near = [s for s in segs if abs(s[0]-mx) < seg_tol]
            sd = abs(val(near[0][1]) - val(near[0][2]))/2 if near else 0.0
            out.append([metric, cell, meth, round(val(my), 3), round(sd, 3)])
    return out

def dots(path, metric):
    pg = fitz.open(path)[0]; val = ycal(pg)
    marks = [((r.x0+r.x1)/2, (r.y0+r.y1)/2)
             for dr in pg.get_drawings()
             for r in [dr["rect"]] if dr.get("fill") is not None
             and not is_black(dr["fill"]) and not is_white(dr["fill"])
             and r.width < 14 and r.height < 14]
    segs = [s for s in vsegs(pg) if not is_black(s[3])]
    return _emit(marks, val, segs, metric, seg_tol=3)

def bars(path, metric):
    pg = fitz.open(path)[0]; val = ycal(pg)
    rects = [((r.x0+r.x1)/2, r.y0)
             for dr in pg.get_drawings()
             for r in [dr["rect"]] if dr.get("fill") is not None
             and not is_black(dr["fill"]) and not is_white(dr["fill"])
             and r.height > 15 and 10 < r.width < 60]      # exclude full-width bg rects
    segs = list(vsegs(pg))
    return _emit(rects, val, segs, metric, seg_tol=4)

def heatmap(path):
    pg = fitz.open(path)[0]
    vals = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                t = s["text"].strip(); x0,y0,x1,y1 = s["bbox"]
                if NUM.match(t): vals.append(((x0+x1)/2, (y0+y1)/2, float(t)))
    cols = sorted(set(round(v[0]) for v in vals))
    colmap = [("K562","d20"),("K562","d"),("RPE1","d20"),("RPE1","d")]
    rows = sorted(set(round(v[1]) for v in vals))
    out = []
    for ri, ry in enumerate(rows):
        for ci, cx in enumerate(cols):
            cell = [v for v in vals if abs(v[1]-ry)<3 and abs(v[0]-cx)<3]
            if cell:
                c, m = colmap[ci]
                out.append([f"pearson_{m}", c, METHODS[ri], cell[0][2], 0.0])
    return out

rows = []
rows += dots("figs/original/Dir_improvement_dots.pdf", "directional")
rows += dots("figs/original/MSE_improvement_dots.pdf", "mse")
rows += bars("figs/original/centroid_accuracy_bars.pdf", "centroid")
rows += heatmap("figs/original/pearson_delta_heatmap.pdf")

with open("figs/fig3_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric","cell","method","mean","sd"]); w.writerows(rows)

# validation print
def show(metric):
    print(f"\n{metric}:")
    for r in rows:
        if r[0] == metric:
            print(f"  {r[1]:5} {r[2]:14} mean={r[3]:.3f} sd={r[4]:.3f}")
for m in ["directional","mse","centroid"]: show(m)
print("\nwrote figs/fig3_values.csv  (%d rows)" % len(rows))
