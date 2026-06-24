"""Reconstruct the TF target-count scatter data from the original vector PDF.
Splits each PathCollection into per-marker subpaths, takes centroids, and maps
PDF points -> data via the axis tick labels. Writes figs/fig2_points.csv."""
import fitz, csv, statistics as st

SRC = "figs/original/tf_k562_rpe1_scatter_colored.pdf"
OUT = "figs/fig2_points.csv"

doc = fitz.open(SRC); pg = doc[0]

# ---- calibration from tick labels ----
xt, yt = {}, {}
for b in pg.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        for s in l["spans"]:
            t = s["text"].strip()
            x0, y0, x1, y1 = s["bbox"]; cx, cy = (x0+x1)/2, (y0+y1)/2
            if t.isdigit():
                # x ticks sit on the bottom axis (cy ~ 240), y ticks on left (cx ~ 22-25)
                if cy > 235:        xt[int(t)] = cx
                elif cx < 30:       yt[int(t)] = cy
def fit(d):
    ks = sorted(d); xs = [d[k] for k in ks]
    n = len(ks); mx = sum(ks)/n; mp = sum(xs)/n
    b = sum((k-mx)*(p-mp) for k,p in zip(ks,xs))/sum((k-mx)**2 for k in ks)
    a = mp - b*mx
    return a, b  # pixel = a + b*data  ->  data = (pixel-a)/b
ax, bx = fit(xt); ay, by = fit(yt)
print("x ticks", xt, "-> px = %.3f + %.5f*data" % (ax, bx))
print("y ticks", yt, "-> px = %.3f + %.5f*data" % (ay, by))
def to_data(cx, cy): return (cx-ax)/bx, (cy-ay)/by

# ---- split drawings into per-marker subpaths ----
def subpaths(items):
    cur, last = [], None
    for it in items:
        pts = [p for p in it[1:] if isinstance(p, fitz.Point)]
        if it[0] == "re": r = it[1]; pts = [r.tl, r.br]
        if not pts: continue
        if last is not None and abs(pts[0].x-last.x)+abs(pts[0].y-last.y) > 0.6:
            if cur: yield cur
            cur = []
        cur += pts; last = pts[-1]
    if cur: yield cur

CLS = {(0.91,0.098,0.173):"exclusive", (0.91,0.467,0.133):"divergent",
       (0.369,0.369,0.369):"cloud", (0.0,0.0,0.0):"black"}
rows = []
for dr in pg.get_drawings():
    fill = dr.get("fill")
    if fill is None: continue
    key = tuple(round(c,3) for c in fill)
    cls = CLS.get(key)
    if cls is None: continue
    for sp in subpaths(dr["items"]):
        xs = [p.x for p in sp]; ys = [p.y for p in sp]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        w, h = max(xs)-min(xs), max(ys)-min(ys)
        if w > 15 or h > 15: continue   # skip non-marker subpaths
        dx, dy = to_data(cx, cy)
        rows.append((round(dx,1), round(dy,1), cls))

from collections import Counter
print("recovered markers by class:", Counter(r[2] for r in rows))
print("total markers:", len(rows))
xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
print("K562 range %.0f..%.0f  RPE1 range %.0f..%.0f" % (min(xs),max(xs),min(ys),max(ys)))
print("highlighted (non-cloud):")
for r in rows:
    if r[2] in ("exclusive","divergent"): print("  ", r)
with open(OUT,"w",newline="") as f:
    w = csv.writer(f); w.writerow(["k562","rpe1","cls"]); w.writerows(rows)
print("wrote", OUT)
