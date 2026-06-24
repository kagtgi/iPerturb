"""Recover the grey TF cloud from the original scatter by rasterizing and
labelling the grey marker pixels (the collection offsets are not exposed as
vectors). Highlighted TFs are pinned to their exact published counts.
Writes figs/fig2_points.csv  (k562,rpe1,cls,label)."""
import fitz, numpy as np, csv
from scipy import ndimage

SRC = "figs/original/tf_k562_rpe1_scatter_colored.pdf"
OUT = "figs/fig2_points.csv"
Z = 8.0
doc = fitz.open(SRC); pg = doc[0]

# calibration from tick labels (PDF points)
xt, yt = {}, {}
for b in pg.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        for s in l["spans"]:
            t = s["text"].strip(); x0,y0,x1,y1 = s["bbox"]; cx,cy=(x0+x1)/2,(y0+y1)/2
            if t.isdigit():
                if cy > 235: xt[int(t)] = cx
                elif cx < 30: yt[int(t)] = cy
def fit(d):
    ks=sorted(d); xs=[d[k] for k in ks]; n=len(ks); mk=sum(ks)/n; mp=sum(xs)/n
    b=sum((k-mk)*(p-mp) for k,p in zip(ks,xs))/sum((k-mk)**2 for k in ks); return mp-b*mk, b
ax,bx = fit(xt); ay,by = fit(yt)

# rasterize
pix = pg.get_pixmap(matrix=fitz.Matrix(Z,Z), alpha=False)
arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(int)
R,G,Bc = arr[...,0],arr[...,1],arr[...,2]
# grey markers are alpha-blended (rendered ~60..180); grey => R~=G~=B.
# exclude: axes(33)/text(<55), diagonal & gridlines (>=182), red/orange (R!=G!=B).
grey = (np.abs(R-G)<10)&(np.abs(G-Bc)<10)&(R>=55)&(R<=182)
struct = np.ones((3,3),bool)
grey = ndimage.binary_dilation(grey, struct, iterations=2)   # coalesce each marker's ring
lab,n = ndimage.label(grey)
print("grey components after dilation:", n)
cents = ndimage.center_of_mass(grey, lab, range(1,n+1))
sizes = ndimage.sum(grey, lab, range(1,n+1))
rows=[]
for (ry,rx),sz in zip(cents,sizes):
    if sz < 60: continue                      # drop fringe specks (dilated marker >= ~60px)
    dx = (rx/Z - ax)/bx; dy = (ry/Z - ay)/by
    if not (-6 <= dx <= 520 and -6 <= dy <= 870): continue
    if dx < 60 and dy > 780: continue         # drop legend swatches (top-left corner)
    rows.append((max(0,round(dx)), max(0,round(dy))))
print("cloud markers kept:", len(rows))
print("K562 %d..%d  RPE1 %d..%d" % (min(r[0] for r in rows),max(r[0] for r in rows),
                                    min(r[1] for r in rows),max(r[1] for r in rows)))

# exact highlighted TFs (ground truth = Table 1)
HL = [("TRIM24",476,0),("ATF4",100,833),("ZEB2",331,4),("REST",107,8),("REL",36,110)]
hlset = {(k,r) for _,k,r in HL}
with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["k562","rpe1","cls","label"])
    for k,r in rows:
        if (k,r) in hlset: continue           # avoid double-plot
        w.writerow([k,r,"cloud",""])
    for name,k,r in HL:
        w.writerow([k,r,"highlight",name])
print("wrote", OUT)
