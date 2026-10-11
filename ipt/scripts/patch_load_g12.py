"""viz/load.py: include the propagation-kernel results (g12) in the unique-target score tables; add rows to the main table and Fig. 3."""
from pathlib import Path

root = Path(__file__).parents[1]
p = root / "ipt" / "viz" / "load.py"
s = p.read_text(encoding="utf8")
if "g12_" not in s:
    old = '    df = pd.concat(parts)\n'
    new = ('    g12p = S / f"g12_{line}.tsv"\n'
           '    if g12p.exists():\n'
           '        g12 = pd.read_csv(g12p, sep="\\t")\n'
           '        g12 = g12[g12.model.str.startswith("prop[L1+L3]") | (g12.model == "prop[integrated]|nnls")].copy()\n'
           '        g12["mid"] = g12.model\n'
           '        parts.append(g12[["fold", "target", "mid"] + METRICS])\n'
           '    df = pd.concat(parts)\n')
    assert old in s
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf8")

t = root / "scripts" / "make_tables.py"
u = t.read_text(encoding="utf8")
if "prop[L1+L3]|rwr" not in u:
    old = '        ("B5_gcn|real|self", "iPerturb: GCN head (B5)"), ("nbrmean_L3|real|self", "iPerturb: neighbour-mean head")]'
    new = ('        ("B5_gcn|real|self", "iPerturb: GCN head (B5)"), ("nbrmean_L3|real|self", "iPerturb: neighbour-mean head"),\n'
           '        ("prop[L1+L3]|rwr", "iPerturb: multi-hop propagation (RWR)"), ("prop[integrated]|nnls", "iPerturb: multi-tier integrated propagation")]')
    assert old in u
    t.write_text(u.replace(old, new, 1), encoding="utf8")

f3 = root / "scripts" / "fig3_generalization.py"
v = f3.read_text(encoding="utf8")
if "prop[L1+L3]|rwr" not in v:
    old = '("B5_gcn|real|self", "iPerturb: GCN head (B5)", GGREEN), ("nbrmean_L3|real|self", "iPerturb: neighbour-mean head", GYELLOW),'
    new = ('("B5_gcn|real|self", "iPerturb: GCN head (B5)", GGREEN), ("nbrmean_L3|real|self", "iPerturb: neighbour-mean head", GYELLOW),\n'
           '    ("prop[L1+L3]|rwr", "iPerturb: multi-hop RWR", GPURPLE), ("prop[integrated]|nnls", "iPerturb: multi-tier integrated", GPURPLE),')
    assert old in v
    f3.write_text(v.replace(old, new, 1), encoding="utf8")
print("ok")
