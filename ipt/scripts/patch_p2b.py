"""prepare2: controls have nperts == 0; keep cells with nperts <= 1 (single-target or control)."""
from pathlib import Path

root = Path(__file__).parents[1]
p = root / "ipt" / "data" / "prepare2.py"
s = p.read_text(encoding="utf8")
old = "k4 = np.ones(len(umi), bool) if nperts is None else (np.asarray(nperts) == 1)"
new = "k4 = np.ones(len(umi), bool) if nperts is None else (np.asarray(nperts) <= 1)"
assert old in s
p.write_text(s.replace(old, new), encoding="utf8")
t = root / "tests" / "test_prepare2.py"
u = t.read_text(encoding="utf8")
u = u.replace("nperts=np.ones(400))", "nperts=np.r_[np.zeros(100), np.ones(290), 2 * np.ones(10)])")
u = u.replace("assert keep.sum() == qc[\"kept\"] and qc[\"fail_umi\"] >= 5", "assert keep.sum() == qc[\"kept\"] and qc[\"fail_umi\"] >= 5 and qc[\"fail_multiguide\"] == 10 and keep[5:100].all()")
t.write_text(u, encoding="utf8")
print("ok")
