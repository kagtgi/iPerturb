"""Fix the aggregate2 call in prepare2.run (pass the QC keep mask)."""
from pathlib import Path

p = Path(__file__).parents[1] / "ipt" / "data" / "prepare2.py"
s = p.read_text(encoding="utf8")
old = "A = aggregate2(gen, pc, bc, len(perts), len(batches), G, ci)"
new = "A = aggregate2(gen, pc, bc, keep, len(perts), len(batches), G, ci)"
assert old in s
p.write_text(s.replace(old, new), encoding="utf8")
print("ok")
