from pathlib import Path
p = Path(__file__).parent / "fig_applications.py"
s = p.read_text(encoding="utf8")
s = s.replace('bx.legend(loc="lower right", fontsize=7,', 'bx.legend(loc="upper right", ncol=2, columnspacing=0.8, fontsize=7,')
a = 'label=f"{lab_} (n = {W[lab_][1]} fold{\'s\' if W[lab_][1] > 1 else \'\'})"'
assert a in s
s = s.replace(a, 'label=lab_')
p.write_text(s, encoding="utf8")
