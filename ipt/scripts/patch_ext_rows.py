"""Add the external baselines (biolord, CPA, scGPT, STATE) to the unique-target table and Table 1."""
from pathlib import Path

root = Path(__file__).parents[1]
p = root / "ipt" / "viz" / "load.py"
s = p.read_text(encoding="utf8")
old = '    g12p = S / f"g12_{line}.tsv"\n'
new = '''    for ext in ("biolord", "cpa", "scgpt", "state_zeroshot", "state_fewshot"):
        ep = S / f"{ext}_{line}.tsv"
        if ep.exists():
            parts.append(pd.read_csv(ep, sep="\\t")[["fold", "target", "mid"] + METRICS])
'''
if "state_fewshot" not in s:
    s = s.replace(old, new + old, 1)
    p.write_text(s, encoding="utf8")

p = root / "scripts" / "make_tables.py"
s = p.read_text(encoding="utf8")
old = '        ("B4_gears|-|-", "GEARS (B4)"),'
new = ('        ("B4_gears|-|-", "GEARS (B4)"), ("B7_cpa|-|-", "CPA (B7)"), ("B8_biolord|-|-", "biolord (B8)"),\n'
       '        ("B9_scgpt_asis|-|-", "scGPT, zero-shot knockout$^\\\\dagger$"), ("B10_state_zeroshot|-|-", "STATE, zero-shot$^\\\\dagger$"),\n'
       '        ("B10_state_fewshot|-|-", "STATE, few-shot$^\\\\dagger$"),')
if "B8_biolord" not in s:
    assert old in s
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf8")
print("ok")

s = p.read_text(encoding="utf8")
old = "for mid, _ in ROWS if (d.mid == mid).any()) for m, _ in MET}"
new = "for mid, lab_ in ROWS if (d.mid == mid).any() and 'dagger' not in lab_) for m, _ in MET}"
if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf8")
    print("best patched")
