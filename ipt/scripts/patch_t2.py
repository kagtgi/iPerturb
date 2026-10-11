from pathlib import Path
p = Path(__file__).parent / "make_tables2.py"
s = p.read_text(encoding="utf8")
s = s.replace('glob.glob(str(S / f"g12cfg_{line}_f*.tsv"))', 'glob.glob(str(S / f"g12cfg_{line}_p*_f*.tsv"))')
p.write_text(s, encoding="utf8")
