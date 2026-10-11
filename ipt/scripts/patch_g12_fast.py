"""g12_prop.py: reduced grid (--fast) so the study finishes; extra tier graphs via --extra name=path."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "g12_prop.py"
s = p.read_text(encoding="utf8")
if "--fast" not in s:
    s = s.replace('ap.add_argument("--extra"', 'ap.add_argument("--fast", action="store_true")\nap.add_argument("--extra"', 1)
    old = "GAMMAS, LAMS = (0.0, 0.5, 1.0), (0.25, 0.5, 0.75, 1.0)"
    new = ("GAMMAS, LAMS = (0.0, 0.5, 1.0), (0.25, 0.5, 0.75, 1.0)\n"
           "if a.fast:\n"
           "    KERNELS = [('onehop', {})] + [('rwr', dict(alpha=al, norm='rw')) for al in (0.3, 0.7)] + [('trunc', dict(steps=k, norm='rw')) for k in (2, 3)]\n"
           "    SINGLE = [('onehop', {}), ('rwr', dict(alpha=0.5, norm='rw'))]\n"
           "    GAMMAS, LAMS = (0.0, 0.5), (0.5, 1.0)\n"
           "else:\n"
           "    SINGLE = KERNELS")
    assert old in s
    s = s.replace(old, new, 1)
    old2 = "        bf = run_graph(gname, tiers[gname])"
    new2 = "        bf = run_graph(gname, tiers[gname], KERNELS if gname == 'L1+L3' else SINGLE)"
    assert old2 in s
    s = s.replace(old2, new2, 1)
    p.write_text(s, encoding="utf8")
print("ok")
