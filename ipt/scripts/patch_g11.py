"""g11_loss.py: add --losses (subset) and --base (l1l3 | l3 | none) so the same code evaluates tier extensions."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "g11_loss.py"
s = p.read_text(encoding="utf8")
if "--losses" not in s:
    s = s.replace('ap.add_argument("--tag", default="")', 'ap.add_argument("--tag", default="")\nap.add_argument("--losses", default="mse_closed,mse,corr,nmse,huber,corr+huber")\nap.add_argument("--base", default="l1l3")\nap.add_argument("--alphas", default="0.5,0.75,0.9,1.0")', 1)
    s = s.replace('KS, ALPHAS = (20, 40), (0.5, 0.75, 0.9, 1.0)', 'KS = (20, 40)\nALPHAS = tuple(float(x) for x in a.alphas.split(","))', 1)
    s = s.replace('parts = [pairs(l1.source, l1.target), pairs(st.a.str.upper(), st.b.str.upper())]',
                  'parts = []\nif a.base in ("l1l3",):\n    parts.append(pairs(l1.source, l1.target))\nif a.base in ("l1l3", "l3"):\n    parts.append(pairs(st.a.str.upper(), st.b.str.upper()))', 1)
    s = s.replace('for loss_name in ("mse_closed", "mse", "corr", "nmse", "huber", "corr+huber"):', 'for loss_name in a.losses.split(","):', 1)
    s = s.replace('E = np.unique(np.sort(np.concatenate(parts), 1), axis=0)', 'E = np.unique(np.sort(np.concatenate(parts), 1), axis=0) if parts else np.zeros((0, 2), int)', 1)
    p.write_text(s, encoding="utf8")
print("patched")
