"""Patch gears_run.py: also predict a sample of TRAINING targets so GEARS can be centred on its own implied training mean."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "gears_run.py"
s = p.read_text(encoding="utf8")
old = '    np.savez_compressed(OUT / f"pred_fold{fo}.npz", targets=np.array(tests), delta=P, genes=np.array(gene_list))'
new = (
    '    rng2 = np.random.default_rng(a.seed + fo)\n'
    '    tr_all = [g[:-5] for g in sd["train"]]\n'
    '    tr_s = [str(g) for g in rng2.choice(tr_all, min(300, len(tr_all)), replace=False)]\n'
    '    predtr = model.predict([[g] for g in tr_s])\n'
    '    Ptr = np.stack([np.asarray(predtr[g]) - ctrl_mean for g in tr_s]).astype(np.float32)\n'
    '    np.savez_compressed(OUT / f"pred_fold{fo}.npz", targets=np.array(tests), delta=P, genes=np.array(gene_list),\n'
    '                        train_targets=np.array(tr_s), train_delta=Ptr, epochs=a.epochs, cells=a.cells)'
)
assert old in s
p.write_text(s.replace(old, new), encoding="utf8")
print("patched")
