"""B9: released scGPT whole-human checkpoint, evaluated AS-IS (no fine-tuning, no training of any kind). Runs on the VM (scgpt-env).

NOT a held-out evaluation: user-directed eval-only row, 'released checkpoint, evaluated as-is (training data overlap)'.

What 'as-is' can mean: the official perturbation route (Tutorial_Perturbation, TransformerGenerator) only loads encoder/value_encoder/transformer_encoder from the
checkpoint and trains the pert_encoder and decoder from scratch, so without fine-tuning its output is a randomly initialised decoder and is meaningless.
The only zero-shot use of the released weights that carries information is therefore in-silico knockout with the pretrained TransformerModel
(full checkpoint incl. its expression decoder): take control cells, set the target gene's binned expression value to 0, and read the change of the model's
predicted (mlm_output) expression of the other genes relative to the unperturbed pass; delta = mean over control cells.
Settings: 100 control cells (GEARS-processed log1p CP10K, binned into 51 bins per cell as in pretraining), sequence = <cls> + the 1199 in-vocabulary genes with the
highest mean control expression (+ the target gene if it is not among them), max_seq_len 1200 as in pretraining. Predictions exist only for those 1199 genes (all
other genes: delta 0 in the scorer). Targets not in the scGPT vocabulary, or with mean control expression <= 0.1 (KO undefined), are excluded and counted.
"""
import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained

ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--cells", type=int, default=100)
ap.add_argument("--nseq", type=int, default=1199)
ap.add_argument("--min-expr", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()
H = Path.home() / "gdata"
CK = Path.home() / "scgpt_ckpt" / "whole-human"
OUT = Path(a.out or Path.home() / "ext" / "scgpt") / a.line
OUT.mkdir(parents=True, exist_ok=True)
folds = json.load(open(H / f"folds_{a.line}.json"))
dev = torch.device("cuda")

x = ad.read_h5ad(H / "pertdata" / a.line.lower() / "perturb_processed.h5ad", backed="r")
genes = np.asarray(x.var["gene_name"].astype(str))
cond = x.obs["condition"].astype(str).to_numpy()
rng = np.random.default_rng(a.seed)
ci = np.sort(rng.choice(np.where(cond == "ctrl")[0], a.cells, replace=False))
Xc = x.X[ci]
Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
Xc = Xc.astype(np.float32)

vocab = GeneVocab.from_file(CK / "vocab.json")
for s in ["<pad>", "<cls>", "<eoc>"]:
    if s not in vocab:
        vocab.append_token(s)
cfg = json.load(open(CK / "args.json"))
invocab = np.array([g in vocab for g in genes])
print(f"{a.line}: genes in scGPT vocab {int(invocab.sum())}/{len(genes)} (dropped {int((~invocab).sum())})", flush=True)
mean_expr = Xc.mean(0)
cand = np.where(invocab)[0]
base = cand[np.argsort(-mean_expr[cand])[:a.nseq]]
gid = np.array([vocab[g] for g in genes])
gpos = {g: i for i, g in enumerate(genes)}

# binned values per control cell over all in-vocab genes (as pretraining: bin the cell's non-zero values)
B = np.zeros_like(Xc)
for i in range(Xc.shape[0]):
    row = np.zeros(Xc.shape[1], np.float32)
    row[invocab] = Xc[i, invocab]
    B[i] = binning(row, cfg["n_bins"])

model = TransformerModel(len(vocab), cfg["embsize"], cfg["nheads"], cfg["d_hid"], cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab,
                         dropout=0.0, pad_token="<pad>", pad_value=cfg["pad_value"], do_mvc=cfg.get("MVC", False), do_dab=False, use_batch_labels=False,
                         input_emb_style=cfg["input_emb_style"], n_input_bins=cfg["n_bins"], use_fast_transformer=False, pre_norm=False)
sd = torch.load(CK / "best_model.pt", map_location="cpu")
load_pretrained(model, sd, strict=False, verbose=False)
miss = [k for k in model.state_dict() if k not in {kk.replace("Wqkv.", "in_proj_") for kk in sd}]
print("model keys not in checkpoint:", len(miss), miss[:5], flush=True)
model.to(dev).eval()
cls_id = vocab["<cls>"]


@torch.no_grad()
def run(sel, ko_pos):
    """sel: gene indices in the sequence; ko_pos: position in sel to zero (or None). Returns [cells, len(sel)] predicted values."""
    ids = torch.tensor(np.r_[cls_id, gid[sel]], device=dev)[None, :].repeat(B.shape[0], 1)
    vals = torch.tensor(np.concatenate([np.zeros((B.shape[0], 1), np.float32), B[:, sel]], 1), device=dev)
    if ko_pos is not None:
        vals[:, ko_pos + 1] = 0
    mask = torch.zeros_like(vals, dtype=torch.bool)
    out = []
    for s in range(0, vals.shape[0], 50):
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(ids[s:s + 50], vals[s:s + 50], src_key_padding_mask=mask[s:s + 50])["mlm_output"]
        out.append(o[:, 1:].float().cpu())
    return torch.cat(out).numpy()


base_set = {int(g): i for i, g in enumerate(base)}
ctrl_base = run(base, None)
t0 = time.time()
need = sorted({g for f in folds for g in f["test"]} | {g for f in folds for g in f["train"]})
res, excl = {}, {"not_in_vocab": [], "low_expr": [], "not_in_data": []}
for g in need:
    if g not in gpos:
        excl["not_in_data"].append(g)
        continue
    j = gpos[g]
    if not invocab[j]:
        excl["not_in_vocab"].append(g)
        continue
    if mean_expr[j] <= a.min_expr:
        excl["low_expr"].append(g)
        continue
    if j in base_set:
        p = base_set[j]
        res[g] = (run(base, p) - ctrl_base).mean(0)
    else:
        sel = np.r_[base, j]
        res[g] = (run(sel, len(base)) - run(sel, None)).mean(0)[:len(base)]
print(f"computed {len(res)} targets in {time.time() - t0:.0f}s; excluded {({k: len(v) for k, v in excl.items()})}", flush=True)
bg = genes[base]
for fo, f in enumerate(folds):
    te = [g for g in f["test"] if g in res]
    tr = [g for g in f["train"] if g in res]
    r2 = np.random.default_rng(a.seed + fo)
    trs = [str(g) for g in r2.choice(tr, min(300, len(tr)), replace=False)]
    np.savez_compressed(OUT / f"pred_fold{fo}.npz", targets=np.array(te), delta=np.stack([res[g] for g in te]).astype(np.float32), genes=np.array(bg),
                        train_targets=np.array(trs), train_delta=np.stack([res[g] for g in trs]).astype(np.float32), n_test_total=len(f["test"]),
                        n_test_excluded=len(f["test"]) - len(te))
    print(f"fold {fo}: predicted {len(te)} of {len(f['test'])} test targets", flush=True)
json.dump({k: v for k, v in excl.items()}, open(OUT / "excluded.json", "w"))
