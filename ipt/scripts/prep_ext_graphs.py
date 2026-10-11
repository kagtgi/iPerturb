"""Convert downloaded tier extensions (artifacts/ext/*) into undirected gene-pair files (a,b) for the propagation experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
EXT = ROOT / "artifacts" / "ext"
OUT = EXT / "graphs"
OUT.mkdir(exist_ok=True)


def save(name, df):
    df = df.dropna()
    df = df[df.a != df.b]
    df = df.assign(a=df.a.astype(str).str.upper(), b=df.b.astype(str).str.upper())
    lo = df[["a", "b"]].min(axis=1)
    hi = df[["a", "b"]].max(axis=1)
    out = pd.DataFrame({"a": lo, "b": hi}).drop_duplicates()
    out.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
    print(f"{name}: {len(out):,} undirected pairs")


# L1 extension: SIGNOR (signed; stored unsigned for the association channel)
p = EXT / "L1" / "signor.tsv"
if p.exists():
    d = pd.read_csv(p, sep="\t")
    save("signor", d.rename(columns={"source": "a", "target": "b"})[["a", "b"]])

# L2 extension: ChIP-Atlas K562 / RPE1: for each target gene keep its 20 highest-scoring factors
for ctx in ("K562", "RPE1"):
    p = EXT / "L2" / f"chipatlas_{ctx}.tsv"
    if p.exists():
        d = pd.read_csv(p, sep="\t", usecols=["source", "target", "score"])
        d = d.sort_values("score", ascending=False).groupby("target").head(20)
        save(f"chipatlas_{ctx}", d.rename(columns={"source": "a", "target": "b"})[["a", "b"]])

# L3 extensions: CORUM complexes, BioGRID physical, IntAct
for name, f in (("corum", "corum_pairs.tsv"), ("biogrid", "biogrid_physical.tsv"), ("intact", "intact_human.tsv")):
    p = EXT / "L3" / f
    if p.exists():
        d = pd.read_csv(p, sep="\t")
        cols = [c for c in d.columns if c in ("a", "b")]
        if len(cols) == 2:
            save(name, d[["a", "b"]])

# L4 extension: GTEx-derived top-50 co-expression
p = EXT / "L4" / "archs4_top50.tsv"
if p.exists():
    d = pd.read_csv(p, sep="\t")
    save("gtex_coex", d[["a", "b"]])
