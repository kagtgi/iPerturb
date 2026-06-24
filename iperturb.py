# -*- coding: utf-8 -*-
"""iPerturb -- interpretable prediction of gene-expression responses to genetic
perturbations via a Hill-kinetics graph neural network over a context-specific
gene regulatory network (GRN).

End-to-end pipeline for both cell lines (K562, then RPE1): build the template GRN
from public databases (TRRUST, OmniPath/CollecTRI, GeneHancer, STRING, COXPRESdb),
select edges under a parameter budget, fit the GRNN, and evaluate on held-out
perturbations.

Designed to run in Google Colab (working paths are under /content/). The
prerequisites below are all handled by notebooks/iPerturb_Colab.ipynb:
  * Python deps      -- see requirements.txt
  * Expression data  -- /content/K562.h5ad, /content/RPE1.h5ad
                        (Replogle et al. 2022; figshare files 35773219, 35775606)
  * GeneHancer v5.26 (license-gated, user-provided):
        /content/GeneHancer_v5.26.gff
        /content/GeneHancer_TFBSs_v5.26.txt
        /content/GeneHancer_Tissues_v5.26.txt
Outputs (figures, metrics TSVs, GRN edge tables) are written under /content/.
"""

import scanpy as sc

DATA_PATH = "/content/K562.h5ad"
adata = sc.read_h5ad(DATA_PATH)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

hvg_list = adata.var_names[adata.var["highly_variable"]].tolist()

print(f"Number of HVGs selected: {len(hvg_list)}")
print(hvg_list)

# ── Cell & KD counts ──────────────────────────────────────────
print("=== Dataset Summary ===")
print(f"Total cells       : {adata.n_obs:,}")
print(f"Total genes       : {adata.n_vars:,}")

# KD labels — check common obs column names
kd_col_candidates = ["perturbation", "gene", "condition", "KD", "knockdown",
                     "target", "gene_name", "perturbation_name"]

kd_col = next((c for c in kd_col_candidates if c in adata.obs.columns), None)

if kd_col:
    kd_counts = adata.obs[kd_col].value_counts()
    print(f"\nKD column found   : '{kd_col}'")
    print(f"Unique KDs        : {kd_counts.nunique():,}")
    print(f"\nPer-KD cell counts:")
    print(kd_counts.to_string())
else:
    # Fallback: show all obs columns so you can identify the right one
    print("\nNo KD column auto-detected. Available obs columns:")
    print(list(adata.obs.columns))
    print("\nSample of adata.obs:")
    print(adata.obs.head(5).to_string())

################################################################################
# CELL 0b — HVG gene list + cell-line marker check
# Insert after the existing Cell 0 (HVG selection)
################################################################################
import os

SYMBOL_COL = "gene_name"
CELL_LINE  = "K562"          # switch to "RPE1" for the other dataset

hvg_symbols = (adata.var
               .loc[adata.var["highly_variable"], SYMBOL_COL]
               .dropna()
               .tolist())

print(f"HVGs: {len(hvg_symbols)}")

with open("/content/gene_list.txt", "w") as f:
    f.write("\n".join(hvg_symbols))
print("✓ /content/gene_list.txt")

# ── Cell-line marker gene lists ───────────────────────────────
MARKER_GENES = {
    "K562": [
        # --- Original HVG-present markers ---
        "HBZ",      # Hemoglobin zeta
        "GATA1",    # Master erythroid TF
        "KLF1",     # Erythroid-specific TF
        "GFI1B",    # Hematopoietic repressor TF
        "GYPA",     # Erythrocyte membrane marker
        "NFE2",     # Erythroid regulatory factor

        # --- Replacements for missing HVG markers (ABL1, HBB, HBE1, CRKL) ---
        "ALAS2",    # Heme synthesis, strong erythroid marker
        "TFRC",     # CD71, classic K562 / erythroid proliferation marker
        "HBA1",   # Band 3, erythroid membrane marker
        "MYC",     # Alpha hemoglobin stabilizing protein
    ],
    "RPE1": [
        "BEST1",    # RPE ion channel, Best macular dystrophy gene
        "RPE65",    # Retinoid isomerohydrolase, RPE-defining gene
        "RLBP1",    # RPE-specific retinoid carrier (= CRALBP)
        "TFPT",     # Highest text-mining score for hTERT-RPE1
        "PRPF31",   # Splicing factor, retinitis pigmentosa
        "TIMP3",    # RPE-expressed, Sorsby fundus dystrophy
        "SERPINF1", # PEDF — secreted neuroprotective factor
        "TTR",      # Transthyretin, top-30 RPE expressed
        "CST3",     # Cystatin C, highly expressed in RPE
        "TYR",      # Tyrosinase, melanin synthesis in RPE
    ],
}

hvg_set    = set(hvg_symbols)
candidates = MARKER_GENES[CELL_LINE]

print(f"\n=== {CELL_LINE} marker genes in HVG list ===")
found, missing = [], []
for gene in candidates:
    if gene in hvg_set:
        found.append(gene)
        print(f"  ✓  {gene}")
    else:
        print(f"  ✗  {gene}  ← not in HVG set")
        missing.append(gene)

print(f"\nFound : {len(found)} / {len(candidates)}")
print(f"Missing: {missing}")

# Genes to visualise after Cell 10
VIZ_GENES = found   # only genes actually in the network

# ────────────────────────────────────────────────────────────
# CELL 2 — Configuration  ← only edit this cell
# ────────────────────────────────────────────────────────────
import os

# Gene list produced by the HVG cell (one HGNC symbol per line)
GENE_LIST_FILE = "/content/gene_list.txt"

CACHE_DIR  = "/content/grn_cache"
OUT_FILE   = "/content/grn_edges.tsv"

TARGET_EDGES     = 4_600
GREEDY_REWARD    = 0.15
STRING_MIN_SCORE = 700   # 400=permissive, 700=high, 900=very high
COEX_TOPN        = 5     # top-N co-expressed partners per gene

# GeneHancer local files (already downloaded to /content/)
GH_GFF_PATH    = "/content/GeneHancer_v5.26.gff"
GH_TFBS_PATH   = "/content/GeneHancer_TFBSs_v5.26.txt"
GH_TISSUE_PATH = "/content/GeneHancer_Tissues_v5.26.txt"

# COXPRESdb token (free at coxpresdb.jp — leave "" to skip auth)
COXPRESDB_TOKEN = ""

# Filter GeneHancer to K562-active enhancers only (recommended)
TISSUE_FILTER = "K562"   # substring match in tissue name; "" = no filter

LEVEL_CONF = {1: 1.0, 2: 0.60, 3: 0.35, 4: 0.20}

# Set to True if COXPRESdb consistently times out from your Colab region
SKIP_COXPRESDB = False

# ────────────────────────────────────────────────────────────
# CELL 3 — Imports & session
# ────────────────────────────────────────────────────────────
# dependencies are installed from requirements.txt (see the Colab notebook)

import logging, time, re, warnings
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress noisy library/network warnings in Colab
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("grn")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (GRN-research)"})
    return s

SESSION = make_session()

def _get(url, params=None, timeout=120):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r

def load_genes() -> set:
    genes = set()
    with open(GENE_LIST_FILE) as f:
        for line in f:
            g = line.strip().split()[0].upper()
            if g:
                genes.add(g)
    log.info("Gene set: %d genes", len(genes))
    return genes

GENE_SET = load_genes()
print(f"✓ {len(GENE_SET)} genes loaded")

# ────────────────────────────────────────────────────────────
# CELL 4 — Level 1a: TRRUST v2
# ────────────────────────────────────────────────────────────
def crawl_trrust(gene_set):
    url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
    log.info("TRRUST → %s", url)
    r = _get(url)
    rows = []
    for line in r.text.strip().split("\n"):
        if line.startswith("#"): continue
        p = line.split("\t")
        if len(p) < 3: continue
        tf, tgt, itype = p[0].upper(), p[1].upper(), p[2]
        if tf not in gene_set or tgt not in gene_set: continue
        sign = 1 if "activation" in itype.lower() else (-1 if "repression" in itype.lower() else 0)
        rows.append({"source": tf, "target": tgt, "sign": sign, "level": 1, "db": "TRRUST"})
    df = pd.DataFrame(rows)
    log.info("  → %d edges", len(df))
    return df

_p = Path(CACHE_DIR)/"l1_trrust.tsv"
L1_TRRUST = pd.read_csv(_p, sep="\t") if _p.exists() else crawl_trrust(GENE_SET)
if not _p.exists(): L1_TRRUST.to_csv(_p, sep="\t", index=False)
print(f"TRRUST: {len(L1_TRRUST)} edges")

# ────────────────────────────────────────────────────────────
# CELL 5 — Level 1b: OmniPath + CollecTRI
# ────────────────────────────────────────────────────────────
def _omnipath_via_lib(gene_set: set, datasets: list[str], label: str) -> pd.DataFrame:
    import omnipath as op
    rows = []
    for ds in datasets:
        try:
            log.info("  omnipath lib: dataset=%s", ds)
            if ds == "collectri":
                df_raw = op.interactions.CollecTRI.get(dorothea_levels=None, genesymbols=True)
            elif ds == "tf_target":
                df_raw = op.interactions.TFtarget.get(genesymbols=True)
            elif ds == "dorothea":
                df_raw = op.interactions.Dorothea.get(dorothea_levels=["A","B"], genesymbols=True)
            else:
                df_raw = op.interactions.AllInteractions.get(datasets=[ds], genesymbols=True)

            for _, row in df_raw.iterrows():
                src = str(row.get("source_genesymbol","")).upper()
                tgt = str(row.get("target_genesymbol","")).upper()
                if src not in gene_set or tgt not in gene_set: continue
                try:
                    sign = int(row.get("consensus_stimulation") or 0) - int(row.get("consensus_inhibition") or 0)
                except: sign = 0
                rows.append({"source": src, "target": tgt, "sign": sign, "level": 1, "db": label})
        except Exception as e:
            log.debug("  omnipath lib %s failed: %s", ds, e)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

def _enrichr_fallback(gene_set: set, libraries: list[str], label: str) -> pd.DataFrame:
    import gseapy as gp
    rows = []
    for lib in libraries:
        try:
            log.info("  Enrichr fallback: %s", lib)
            enr = gp.get_library(lib, organism="Human")
            for tf, targets in enr.items():
                tf = tf.split(" ")[0].upper()
                if tf not in gene_set: continue
                for tgt in targets:
                    tgt = tgt.upper()
                    if tgt in gene_set and tgt != tf:
                        rows.append({"source": tf, "target": tgt, "sign": 0, "level": 1, "db": f"Enrichr:{lib}"})
        except Exception as e:
            log.debug("  Enrichr %s: %s", lib, e)
    df = pd.DataFrame(rows).drop_duplicates(["source","target"]) if rows else \
         pd.DataFrame(columns=["source","target","sign","level","db"])
    return df

def crawl_omnipath_robust(gene_set: set, datasets: list[str], label: str, cache_name: str) -> pd.DataFrame:
    _p = Path(CACHE_DIR) / f"l1_{cache_name}.tsv"
    if _p.exists():
        df = pd.read_csv(_p, sep="\t")
        log.info("[cache] %s: %d edges", label, len(df))
        return df

    df = pd.DataFrame(columns=["source","target","sign","level","db"])
    # 1. Library
    try: df = _omnipath_via_lib(gene_set, datasets, label)
    except Exception as e: log.warning("Lib path failed: %s", e)

    # 2. Enrichr fallback
    if df.empty:
        log.warning("  lib empty → falling back to Enrichr")
        try: df = _enrichr_fallback(gene_set, ["ChEA_2022", "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X", "TF_Perturbations_Followed_by_Expression"], label)
        except Exception as e: log.warning("Enrichr fallback failed: %s", e)

    if not df.empty: df.to_csv(_p, sep="\t", index=False)
    return df

L1_OMNIPATH  = crawl_omnipath_robust(GENE_SET, ["tf_target", "collectri"], "OmniPath",  "omnipath")
L1_COLLECTRI = crawl_omnipath_robust(GENE_SET, ["collectri"], "CollecTRI", "collectri")
print(f"OmniPath: {len(L1_OMNIPATH)} | CollecTRI: {len(L1_COLLECTRI)}")

# ────────────────────────────────────────────────────────────
# CELL 6 — Level 2: GeneHancer (GFF + TFBS + Tissue filter)
# ────────────────────────────────────────────────────────────
import os
os.makedirs(CACHE_DIR, exist_ok=True)
_l2_cache = Path(CACHE_DIR) / "l2_genehancer.tsv"
if _l2_cache.exists(): _l2_cache.unlink()

def parse_gff_gene_associations(gff_path) -> pd.DataFrame:
    records = []
    _skip = re.compile(r'^(ENSG|ENSM|lnc-|piR-|LOC\d|HSALNG|FAM\d)', re.IGNORECASE)
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9: continue
            if parts[2].lower() != "enhancer": continue
            attr_str = parts[8]
            m = re.search(r'genehancer_id=([^;]+)', attr_str)
            if not m: continue
            gh_id = m.group(1).strip()
            for gm in re.finditer(r'connected_gene=([^;]+);score=([\d.]+)', attr_str):
                gene = gm.group(1).strip().upper()
                if not gene or _skip.match(gene): continue
                try: sc = float(gm.group(2))
                except: sc = 0.0
                records.append({"enhancer_id": gh_id, "gene": gene, "score": sc})
    df = pd.DataFrame(records)
    return df

def parse_tfbs(tfbs_path, tissue_filter: str = "") -> pd.DataFrame:
    df = pd.read_csv(tfbs_path, sep="\t", comment="#", dtype=str)
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "ghid": col_map[c] = "ghid"
        elif cl == "tf": col_map[c] = "tf"
        elif "tissue" in cl: col_map[c] = "tissues"
    df = df.rename(columns=col_map)[["ghid", "tf", "tissues"]]
    df["ghid"] = df["ghid"].str.strip(); df["tf"] = df["tf"].str.strip().str.upper()
    df["tissues"] = df["tissues"].fillna("")
    if tissue_filter:
        df = df[df["tissues"].str.contains(tissue_filter, case=False, na=False)]
    return df[["ghid", "tf"]].drop_duplicates()

def parse_tissues(tissue_path, filter_str="") -> set:
    active = set()
    with open(tissue_path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            gh_id, tissue = parts[0].strip(), parts[2].strip()
            if not filter_str or filter_str.lower() in tissue.lower(): active.add(gh_id)
    return active

def build_l2_edges(gene_set, gff_df, tfbs_df, active_enhancers) -> pd.DataFrame:
    if tfbs_df.empty: return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])
    if active_enhancers: gff_df = gff_df[gff_df["enhancer_id"].isin(active_enhancers)]
    tfbs_df = tfbs_df.rename(columns={"ghid": "enhancer_id"})
    merged  = gff_df.merge(tfbs_df, on="enhancer_id", how="inner")
    rows = []
    for _, row in merged.iterrows():
        tf, tgt = row["tf"], row["gene"]
        if tf in gene_set and tgt in gene_set and tf != tgt:
            rows.append({"source": tf, "target": tgt, "sign": 0, "level": 2, "db": "GeneHancer_TFBS"})
    return pd.DataFrame(rows).drop_duplicates(["source", "target"]) if rows else pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

gff_df = parse_gff_gene_associations(GH_GFF_PATH)
tfbs_df = parse_tfbs(GH_TFBS_PATH, tissue_filter=TISSUE_FILTER)
active_enh = parse_tissues(GH_TISSUE_PATH, TISSUE_FILTER)
L2_GH = build_l2_edges(GENE_SET, gff_df, tfbs_df, active_enh)
L2_GH.to_csv(_l2_cache, sep="\t", index=False)
print(f"GeneHancer L2: {len(L2_GH)} edges")

# ────────────────────────────────────────────────────────────
# CELL 7 — Level 3: STRING PPI + transitive closure
# ────────────────────────────────────────────────────────────
def crawl_string_ppi(gene_set, min_score=700):
    url_map = "https://string-db.org/api/json/get_string_ids"
    url_net = "https://string-db.org/api/json/network"
    genes, id_map = sorted(gene_set), {}
    for i in tqdm(range(0, len(genes), 500), desc="STRING id-map"):
        batch = genes[i:i+500]
        try:
            r = SESSION.post(url_map, data={"identifiers": "\r".join(batch), "species": 9606, "limit": 1, "echo_query": 1, "caller_identity": "grn_crawler"}, timeout=60)
            r.raise_for_status()
            for rec in r.json():
                sym, sid = rec.get("queryItem","").upper(), rec.get("stringId","")
                if sym and sid: id_map[sym] = sid
        except Exception as e: log.debug("STRING id_map batch %d: %s", i, e)
        time.sleep(0.5)
    if not id_map: return pd.DataFrame(columns=["source","target","sign","level","db"])
    rev, rows, ids = {v: k for k, v in id_map.items()}, [], list(id_map.values())
    for i in tqdm(range(0, len(ids), 500), desc="STRING network"):
        try:
            r = SESSION.post(url_net, data={"identifiers": "\r".join(ids[i:i+500]), "species": 9606, "required_score": min_score, "caller_identity": "grn_crawler"}, timeout=120)
            r.raise_for_status()
            for itx in r.json():
                a, b = rev.get(itx.get("stringId_A",""),"").upper(), rev.get(itx.get("stringId_B",""),"").upper()
                if a in gene_set and b in gene_set and a != b:
                    rows += [{"source": a, "target": b, "sign": 0, "level": 3, "db": "STRING"},
                             {"source": b, "target": a, "sign": 0, "level": 3, "db": "STRING"}]
        except Exception as e: log.debug("STRING net batch %d: %s", i, e)
        time.sleep(0.5)
    return pd.DataFrame(rows).drop_duplicates(["source","target"]) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

def build_l3_transitive(l2, ppi):
    if l2.empty or ppi.empty: return pd.DataFrame(columns=["source","target","sign","level","db"])
    partners = {}
    for _, row in ppi.iterrows(): partners.setdefault(row["source"], set()).add(row["target"])
    rows = []
    for _, row in l2.iterrows():
        A, B = row["source"], row["target"]
        for C in partners.get(A, set()):
            if C != B: rows.append({"source": C, "target": B, "sign": 0, "level": 3, "db": "L2+PPI"})
    return pd.DataFrame(rows).drop_duplicates(["source","target"]) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

_pp, _pt = Path(CACHE_DIR)/"l3_ppi_raw.tsv", Path(CACHE_DIR)/"l3_transitive.tsv"
PPI_RAW = pd.read_csv(_pp, sep="\t") if _pp.exists() else crawl_string_ppi(GENE_SET, STRING_MIN_SCORE)
if not _pp.exists(): PPI_RAW.to_csv(_pp, sep="\t", index=False)
L3 = pd.read_csv(_pt, sep="\t") if _pt.exists() else build_l3_transitive(L2_GH, PPI_RAW)
if not _pt.exists(): L3.to_csv(_pt, sep="\t", index=False)
print(f"STRING PPI: {len(PPI_RAW)//2} undirected pairs  |  L3 transitive: {len(L3)} edges")

# ────────────────────────────────────────────────────────────
# CELL 8 — COXPRESdb v8.1: Robust Download + Parse (Colab)
# ✅ Handles: download failures, manual upload fallback,
#    ID format detection, chunked parsing with MR ranking
# ────────────────────────────────────────────────────────────

# ── PREREQUISITES (define in prior cell) ─────────────────────
# GENE_SET = {"TP53", "BRCA1", "EGFR"}  # UPPERCASE symbols
# COEX_TOPN = 5

# ── Imports ─────────────────────────────────────────────────
import os, sys, zipfile, logging, warnings, time, shutil
from pathlib import Path
from typing import Set, Optional, List
import pandas as pd
import requests  # Colab has this pre-installed

# ── Logging ─────────────────────────────────────────────────
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────
CACHE_DIR             = "/content/cache"
COXPRESDB_EXTRACT_DIR = "/content/coxpresdb_extract"
COXPRESDB_CACHE       = Path(CACHE_DIR) / "l4_coex.tsv"

# Zenodo download URLs (v8.1 human union dataset)
# Try these in order if one fails:
COXPRESDB_URLS = [
    # Primary: union (microarray + RNA-seq)
    "https://zenodo.org/records/6861444/files/Hsa-u.v22-05.G16651-S245698.combat_pca.subagging.z.d.zip",
    # Fallback 1: microarray-only (smaller, ~300 MB)
    "https://zenodo.org/records/6861444/files/Hsa-m.v21-06.G20283-S25362.combat_pca.subagging.ls.d.zip",
    # Fallback 2: RNA-seq-only
    "https://zenodo.org/records/6861444/files/Hsa-r.v21-06.G16651-S245698.combat_pca.subagging.ls.d.zip",
]

# Local paths (will be set after successful download)
COXPRESDB_LOCAL_ZIP = None


# ── Helper: Robust Download with Retry/Fallback ─────────────
def download_coxpresdb_with_fallback(output_path: str) -> bool:
    """
    Attempt to download COXPRESdb data with multiple URLs and retry logic.
    Returns True if successful, False otherwise.
    """
    global COXPRESDB_LOCAL_ZIP

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1e6:
        size_mb = os.path.getsize(output_path) / 1e6
        log.info("[cache] Found existing zip: %.1f MB", size_mb)
        COXPRESDB_LOCAL_ZIP = output_path
        return True

    log.info("Attempting to download COXPRESdb human coexpression data...")

    for i, url in enumerate(COXPRESDB_URLS, 1):
        log.info("Try %d/%d: %s", i, len(COXPRESDB_URLS), url[:80] + "...")

        try:
            # Method 1: HTTP stream download (pure Python, no shell magics)
            import urllib.request, shutil as _sh
            _req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GRN-research)"})
            with urllib.request.urlopen(_req, timeout=120) as _resp, open(output_path, "wb") as _out:
                _sh.copyfileobj(_resp, _out)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 1e6:
                size_mb = os.path.getsize(output_path) / 1e6
                log.info("✓ Download successful: %.1f MB", size_mb)
                COXPRESDB_LOCAL_ZIP = output_path
                return True
            else:
                log.warning("Downloaded file too small or missing; cleaning up")
                if os.path.exists(output_path):
                    os.remove(output_path)

        except Exception as e:
            log.warning("Download attempt %d failed: %s", i, e)
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    # ── Fallback: Manual Upload Instructions ─────────────────
    log.error("❌ All download attempts failed.")
    print("\n" + "⚠️ MANUAL UPLOAD FALLBACK".center(60, "─"))
    print("1. Visit: https://zenodo.org/records/6861444")
    print("2. Download ONE of these files:")
    print("   • Hsa-u.v22-05...zip  (union, ~1.1 GB) ← RECOMMENDED")
    print("   • Hsa-m.v21-06...zip  (microarray, ~300 MB)")
    print("   • Hsa-r.v21-06...zip  (RNA-seq, ~800 MB)")
    print("3. Upload to Colab via folder icon 📁 in left sidebar")
    print("4. Set the path below and re-run:")
    print(f'   COXPRESDB_LOCAL_ZIP = "/content/your-downloaded-file.zip"')
    print("─" * 60 + "\n")
    return False


# ── Helper: Column Detection ─────────────────────────────────
def _detect_col(header: list, patterns: list, fallback: int) -> int:
    for i, col in enumerate(header):
        col_lower = col.lower().strip()
        if any(p.lower() in col_lower for p in patterns):
            return i
    return fallback


# ── Helper: Gene ID Normalization ────────────────────────────
def _normalise_id(raw: str, gene_id_type: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    if gene_id_type == "entrez":
        return raw if raw.isdigit() else None
    return raw.upper()  # symbol / ensembl


# ── Main Parser ──────────────────────────────────────────────
def parse_coxpresdb_bulk(
    gene_set: Set[str],
    top_n: int = 5,
    chunk_limit: Optional[int] = None,
    gene_id_type: str = "symbol",
    zip_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse COXPRESdb v8.1 bulk data with robust error handling.

    Parameters
    ----------
    gene_set : Set[str] - genes to filter (format depends on gene_id_type)
    top_n : int - partners per gene to retain (ranked by MR)
    chunk_limit : Optional[int] - limit chunks for testing
    gene_id_type : str - "symbol" | "entrez" | "ensembl"
    zip_path : Optional[str] - override auto-detected zip path

    Returns
    -------
    pd.DataFrame with columns: source, target, sign, level, db
    """
    global COXPRESDB_LOCAL_ZIP

    if not gene_set:
        log.error("gene_set is empty")
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Set zip path
    if zip_path:
        COXPRESDB_LOCAL_ZIP = zip_path
    elif not COXPRESDB_LOCAL_ZIP:
        COXPRESDB_LOCAL_ZIP = "/content/Hsa_union_coex.zip"

    # Download if needed
    if not os.path.exists(COXPRESDB_LOCAL_ZIP):
        if not download_coxpresdb_with_fallback(COXPRESDB_LOCAL_ZIP):
            return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Extract if needed
    extract_dir = Path(COXPRESDB_EXTRACT_DIR)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting zip (this may take 1-3 min)...")
        try:
            with zipfile.ZipFile(COXPRESDB_LOCAL_ZIP, "r") as zf:
                zf.extractall(extract_dir)
            log.info("✓ Extraction complete")
        except zipfile.BadZipFile:
            log.error("❌ Invalid zip file — re-download or upload manually")
            return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Find chunk files (.d extension or numeric names)
    chunk_files = sorted(
        p for p in extract_dir.iterdir()
        if p.suffix == ".d" and p.stem.isdigit()
    )
    if not chunk_files:
        chunk_files = sorted(
            p for p in extract_dir.iterdir()
            if p.is_file() and p.name.isdigit()
        )
    if not chunk_files:
        log.error("No chunk files found in %s", extract_dir)
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    log.info("Found %d chunk files | gene_id_type=%s", len(chunk_files), gene_id_type)

    # Collect all valid pairs with MR scores
    collected: List[dict] = []
    files_processed = 0

    for path in chunk_files:
        if chunk_limit and files_processed >= chunk_limit:
            break
        files_processed += 1

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header_line = f.readline().strip()
                if not header_line:
                    continue
                header = header_line.split("\t")

                q_idx = _detect_col(header, ["gene1","query","source","gene_a","symbol1","id1"], 0)
                t_idx = _detect_col(header, ["gene2","target","partner","gene_b","symbol2","id2"], 1)
                mr_idx = _detect_col(header, ["mr","mutual_rank","mutualrank","rank","score"], -1)

                if q_idx == t_idx:
                    t_idx = q_idx + 1

                for row_num, line in enumerate(f):
                    parts = line.strip().split("\t")
                    if len(parts) <= max(q_idx, t_idx):
                        continue

                    src = _normalise_id(parts[q_idx], gene_id_type)
                    tgt = _normalise_id(parts[t_idx], gene_id_type)

                    if not src or not tgt or src == tgt:
                        continue
                    if src not in gene_set or tgt not in gene_set:
                        continue

                    # Parse MR (lower = stronger); fallback to row order
                    if mr_idx != -1 and mr_idx < len(parts):
                        try:
                            mr = float(parts[mr_idx])
                        except ValueError:
                            mr = float(row_num)
                    else:
                        mr = float(row_num)

                    collected.append({"source": src, "target": tgt, "mr": mr})

        except Exception as e:
            log.warning("Error reading %s: %s", path.name, e)
            continue

    if not collected:
        log.warning("No valid gene pairs found — check gene_id_type and GENE_SET")
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Rank by MR (ascending) and keep top_n per source
    raw_df = pd.DataFrame(collected).sort_values("mr")
    top_df = raw_df.groupby("source", sort=False).head(top_n).reset_index(drop=True)

    # Build bidirectional edges, deduplicate
    fwd = top_df[["source", "target"]].copy()
    rev = top_df.rename(columns={"source": "target", "target": "source"})[["source", "target"]].copy()

    edges = (pd.concat([fwd, rev], ignore_index=True)
               .drop_duplicates(subset=["source", "target"])
               .reset_index(drop=True))

    edges["sign"] = 0
    edges["level"] = 4
    edges["db"] = "COXPRESdb"

    log.info("COXPRESdb: %d edges | %d unique genes | top-%d by MR",
             len(edges), edges[["source","target"]].stack().nunique(), top_n)
    return edges


# ── Main Execution ───────────────────────────────────────────
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Check cache first
if COXPRESDB_CACHE.exists() and COXPRESDB_CACHE.stat().st_size > 100:
    try:
        L4 = pd.read_csv(COXPRESDB_CACHE, sep="\t")
        if not L4.empty:
            log.info("[cache] Loaded %d edges from %s", len(L4), COXPRESDB_CACHE)
        else:
            log.warning("Cached file is empty — will re-parse")
            L4 = None
    except Exception as e:
        log.warning("Cache read error: %s — will re-parse", e)
        L4 = None
else:
    L4 = None

# Parse if not cached
if L4 is None:
    if "GENE_SET" not in globals() or "COEX_TOPN" not in globals():
        raise NameError("Define GENE_SET (set) and COEX_TOPN (int) before running")

    # Normalize gene set for symbol matching
    gene_set_norm = {g.strip().upper() for g in GENE_SET if g}

    L4 = parse_coxpresdb_bulk(
        gene_set_norm,
        top_n=COEX_TOPN,
        chunk_limit=None,          # Set to 50 for quick testing
        gene_id_type="symbol",     # Try "entrez" if no matches
        zip_path=None,             # Or set manually: "/content/my-file.zip"
    )

    if not L4.empty:
        L4.to_csv(COXPRESDB_CACHE, sep="\t", index=False)
        log.info("✓ Saved to %s", COXPRESDB_CACHE)

# ── Output Summary ───────────────────────────────────────────
print(f"\n📊 COXPRESdb L4: {len(L4):,} edges" if L4 is not None else "\n📊 COXPRESdb L4: (parse failed)")

if L4 is not None and not L4.empty:
    print(f"   • Unique genes: {L4[['source','target']].stack().nunique():,}")
    print(f"   • Columns: {list(L4.columns)}")
    print(f"   • Sample:\n{L4.head(3).to_string(index=False)}")
elif L4 is not None:
    print("  ⚠️  0 edges — troubleshooting:")
    print("     1. Try gene_id_type='entrez' (COXPRESdb often uses numeric IDs)")
    print("     2. Verify GENE_SET matches file's ID format")
    print("     3. Test with chunk_limit=10 to inspect first chunk")
    print("     4. Run diagnostic: check header/column names in .d files")
else:
    print("  ❌ Parsing failed — check logs above for download/extract errors")

# ────────────────────────────────────────────────────────────
# 💡 Quick Test Mode (uncomment to debug):
# ────────────────────────────────────────────────────────────
# L4_test = parse_coxpresdb_bulk(
#     {"TP53", "BRCA1", "EGFR"},
#     top_n=3,
#     chunk_limit=20,      # Only scan first 20 chunks
#     gene_id_type="symbol"  # Try "entrez" if this returns 0 edges
# )
# print(f"Test run: {len(L4_test)} edges")
# if not L4_test.empty:
#     print(L4_test.head())
# ────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────
# CELL 9 — Merge all levels
# ────────────────────────────────────────────────────────────
def merge_edges(dfs):
    valid_dfs = [d for d in dfs if not d.empty]
    if not valid_dfs: return pd.DataFrame(columns=["source","target","sign","level","db"])
    all_e = pd.concat(valid_dfs, ignore_index=True)

    # Pandas 2.2+ compatible aggregation (replaces deprecated .apply())
    agg_dict = {
        "level": "min",
        "sign": lambda x: int(x.dropna().iloc[0]) if x.dropna().nunique() == 1 else 0,
        "db": lambda x: ",".join(sorted(x.unique()))
    }
    merged = all_e.groupby(["source", "target"], as_index=False).agg(agg_dict)
    log.info("Merged pool: %d unique (source,target) pairs", len(merged))
    return merged

POOL = merge_edges([L1_TRRUST, L1_OMNIPATH, L1_COLLECTRI, L2_GH, L3, L4])

print("\n=== Edge pool ===")
print(POOL.groupby("level")["source"].count().rename("edges").to_string())
print(f"\nPool total  : {len(POOL):,}")
print(f"Unique src  : {POOL['source'].nunique():,}")
print(f"Unique tgt  : {POOL['target'].nunique():,}")
print(f"\nSigned (L1) : {(POOL['sign'] != 0).sum():,}")
print(f"Unsigned    : {(POOL['sign'] == 0).sum():,}")

# Save final pool
POOL.to_csv(OUT_FILE, sep="\t", index=False)
print(f"\n✓ Saved to {OUT_FILE}")

# ────────────────────────────────────────────────────────────
# CELL 10 — Rule-of-10 budget + greedy (param-count stopping)
# ────────────────────────────────────────────────────────────
import networkx as nx
from tqdm import tqdm

# ── Parameter budget ─────────────────────────────────────────
#
#  Sign treatment:
#    sign ∈ {+1, −1}  (L1 known)  →  fixed constant, NOT learned
#                                     edge costs 2 params (K_d, n)
#    sign = 0         (L2–L4)     →  learnable s_ij ∈ [−1, +1]
#                                     edge costs 3 params (K_d, n, s)
#
#  Total params:
#    gene params  :  3 × 2000                      =  6 000
#    edge params  :  2 × E_signed + 3 × E_unsigned  ≤  22 000
#
#  Worst case (all unsigned):  E ≤ 22 000 / 3 ≈  7 333  →  z ≈ 3.7
#  Best case  (all signed  ):  E ≤ 22 000 / 2 = 11 000  →  z = 5.5
#  Typical    (f_signed ≈ 0.3):
#    E ≤ 22 000 / (3 − f) = 22 000 / 2.7 ≈  8 148
#
#  → Stop greedily when cumulative param cost hits 22 000.
#    This is tighter and more honest than a fixed edge count.

N_GENES             = 2_000
PARAMS_GENE         = 3          # V_j, α_j, b_j  (per gene, always learned)
PARAMS_EDGE_SIGNED  = 2          # K_d, n          (sign is a known constant)
PARAMS_EDGE_UNSIGNED= 3          # K_d, n, s_ij    (sign is learned ∈ [−1,+1])
N_CONSTRAINTS       = 280_000

PARAM_BUDGET        = N_CONSTRAINTS // 10                        # 28 000
GENE_PARAM_TOTAL    = PARAMS_GENE * N_GENES                      #  6 000
EDGE_PARAM_BUDGET   = PARAM_BUDGET - GENE_PARAM_TOTAL            # 22 000

# For reporting: estimate z under typical pool composition
# (computed precisely inside the algorithm when pool sign-mix is known)
_z_worst = EDGE_PARAM_BUDGET / PARAMS_EDGE_UNSIGNED / N_GENES    # ≈ 3.67
_z_best  = EDGE_PARAM_BUDGET / PARAMS_EDGE_SIGNED   / N_GENES    # = 5.5

print(f"Parameter budget    : {PARAM_BUDGET:,}")
print(f"Gene params         : {GENE_PARAM_TOTAL:,}")
print(f"Edge param budget   : {EDGE_PARAM_BUDGET:,}")
print(f"z  (all unsigned)   : {_z_worst:.2f}  →  E_max ≈ {int(_z_worst*N_GENES):,}")
print(f"z  (all signed)     : {_z_best:.2f}   →  E_max ≈ {int(_z_best *N_GENES):,}")
print("Stopping criterion  : cumulative edge-param cost ≤ EDGE_PARAM_BUDGET")


# ── Helper: param cost per edge ───────────────────────────────
def _edge_cost(sign: int) -> int:
    """2 if sign is known (±1), 3 if unknown (0)."""
    return PARAMS_EDGE_SIGNED if sign != 0 else PARAMS_EDGE_UNSIGNED


# ── Main algorithm ────────────────────────────────────────────
def greedy_select_connected(edges,
                             edge_param_budget = EDGE_PARAM_BUDGET,
                             reward            = GREEDY_REWARD,
                             level_conf        = LEVEL_CONF):
    """
    Select edges until the GRNN parameter budget is exhausted.
    Stopping criterion: sum(param_cost per selected edge) ≤ edge_param_budget.

    Each edge costs:
      2 params  if sign ∈ {+1, −1}  (K_d, n;  sign fixed from database)
      3 params  if sign = 0          (K_d, n, s_ij learnable ∈ [−1, +1])

    Phase 0: Pool → LCC (drop unreachable nodes)
    Phase 1: Kruskal spanning tree (connectivity guarantee)
    Phase 2: Reward-penalty greedy fill up to param budget
    """
    edges  = edges.copy().reset_index(drop=True)
    signs  = edges["sign"].to_numpy(dtype=int)
    costs  = np.array([_edge_cost(s) for s in signs], dtype=int)

    # ── Phase 0: LCC restriction ─────────────────────────────
    G_pool = nx.from_pandas_edgelist(
                 edges, "source", "target", create_using=nx.Graph())
    comps  = sorted(nx.connected_components(G_pool), key=len, reverse=True)

    if len(comps) > 1:
        dropped = set().union(*comps[1:])
        log.warning(
            "Pool: %d components → restricting to LCC (%d nodes). "
            "Dropping %d unreachable nodes: %s …",
            len(comps), len(comps[0]), len(dropped), sorted(dropped)[:8],
        )
        lcc   = comps[0]
        edges = edges[
            edges["source"].isin(lcc) & edges["target"].isin(lcc)
        ].reset_index(drop=True)
        signs = edges["sign"].to_numpy(dtype=int)
        costs = np.array([_edge_cost(s) for s in signs], dtype=int)
        log.warning("Pool after LCC: %d edges, %d nodes.", len(edges), len(lcc))
    else:
        log.info("Pool connected (%d nodes). No repair needed.", len(comps[0]))

    # ── Arrays ───────────────────────────────────────────────
    conf_arr = edges["level"].map(level_conf).fillna(0.1).to_numpy(dtype=float)
    src_arr  = edges["source"].to_numpy()
    tgt_arr  = edges["target"].to_numpy()

    all_nodes = sorted(set(src_arr) | set(tgt_arr))
    N         = len(all_nodes)
    node2id   = {n: i for i, n in enumerate(all_nodes)}
    src_ids   = np.array([node2id[s] for s in src_arr], dtype=int)
    tgt_ids   = np.array([node2id[t] for t in tgt_arr], dtype=int)

    # Check spanning-tree cost fits budget
    span_cost = sum(
        sorted([_edge_cost(s) for s in signs])[: N - 1]
    )
    if span_cost > edge_param_budget:
        log.warning(
            "Spanning tree alone costs %d params > budget %d. "
            "Output may not be fully connected.", span_cost, edge_param_budget)

    # ── Union-Find ───────────────────────────────────────────
    parent = np.arange(N, dtype=int)
    rnk    = np.zeros(N, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return False
        if rnk[rx] < rnk[ry]: rx, ry = ry, rx
        parent[ry] = rx
        if rnk[rx] == rnk[ry]: rnk[rx] += 1
        return True

    remaining      = np.ones(len(edges), dtype=bool)
    selected       = []
    params_used    = 0

    # ── Phase 1: Kruskal spanning tree ───────────────────────
    n_comp   = N
    bridged  = 0
    p1_order = np.argsort(-conf_arr)   # best-confidence first

    for idx in p1_order:
        if n_comp <= 1: break
        cost = costs[idx]
        if params_used + cost > edge_param_budget: continue  # too expensive
        if union(src_ids[idx], tgt_ids[idx]):
            selected.append(int(idx))
            remaining[int(idx)] = False
            params_used += cost
            n_comp      -= 1
            bridged     += 1

    if n_comp > 1:
        log.error("Phase 1: %d components remain after spanning tree.", n_comp)
    else:
        log.info("Phase 1: %d spanning-tree edges (%d params used) → connected ✓",
                 bridged, params_used)

    # ── Phase 2: reward-penalty greedy ───────────────────────
    accum = np.zeros(len(edges), dtype=float)

    pbar = tqdm(desc="Greedy phase 2", total=edge_param_budget - params_used)
    while params_used < edge_param_budget and remaining.any():

        # (1) reward all remaining
        accum[remaining] += reward

        # (2) pick argmax
        best = int(np.argmax(np.where(remaining, conf_arr + accum, -np.inf)))
        cost = costs[best]

        if params_used + cost > edge_param_budget:
            # Try to fill remaining budget with cheaper (signed) edges only
            signed_remaining = remaining & (costs == PARAMS_EDGE_SIGNED)
            if not signed_remaining.any():
                break
            best = int(np.argmax(
                np.where(signed_remaining, conf_arr + accum, -np.inf)))
            cost = costs[best]
            if params_used + cost > edge_param_budget:
                break

        selected.append(best)
        remaining[best] = False
        params_used     += cost
        pbar.update(cost)

        # (3) penalise edges sharing source u or target v
        accum[remaining & (src_arr == src_arr[best])] -= reward  # (u, v')
        accum[remaining & (tgt_arr == tgt_arr[best])] -= reward  # (u', v)

    pbar.close()

    # ── Build result ─────────────────────────────────────────
    result = edges.iloc[selected].drop(columns=["conf"], errors="ignore").copy()

    # ── Hard connectivity assert ─────────────────────────────
    U         = nx.from_pandas_edgelist(result, "source", "target",
                                        create_using=nx.Graph())
    out_comps = sorted(nx.connected_components(U), key=len, reverse=True)
    if len(out_comps) > 1:
        raise RuntimeError(
            f"Output NOT connected ({len(out_comps)} components). "
            f"Sizes: {[len(c) for c in out_comps[:10]]}"
        )

    n_signed   = (result["sign"] != 0).sum()
    n_unsigned = (result["sign"] == 0).sum()
    z_actual   = len(result) / N_GENES

    log.info(
        "✓  %d edges | %d nodes | 1 component | %d params used / %d budget\n"
        "   signed (%d×2=%d params)  unsigned (%d×3=%d params)  z=%.2f",
        len(result), U.number_of_nodes(), params_used, edge_param_budget,
        n_signed,   n_signed   * PARAMS_EDGE_SIGNED,
        n_unsigned, n_unsigned * PARAMS_EDGE_UNSIGNED,
        z_actual,
    )
    return result, params_used


# ── Run ───────────────────────────────────────────────────────
SELECTED, PARAMS_USED = greedy_select_connected(POOL)
SELECTED.to_csv(OUT_FILE, sep="\t", index=False)

n_signed   = (SELECTED["sign"] != 0).sum()
n_unsigned = (SELECTED["sign"] == 0).sum()

print(f"\n=== Final GRN ===")
print(f"Total edges         : {len(SELECTED):,}")
print(f"  signed  (±1, 2p)  : {n_signed:,}   → {n_signed*2:,} params")
print(f"  unsigned (0, 3p)  : {n_unsigned:,}  → {n_unsigned*3:,} params")
print(f"Edge params used    : {PARAMS_USED:,}  /  {EDGE_PARAM_BUDGET:,}")
print(f"Gene params         : {GENE_PARAM_TOTAL:,}")
print(f"Total params        : {PARAMS_USED + GENE_PARAM_TOTAL:,}  /  {PARAM_BUDGET:,}")
print(f"z edges/gene        : {len(SELECTED)/N_GENES:.2f}")
print(f"\n✓ Saved → {OUT_FILE}")

genes_to_check = ['GATA1', 'MYC', 'BCL11A', 'HBG']

present_genes = set()
for gene in genes_to_check:
    if (SELECTED['source'] == gene).any() or (SELECTED['target'] == gene).any():
        present_genes.add(gene)

print(f"Genes present in K562 GRN: {list(present_genes)}")
missing_genes = set(genes_to_check) - present_genes
print(f"Genes not present in K562 GRN: {list(missing_genes)}")

# Extract full gene list from the GRN graph
grn_genes = set(SELECTED['source']).union(set(SELECTED['target']))
grn_gene_list = sorted(list(grn_genes))

print(f"Total unique genes in the GRN graph: {len(grn_gene_list)}")
print(grn_gene_list)

# Log/save the GRN
log_path = 'K562_grn_logged.tsv'
SELECTED.to_csv(log_path, sep='\t', index=False)
print(f"\nLogged K562 GRN to '{log_path}'")

# ────────────────────────────────────────────────────────────
# CHECK — Is selected output fully connected?
# ────────────────────────────────────────────────────────────
import networkx as nx

def check_fully_connected(edges, source_col="source", target_col="target"):
    G = nx.from_pandas_edgelist(
        edges,
        source=source_col,
        target=target_col,
        create_using=nx.Graph()
    )

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        print("Output graph is empty.")
        return False

    is_connected = nx.is_connected(G)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)

    print("=== Connectivity check ===")
    print(f"Nodes       : {n_nodes:,}")
    print(f"Edges       : {n_edges:,}")
    print(f"Components  : {len(comps):,}")
    print(f"Connected   : {is_connected}")

    if not is_connected:
        print("\nComponent sizes:")
        print([len(c) for c in comps[:20]])

        print("\nExample nodes from smaller components:")
        for i, comp in enumerate(comps[1:6], start=2):
            print(f"Component {i} size={len(comp)} nodes={sorted(list(comp))[:10]}")

    return is_connected


# If SELECTED is already in memory:
IS_CONNECTED = check_fully_connected(SELECTED)

# Optional hard assert:
assert IS_CONNECTED, "Selected GRN is NOT fully connected."

# ────────────────────────────────────────────────────────────
# CELL 11 — Plot FULL selected GRN with Cytoscape.js
# ────────────────────────────────────────────────────────────
# Plots ALL edges in SELECTED, not only top-300.
# Warning: if SELECTED has ~8k–11k edges, rendering can be slow in Colab.

import json
import html as html_escape
from collections import Counter

import pandas as pd
import networkx as nx
from IPython.display import display, HTML


# ── Connectivity check ───────────────────────────────────────
G_full = nx.from_pandas_edgelist(
    SELECTED,
    source="source",
    target="target",
    create_using=nx.Graph()
)

print("=== Full SELECTED connectivity ===")
print(f"Nodes      : {G_full.number_of_nodes():,}")
print(f"Edges      : {G_full.number_of_edges():,}")
print(f"Components : {nx.number_connected_components(G_full):,}")
print(f"Connected  : {nx.is_connected(G_full)}")

if not nx.is_connected(G_full):
    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
    print("Component sizes:", [len(c) for c in comps[:20]])


def build_full_cytoscape_html(
    edges_df: pd.DataFrame,
    height_px: int = 800,
) -> str:
    """
    Convert FULL edges_df with columns source, target, sign, level, db optional
    into an interactive Cytoscape.js graph.
    """

    required = {"source", "target", "sign", "level"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = edges_df.copy().reset_index(drop=True)

    if "db" not in df.columns:
        df["db"] = "NA"

    deg = Counter(df["source"].tolist() + df["target"].tolist())
    max_deg = max(deg.values()) if deg else 1

    node_ids = sorted(set(df["source"]) | set(df["target"]))

    cy_nodes = [
        {
            "data": {
                "id": str(n),
                "label": str(n),
                "deg": int(deg[n]),
                "size": float(10 + 35 * (deg[n] / max_deg)),
            }
        }
        for n in node_ids
    ]

    edge_colour = {
        1: "#27ae60",
        -1: "#e74c3c",
        0: "#95a5a6",
    }

    level_width = {
        1: 2.4,
        2: 1.8,
        3: 1.3,
        4: 0.8,
    }

    cy_edges = []
    for i, e in enumerate(df.itertuples(index=False)):
        source = str(getattr(e, "source"))
        target = str(getattr(e, "target"))
        sign = int(getattr(e, "sign"))
        level = int(getattr(e, "level"))
        db = str(getattr(e, "db"))

        cy_edges.append(
            {
                "data": {
                    "id": f"edge_{i}",
                    "source": source,
                    "target": target,
                    "sign": sign,
                    "level": level,
                    "db": db,
                    "color": edge_colour.get(sign, "#95a5a6"),
                    "width": level_width.get(level, 0.8),
                }
            }
        )

    elements_json = json.dumps(cy_nodes + cy_edges)

    html = f"""
<div id="grn-wrap" style="font-family:sans-serif">

  <div style="display:flex;gap:16px;align-items:center;margin-bottom:8px;flex-wrap:wrap">
    <b style="font-size:15px">FULL GRN: {len(df)} edges / {len(node_ids)} genes</b>

    <label>Layout&nbsp;
      <select id="layoutSel" onchange="changeLayout()" style="padding:3px 6px">
        <option value="cose" selected>CoSE force</option>
        <option value="circle">Circle</option>
        <option value="grid">Grid</option>
        <option value="concentric">Concentric</option>
        <option value="breadthfirst">BFS tree</option>
      </select>
    </label>

    <label>Highlight gene&nbsp;
      <input id="geneBox" type="text"
             placeholder="e.g. MYC"
             style="width:110px;padding:3px"
             oninput="highlightGene(this.value.trim())">
    </label>

    <button onclick="cy.fit()" style="padding:4px 10px">Reset view</button>
  </div>

  <div style="display:flex;gap:12px;margin-bottom:6px;font-size:12px;flex-wrap:wrap">
    <span><span style="color:#27ae60;font-weight:bold">→</span> Activation</span>
    <span><span style="color:#e74c3c;font-weight:bold">→</span> Repression</span>
    <span><span style="color:#95a5a6;font-weight:bold">→</span> Unsigned</span>
    <span>|</span>
    <span style="background:#2980b9;color:#fff;padding:1px 6px;border-radius:3px">L1 curated</span>
    <span style="background:#8e44ad;color:#fff;padding:1px 6px;border-radius:3px">L2 GeneHancer</span>
    <span style="background:#e67e22;color:#fff;padding:1px 6px;border-radius:3px">L3 GeneHancer+PPI</span>
    <span style="background:#16a085;color:#fff;padding:1px 6px;border-radius:3px">L4 co-expression</span>
  </div>

  <div id="cy"
       style="width:100%;height:{height_px}px;border:1px solid #ddd;border-radius:6px;background:#fafafa">
  </div>

  <div id="tip"
       style="display:none;position:fixed;background:rgba(0,0,0,.78);color:#fff;
              padding:6px 10px;border-radius:5px;font-size:12px;
              pointer-events:none;z-index:9999">
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>

<script>
const ALL_ELEMENTS = {elements_json};

var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: ALL_ELEMENTS,

  style: [
    {{
      selector: 'node',
      style: {{
        'label': 'data(label)',
        'width': 'data(size)',
        'height': 'data(size)',
        'background-color': '#3498db',
        'color': '#222',
        'font-size': 7,
        'text-valign': 'center',
        'text-halign': 'center',
        'text-wrap': 'wrap',
        'overlay-padding': '4px'
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 'data(width)',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'opacity': 0.45
      }}
    }},
    {{
      selector: '.faded',
      style: {{ 'opacity': 0.04 }}
    }},
    {{
      selector: '.highlighted',
      style: {{
        'background-color': '#f39c12',
        'border-width': 3,
        'border-color': '#e67e22',
        'z-index': 10,
        'font-size': 13
      }}
    }},
    {{
      selector: '.highlighted-edge',
      style: {{
        'opacity': 1,
        'width': 3.5
      }}
    }}
  ],

  layout: {{
    name: 'cose',
    animate: false,
    randomize: true,
    nodeRepulsion: 12000,
    idealEdgeLength: 60,
    edgeElasticity: 80,
    numIter: 2500
  }},

  wheelSensitivity: 0.25
}});

const tip = document.getElementById('tip');

cy.on('mouseover', 'node', e => {{
  const d = e.target.data();
  tip.style.display = 'block';
  tip.innerHTML = `<b>${{d.label}}</b><br>degree: ${{d.deg}}`;
}});

cy.on('mouseover', 'edge', e => {{
  const d = e.target.data();
  const signStr = d.sign === 1 ? 'activation' : d.sign === -1 ? 'repression' : 'unsigned';
  tip.style.display = 'block';
  tip.innerHTML = `${{d.source}} → ${{d.target}}<br>${{signStr}} | L${{d.level}}<br><small>${{d.db}}</small>`;
}});

cy.on('mouseout', () => {{
  tip.style.display = 'none';
}});

document.addEventListener('mousemove', e => {{
  tip.style.left = (e.clientX + 14) + 'px';
  tip.style.top = (e.clientY - 10) + 'px';
}});

function layoutOptions(name) {{
  const opts = {{
    cose: {{
      name: 'cose',
      animate: false,
      randomize: true,
      nodeRepulsion: 12000,
      idealEdgeLength: 60,
      edgeElasticity: 80,
      numIter: 2500
    }},
    circle: {{
      name: 'circle',
      animate: false
    }},
    grid: {{
      name: 'grid',
      animate: false
    }},
    concentric: {{
      name: 'concentric',
      animate: false,
      concentric: n => n.data('deg'),
      levelWidth: () => 2
    }},
    breadthfirst: {{
      name: 'breadthfirst',
      animate: false,
      directed: true
    }}
  }};

  return opts[name] || opts.cose;
}}

function changeLayout() {{
  const name = document.getElementById('layoutSel').value;
  cy.layout(layoutOptions(name)).run();
}}

function highlightGene(geneRaw) {{
  const gene = geneRaw.toUpperCase();

  cy.elements().removeClass('highlighted highlighted-edge faded');

  if (!gene) return;

  const target = cy.nodes().filter(n => {{
    return String(n.data('label')).toUpperCase() === gene;
  }});

  if (target.length === 0) return;

  cy.elements().addClass('faded');

  const hood = target.closedNeighborhood();
  hood.removeClass('faded');
  hood.nodes().addClass('highlighted');
  hood.edges().addClass('highlighted-edge');

  target.addClass('highlighted');
  cy.fit(hood, 60);
}}
</script>
"""
    return html


def export_graphml(edges_df: pd.DataFrame, path: str = "/content/grn_full.graphml"):
    """
    Export FULL graph as GraphML for Cytoscape desktop.
    """

    required = {"source", "target", "sign", "level"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = edges_df.copy()

    if "db" not in df.columns:
        df["db"] = "NA"

    nodes = sorted(set(df["source"]) | set(df["target"]))

    def esc(x):
        return html_escape.escape(str(x), quote=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
        '  <key id="sign" for="edge" attr.name="sign" attr.type="int"/>',
        '  <key id="level" for="edge" attr.name="level" attr.type="int"/>',
        '  <key id="db" for="edge" attr.name="db" attr.type="string"/>',
        '  <graph id="GRN" edgedefault="directed">',
    ]

    for n in nodes:
        lines.append(f'    <node id="{esc(n)}"/>')

    for i, e in enumerate(df.itertuples(index=False)):
        lines.extend(
            [
                f'    <edge id="e{i}" source="{esc(getattr(e, "source"))}" target="{esc(getattr(e, "target"))}">',
                f'      <data key="sign">{int(getattr(e, "sign"))}</data>',
                f'      <data key="level">{int(getattr(e, "level"))}</data>',
                f'      <data key="db">{esc(getattr(e, "db"))}</data>',
                '    </edge>',
            ]
        )

    lines.extend(["  </graph>", "</graphml>"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Full GraphML saved → {path}")


# ── Render FULL graph and export FULL GraphML ────────────────
display(HTML(build_full_cytoscape_html(SELECTED, height_px=850)))
export_graphml(SELECTED, "/content/grn_full.graphml")

################################################################################
# CELL 12 — GRNN: build → plots before → train → plots after → target GRN
#
# Biology fixes vs. naive GRNN
# ─────────────────────────────
#   [1] x0: log1p(mean(expm1(X))) — bias-free baseline (Jensen's gap fixed)
#   [2] perturbed_value: observed expression of the KO'd gene per experiment
#   [3] Kd_raw: inv_softplus(x0_src) → Hill occupancy φ(x0) ≈ 0.5 at init
#   [4] Dual-channel _step: h_act in numerator / h_rep in denominator
#       (non-zero gradients for both channels regardless of act/rep ratio)
#
# Weight parameterisation
# ───────────────────────
#   sign ∈ {+1,−1} → w_hat = sign · sigmoid(w_raw)   direction fixed by DB
#   sign = 0       → w_hat = tanh(w_raw)              direction learned
#
# Visual conventions (all panels)
# ────────────────────────────────
#   Red    = activation  (eff > 0)     Blue  = repression (eff < 0)
#   Grey   = unknown / near-zero       RdBu_r palette (CVD-safe)
#   Solid  = DB-constrained direction  Dashed = sign=0 (direction learned)
#
# Outputs
# ───────
#   /content/grn_plots/{CELL_LINE}_heatmap_{before,after}.png/.pdf
#   /content/grn_plots/{CELL_LINE}_{GENE}_{before,after,changed}.png/.pdf
#   /content/grn_plots/{CELL_LINE}_{GENE}_{before,after}.png/.pdf  ← targets
#   (GENE = VIZ_GENES + GATA1 + MITF)
################################################################################

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import pandas as pd, numpy as np, json, networkx as nx, logging
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from IPython.display import display, HTML
from tqdm import tqdm

log = logging.getLogger("grn")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ════════════════════════════════════════════════════════════
#  PUBLICATION STYLE
#  Q1 journal column widths (Nature Methods / Cell / Cell Systems)
#  NeurIPS: COL_1=3.46  Cell Press: CP_COL1=3.35, CP_COL15=4.49, CP_COL2=6.85
# ════════════════════════════════════════════════════════════
COL_1,  COL_15,  COL_2  = 3.46, 5.00, 7.20    # NeurIPS / Nature Methods
CP_COL1, CP_COL15, CP_COL2 = 3.35, 4.49, 6.85 # Cell Press

mpl.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":    "dejavusans",
    "font.size":           7,   "axes.titlesize":     8,
    "axes.labelsize":      7,   "xtick.labelsize":    6.5,
    "ytick.labelsize":     6.5, "legend.fontsize":    6.5,
    "legend.frameon":      False,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.5, "ytick.major.width":  0.5,
    "xtick.major.size":    2.5, "ytick.major.size":   2.5,
    "lines.linewidth":     1.0, "patch.linewidth":    0.5,
    "axes.spines.top":     False, "axes.spines.right": False,
    "axes.grid":           False,
    "savefig.dpi":         600,   # Cell Press line-art standard
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "figure.dpi":          150,
    "pdf.fonttype":        42,  # TrueType embedded — editable in Illustrator
    "ps.fonttype":         42,
    "svg.fonttype":        "none",
})

# CVD-safe RdBu palette (Nature/Cell figure convention)
ACT_C    = "#B2182B"   # deep red    — activation
REP_C    = "#2166AC"   # deep blue   — repression
UNK_C    = "#BDBDBD"   # neutral grey
NODE_CTR = "#F4A261"   # warm amber  — centre / perturbed gene
NODE_H1  = "#457B9D"   # slate blue  — direct neighbours
NODE_H2  = "#D9D9D9"   # cool grey   — 2nd hop
EDGE_LBL = "#3A3A3A"
CMAP_DIV = plt.get_cmap("RdBu_r")
EDGE_C_SIGN = {1: ACT_C, -1: REP_C, 0: UNK_C}
NODE_C   = {"center": NODE_CTR, "hop1": NODE_H1, "hop2": NODE_H2}
NODE_S   = {"center": 360,      "hop1": 220,     "hop2": 110}

PLOT_DIR = Path("/content/grn_plots")
PLOT_DIR.mkdir(exist_ok=True)

ensg2sym         = adata.var["gene_name"].to_dict()
GENE_NAMES       = [ensg2sym.get(e, e) for e in hvg_list]
PERTURBATION_COL = "gene"
CONTROL_LABEL    = "non-targeting"


# ════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════
def _logmean(X):
    """log1p(mean(expm1(X))) — bias-free baseline in log1p count space."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.log1p(np.expm1(X).mean(axis=0)).ravel()


def _inv_softplus(y, eps=1e-6):
    """Inverse of F.softplus so softplus(_inv_softplus(y)) ≈ y."""
    y = torch.as_tensor(y, dtype=torch.float32).clamp(min=eps)
    return torch.log(torch.expm1(y))


def _save(fig, stem):
    """Save PNG (600 dpi) and PDF (vector TrueType) side by side."""
    for suffix in (".png", ".pdf"):
        kw = dict(facecolor="white", bbox_inches="tight")
        if suffix == ".png": kw["dpi"] = 600
        fig.savefig(PLOT_DIR / f"{stem}{suffix}", **kw)
    print(f"  ✓ {stem}.png + .pdf")


def _eff_color(w, thr=0.02):
    return ACT_C if w > thr else REP_C if w < -thr else UNK_C


def _before_edge_color(sign):
    return EDGE_C_SIGN.get(int(sign), UNK_C)


# ════════════════════════════════════════════════════════════
#  1. GRNN
# ════════════════════════════════════════════════════════════
class GRNN(nn.Module):
    """
    Gene Regulatory Neural Network — biophysical Hill-kinetics model.

    Key design decisions
    ────────────────────
    w_hat: sign ≠ 0 → sign·sigmoid(w_raw)   (DB-constrained direction)
           sign = 0 → tanh(w_raw)            (direction learned from data)

    Kd_raw = inv_softplus(x0_src) so Hill occupancy φ(x0) ≈ 0.5 at init,
    placing each TF in the responsive (unsaturated) regime.

    Dual-channel _step: separate scatter_add for activation (h_act) and
    repression (h_rep) guarantees non-zero gradients for both channels.
      x_ss = (V·(1+h_act) + b) / (1 + α + h_rep)

    w_raw init (inv_sigmoid of desired |w_hat| magnitude):
      Activation L1→0.85  L2→0.65  L3→0.45  L4→0.25
      Repression L1→0.90  L2→0.70  L3→0.50  L4→0.30
      Unknown    → N(0,0.1)
    """
    def __init__(self, gene_names, src_idx, tgt_idx, signs, levels,
                 x0_init, max_iter=100, eps=1e-5):
        super().__init__()
        self.gene_names = gene_names
        self.gene2idx   = {g: i for i, g in enumerate(gene_names)}
        self.N = len(gene_names); self.E = src_idx.numel()
        self.max_iter = max_iter; self.eps = eps

        self.register_buffer("src_idx", src_idx.long())
        self.register_buffer("tgt_idx", tgt_idx.long())
        self.register_buffer("signs",   signs.float())
        self.register_buffer("levels",  levels.float())

        self.log_V     = nn.Parameter(torch.log(x0_init.clamp(min=1e-6)))
        self.log_alpha = nn.Parameter(torch.full((self.N,), -2.3))
        self.b_raw     = nn.Parameter(torch.full((self.N,), -4.0))
        self.Kd_raw    = nn.Parameter(_inv_softplus(x0_init[src_idx].clamp(min=1e-3)))
        self.n_raw     = nn.Parameter(torch.zeros(self.E))

        def inv_sig(p): return float(np.log(p / (1 - p)))
        w = torch.zeros(self.E)
        act = signs == 1; rep = signs == -1; unk = signs == 0
        l1 = levels==1; l2 = levels==2; l3 = levels==3; l4 = levels==4
        w[act&l1]=inv_sig(0.85); w[act&l2]=inv_sig(0.65)
        w[act&l3]=inv_sig(0.45); w[act&l4]=inv_sig(0.25)
        w[rep&l1]=inv_sig(0.90); w[rep&l2]=inv_sig(0.70)
        w[rep&l3]=inv_sig(0.50); w[rep&l4]=inv_sig(0.30)
        w[unk]   =torch.randn(unk.sum()) * 0.1
        self.w_raw = nn.Parameter(w)

    @property
    def V(self):     return torch.exp(self.log_V)
    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def b(self):     return F.softplus(self.b_raw)
    @property
    def Kd(self):    return F.softplus(self.Kd_raw) + 1e-6
    @property
    def n(self):     return 1.0 + 3.0 * torch.sigmoid(self.n_raw)
    @property
    def w_hat(self):
        return torch.where(self.signs != 0,
                           self.signs * torch.sigmoid(self.w_raw),
                           torch.tanh(self.w_raw))

    @property
    def eff_weight(self):
        """eff_ij = ŵ_ij · φ_ij(x0_i) — regulatory drive at baseline."""
        with torch.no_grad():
            x0  = self.V[self.src_idx] * 0.8
            xn  = x0.pow(self.n); Kdn = self.Kd.pow(self.n)
            phi = xn / (Kdn + xn + 1e-12)
            return (self.w_hat * phi).cpu().numpy()

    def _eff_map(self):
        arr = self.eff_weight
        return {(self.gene_names[int(s)], self.gene_names[int(t)]): float(w)
                for s, t, w in zip(self.src_idx.cpu(), self.tgt_idx.cpu(), arr)}

    def _step(self, x):
        """Dual-channel thermodynamic update (non-zero gradients guaranteed)."""
        phi   = (x[self.src_idx].pow(self.n)
                 / (self.Kd.pow(self.n) + x[self.src_idx].pow(self.n) + 1e-12))
        drive = self.w_hat * phi
        h_act = torch.zeros(self.N, device=x.device, dtype=x.dtype)
        h_rep = torch.zeros(self.N, device=x.device, dtype=x.dtype)
        h_act.scatter_add_(0, self.tgt_idx, drive.clamp(min=0))
        h_rep.scatter_add_(0, self.tgt_idx, (-drive).clamp(min=0))
        return (self.V * (1.0 + h_act) + self.b) / (1.0 + self.alpha + h_rep)

    def forward(self, x0, perturbed_idx=None, perturbed_value=None):
        x = x0.clone()
        if perturbed_idx is not None and perturbed_value is not None:
            x[perturbed_idx] = perturbed_value
        for t in range(self.max_iter):
            x_new = self._step(x)
            if perturbed_idx is not None and perturbed_value is not None:
                x_new[perturbed_idx] = perturbed_value
            if (x_new - x).norm() < self.eps:
                return x_new, t + 1
            x = x_new
        return x, self.max_iter


# ════════════════════════════════════════════════════════════
#  2. TSV → model
# ════════════════════════════════════════════════════════════
def grn_tsv_to_grnn(tsv_path, gene_names, x0, max_iter=100, eps=1e-5):
    df = pd.read_csv(tsv_path, sep="\t")
    g2i = {g: i for i, g in enumerate(gene_names)}
    df  = df[df["source"].isin(g2i) & df["target"].isin(g2i)].copy()
    src = torch.tensor([g2i[g] for g in df["source"]], dtype=torch.long)
    tgt = torch.tensor([g2i[g] for g in df["target"]], dtype=torch.long)
    signs  = torch.tensor(df["sign"].fillna(0).astype(int).values,  dtype=torch.float32)
    levels = torch.tensor(df["level"].fillna(4).astype(int).values, dtype=torch.float32)
    model  = GRNN(gene_names, src, tgt, signs, levels,
                  torch.tensor(x0, dtype=torch.float32), max_iter, eps)
    log.info("GRNN: %d genes | %d edges (%d learned-sign) | %d params",
             len(gene_names), len(df), int((signs==0).sum()),
             sum(p.numel() for p in model.parameters()))
    return model, df


# ════════════════════════════════════════════════════════════
#  3. Dataset — FIX [1] logmean baseline, FIX [2] observed pv
# ════════════════════════════════════════════════════════════
class PerturbseqDataset(Dataset):
    def __init__(self, adata, gene_names, perturbation_col="gene",
                 control_label="non-targeting", min_cells=5,
                 ensg2sym=None, verbose=True):
        if ensg2sym is None: ensg2sym = {}
        var_ids = adata.var_names.tolist()
        sym2pos = {}
        for pos, ensg in enumerate(var_ids):
            sym = ensg2sym.get(ensg, ensg)
            if sym not in sym2pos: sym2pos[sym] = pos

        hvg_pos, valid_genes = [], []
        for sym in gene_names:
            if sym in sym2pos:
                hvg_pos.append(sym2pos[sym]); valid_genes.append(sym)
        if not valid_genes: raise ValueError("No valid HVG genes found.")

        self.gene_names = valid_genes
        self.gene2idx   = {g: i for i, g in enumerate(valid_genes)}
        self.hvg_pos    = np.array(hvg_pos, dtype=np.int64)

        ctrl_mask = adata.obs[perturbation_col].values == control_label
        self.x0   = torch.tensor(
            _logmean(adata.X[ctrl_mask])[self.hvg_pos], dtype=torch.float32)

        self.samples = []
        for gene_sym, grp in adata.obs.groupby(perturbation_col):
            if gene_sym == control_label or gene_sym not in self.gene2idx: continue
            if len(grp) < min_cells: continue
            gene_idx   = self.gene2idx[gene_sym]
            x_obs_full = _logmean(adata[grp.index].X)[self.hvg_pos]
            pv         = float(x_obs_full[gene_idx])   # FIX [2]: observed KD level
            self.samples.append((gene_idx, pv,
                                  torch.tensor(x_obs_full, dtype=torch.float32)))
        if not self.samples: raise ValueError("No perturbation samples.")
        if verbose:
            print(f"PerturbseqDataset: {len(self.samples)} experiments "
                  f"| {len(valid_genes)} HVGs")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, p, x = self.samples[idx]
        return {"perturbed_idx": int(g), "perturbed_value": float(p),
                "x_obs": x, "x0": self.x0}


def split_dataset(dataset, train_frac=0.70, val_frac=0.10, seed=42):
    n = len(dataset); n_tr = int(train_frac*n); n_v = int(val_frac*n)
    return random_split(dataset, [n_tr, n_v, n-n_tr-n_v],
                        generator=torch.Generator().manual_seed(seed))


# ════════════════════════════════════════════════════════════
#  4. Loss functions
# ════════════════════════════════════════════════════════════
def grnn_loss(x_pred, x_obs, x0, top_k=20,
              lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0,
              lam_deg=2.0,  lam_balance=0.5, gamma_focus=2.0):
    """
    WMSE     : weighted MSE — weight ∝ |Δobs|, prevents mode collapse
    AFDA     : autofocus direction-aware — focal weight on hard examples,
               cosine-based direction agreement, applied to |Δobs| ≥ 0.1
    Balance  : soft up-fraction alignment — prevents systematic act/rep bias
    Delta    : MSE on Δ = x_pred − x0  (avoids Δ=0 shortcut)
    DEG      : top-k importance-weighted MSE
    """
    d_obs = x_obs - x0; d_pred = x_pred - x0

    # WMSE
    w = d_obs.abs() + 1e-6; w = w / w.sum()
    loss_wmse = (w * (x_pred - x_obs).pow(2)).sum()

    # AFDA
    mask = d_obs.abs() >= 0.1
    if mask.any():
        dp = d_pred[mask]; do = d_obs[mask]
        cos    = (dp * do) / ((dp.abs() + 1e-8) * (do.abs() + 1e-8))
        agree  = (1.0 + cos) / 2.0
        fw     = (1.0 - agree).clamp(0, 1).pow(gamma_focus)
        loss_afda = (fw * (dp - do).pow(2)).mean()
    else:
        loss_afda = x_pred.new_tensor(0.0)

    # Balance
    tau = 0.05
    loss_balance = (torch.sigmoid(d_pred/tau).mean()
                    - torch.sigmoid(d_obs/tau).mean()).pow(2)

    # Delta + DEG
    loss_delta = F.mse_loss(d_pred, d_obs)
    k = min(top_k, len(d_obs))
    topk = d_obs.abs().topk(k).indices
    wts  = d_obs.abs()[topk] / (d_obs.abs()[topk].sum() + 1e-12)
    loss_deg = (wts * (x_pred[topk] - x_obs[topk]).pow(2)).sum()

    return (lam_wmse * loss_wmse + lam_afda * loss_afda
            + lam_balance * loss_balance + lam_delta * loss_delta
            + lam_deg * loss_deg)


@torch.no_grad()
def _eval_loss(model, subset, x0, device,
               top_k, lam_wmse, lam_afda, lam_delta,
               lam_deg, lam_balance, gamma_focus):
    model.eval(); total = 0.0
    for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
        x_obs = batch["x_obs"][0].to(device)
        xp, _ = model(x0, int(batch["perturbed_idx"][0]),
                      float(batch["perturbed_value"][0]))
        total += grnn_loss(xp, x_obs, x0, top_k, lam_wmse, lam_afda,
                           lam_delta, lam_deg, lam_balance, gamma_focus).item()
    return total / max(len(subset), 1)


@torch.no_grad()
def _sign_balance_report(model, subset, x0, device):
    model.eval(); pu, ou = [], []
    for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
        x_obs = batch["x_obs"][0].to(device)
        xp, _ = model(x0, int(batch["perturbed_idx"][0]),
                      float(batch["perturbed_value"][0]))
        pu.append((xp - x0 > 0).float().mean().item())
        ou.append((x_obs - x0 > 0).float().mean().item())
    return float(np.mean(pu)), float(np.mean(ou))


@torch.no_grad()
def _convergence_report(model, subset, x0, device):
    model.eval(); iters = []
    for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
        _, n = model(x0, int(batch["perturbed_idx"][0]),
                     float(batch["perturbed_value"][0]))
        iters.append(n)
    iters = np.array(iters)
    return float(np.median(iters)), float((iters == model.max_iter).mean())


def train_grnn(model, train_ds, val_ds, n_epochs=50, lr=1e-3,
               top_k_deg=20, lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0,
               lam_deg=2.0, lam_balance=0.5, gamma_focus=2.0,
               device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    x0    = train_ds.dataset.x0.to(device)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    train_h, val_h = [], []
    for epoch in range(1, n_epochs + 1):
        model.train(); el = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False):
            x_obs = batch["x_obs"][0].to(device)
            opt.zero_grad()
            xp, _ = model(x0, int(batch["perturbed_idx"][0]),
                          float(batch["perturbed_value"][0]))
            loss = grnn_loss(xp, x_obs, x0, top_k_deg, lam_wmse, lam_afda,
                             lam_delta, lam_deg, lam_balance, gamma_focus)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += loss.item()
        sched.step()
        train_h.append(el / len(loader))
        val_h.append(_eval_loss(model, val_ds, x0, device, top_k_deg,
                                lam_wmse, lam_afda, lam_delta, lam_deg,
                                lam_balance, gamma_focus))
        if epoch % 5 == 0 or epoch == 1:
            pu, ou = _sign_balance_report(model, val_ds, x0, device)
            mi, fn = _convergence_report(model, val_ds, x0, device)
            log.info("Ep %3d  tr=%.4f  val=%.4f  bias=%.2f  "
                     "med_it=%.0f  non_conv=%.1f%%",
                     epoch, train_h[-1], val_h[-1], pu-ou, mi, 100*fn)
    return train_h, val_h


def plot_loss_curves(train_h, val_h, cell_line):
    fig, ax = plt.subplots(figsize=(COL_1, COL_1 * 0.62))
    ep = np.arange(1, len(train_h)+1)
    ax.plot(ep, train_h, color="#2A6F97", lw=1.1, label="train")
    ax.plot(ep, val_h,   color="#E07A5F", lw=1.1, label="validation")
    ax.set_xlabel("Epoch", labelpad=2); ax.set_ylabel("Loss", labelpad=2)
    ax.set_yscale("log"); ax.legend(loc="upper right")
    ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout(pad=0.2)
    _save(fig, f"{cell_line}_loss_curves")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════
#  5. Visualisation — ego-graph panels (VIZ_GENES)
# ════════════════════════════════════════════════════════════
def _top_changed_subdf(df, gene, eff_before, eff_after, top_k=10):
    """Ego-graph: edges where gene is source OR target, ranked by |Δeff|."""
    direct = df[(df["source"]==gene)|(df["target"]==gene)].copy()
    if direct.empty: return pd.DataFrame(), set()
    edges  = [(r.source, r.target) for _, r in direct.iterrows()]
    deltas = {e: abs(eff_after.get(e,0)-eff_before.get(e,0)) for e in edges}
    top    = set(sorted(deltas, key=lambda e: deltas[e], reverse=True)[:top_k])
    filt   = direct[direct.apply(lambda r:(r.source,r.target) in top, axis=1)]
    G      = nx.from_pandas_edgelist(filt,"source","target",create_using=nx.DiGraph())
    hop1   = (set(G.successors(gene))|set(G.predecessors(gene)))-{gene}
    return filt, hop1


def _node_hop(n, center, hop1):
    return "center" if n==center else "hop1" if n in hop1 else "hop2"


def _layout(G, gene, hop1):
    hop2   = set(G.nodes)-{gene}-hop1
    shells = [[gene], sorted(hop1&set(G.nodes))]
    if hop2: shells.append(sorted(hop2&set(G.nodes)))
    try:    return nx.shell_layout(G, nlist=[s for s in shells if s])
    except: return nx.kamada_kawai_layout(G)


def _draw_network(G, gene, hop1, *, sign_map, edge_color_map,
                  edge_width_map, edge_label_map,
                  node_sizes_override=None, panel_label, stem):
    pos   = _layout(G, gene, hop1)
    nodes = list(G.nodes())
    sizes = (node_sizes_override if node_sizes_override is not None
             else [NODE_S[_node_hop(n,gene,hop1)] for n in nodes])
    smap  = dict(zip(nodes, sizes))
    edges = list(G.edges())
    solid_e  = [e for e in edges if sign_map.get(e,0)!=0]
    dashed_e = [e for e in edges if sign_map.get(e,0)==0]

    fig, ax = plt.subplots(figsize=(COL_1, COL_1*0.95))
    kw = dict(ax=ax, arrows=True, arrowsize=8,
              connectionstyle="arc3,rad=0.10",
              node_size=[smap[n] for n in nodes])
    if solid_e:
        nx.draw_networkx_edges(G, pos, edgelist=solid_e,
                               edge_color=[edge_color_map[e] for e in solid_e],
                               width=[edge_width_map[e] for e in solid_e],
                               style="solid", alpha=0.92, **kw)
    if dashed_e:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_e,
                               edge_color=[edge_color_map[e] for e in dashed_e],
                               width=[edge_width_map[e] for e in dashed_e],
                               style=(0,(3,2)), alpha=0.75, **kw)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                           node_color=[NODE_C[_node_hop(n,gene,hop1)] for n in nodes],
                           node_size=sizes, edgecolors="white",
                           linewidths=0.8, alpha=0.95)
    for txt in nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5,
                                        font_family="Arial",
                                        font_weight="bold").values():
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_map, ax=ax,
                                  font_size=5.5, font_color=EDGE_LBL,
                                  bbox=dict(facecolor="white",edgecolor="none",
                                            alpha=0.75,pad=0.6), rotate=False)
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.axis("off"); ax.margins(0.10)
    fig.tight_layout(pad=0.2)
    _save(fig, stem); plt.show(); plt.close(fig)


def save_before_jpg(df, gene, cell_line, eff_before, eff_after, top_k=10):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: print(f"  {gene}: no changed edges, skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    edges = list(G.edges())
    sm = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    _draw_network(G, gene, hop1,
                  sign_map      ={e: sm.get(e,0) for e in edges},
                  edge_color_map={e: _before_edge_color(sm.get(e,0)) for e in edges},
                  edge_width_map={e: 1.4 for e in edges},
                  edge_label_map={e: f"{eff_before.get(e,0.0):+.2f}" for e in edges},
                  panel_label="a", stem=f"{cell_line}_{gene}_before")


def save_after_jpg(df, gene, cell_line, eff_before, eff_after, model, top_k=10):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: print(f"  {gene}: no changed edges, skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    V_map = {g: float(v) for g,v in zip(model.gene_names, model.V.detach().cpu().numpy())}
    V_max = max(V_map.values())+1e-9
    edges = list(G.edges())
    em    = {e: eff_after.get(e,0.0) for e in edges}
    e_max = max(abs(v) for v in em.values())+1e-9
    sm = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    ns = [380 if _node_hop(n,gene,hop1)=="center"
          else 80+320*V_map.get(n,1.0)/V_max for n in G.nodes()]
    _draw_network(G, gene, hop1,
                  sign_map           ={e: sm.get(e,0) for e in edges},
                  edge_color_map     ={e: _eff_color(em[e]) for e in edges},
                  edge_width_map     ={e: 0.6+3.0*abs(em[e])/e_max for e in edges},
                  edge_label_map     ={e: f"{em[e]:+.2f}" for e in edges},
                  node_sizes_override=ns,
                  panel_label="b", stem=f"{cell_line}_{gene}_after")


def save_changed_jpg(df, gene, cell_line, eff_before, eff_after, top_k=10):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: print(f"  {gene}: no changed edges, skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    edges = list(G.edges())
    delta = {e: eff_after.get(e,0)-eff_before.get(e,0) for e in edges}
    d_max = max(abs(v) for v in delta.values())+1e-9
    sm = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    _draw_network(G, gene, hop1,
                  sign_map      ={e: sm.get(e,0) for e in edges},
                  edge_color_map={e: _eff_color(eff_after.get(e,0)) for e in edges},
                  edge_width_map={e: 0.6+3.6*abs(delta[e])/d_max for e in edges},
                  edge_label_map={e: f"Δ{delta[e]:+.2f}" for e in edges},
                  panel_label="c", stem=f"{cell_line}_{gene}_changed")


def save_weight_heatmap(eff_weights, label, cell_line, max_label=60):
    if not eff_weights: print("  ⚠ Empty weight map."); return
    rec = [(s,t,w) for (s,t),w in eff_weights.items()]
    df  = pd.DataFrame(rec, columns=["source","target","weight"])
    src = sorted(df["source"].unique()); tgt = sorted(df["target"].unique())
    M   = (df.pivot_table(index="source",columns="target",
                           values="weight",aggfunc="first")
              .reindex(index=src,columns=tgt).fillna(0.0).values)
    ns, nt = len(src), len(tgt)
    vmax = float(np.quantile(np.abs(M),0.98)) or 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    w = min(COL_2, max(COL_1, 0.045*nt+1.4))
    h = min(COL_2, max(COL_1, 0.045*ns+1.0))
    fig, ax = plt.subplots(figsize=(w,h))
    im = ax.imshow(M, aspect="auto", cmap=CMAP_DIV, norm=norm,
                   interpolation="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, aspect=18, pad=0.015)
    cb.set_label(r"$\hat{w}\!\cdot\!\varphi(x_0)$", fontsize=6.5, labelpad=4)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2)
    thr = max_label
    if nt<=thr: ax.set_xticks(range(nt)); ax.set_xticklabels(tgt,rotation=90,fontsize=max(4,380//nt))
    else:       ax.set_xticks([]); ax.text(0.5,-0.04,f"{nt} target genes",transform=ax.transAxes,ha="center",va="top",fontsize=6)
    if ns<=thr: ax.set_yticks(range(ns)); ax.set_yticklabels(src,fontsize=max(4,380//ns))
    else:       ax.set_yticks([]); ax.text(-0.04,0.5,f"{ns} source genes",transform=ax.transAxes,ha="right",va="center",rotation=90,fontsize=6)
    ax.set_xlabel("Target gene",labelpad=3); ax.set_ylabel("Source gene",labelpad=3)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    fig.tight_layout(pad=0.3)
    _save(fig, f"{cell_line}_heatmap_{label}")
    plt.show(); plt.close(fig)
    print(f"  ({ns}×{nt})")


# ════════════════════════════════════════════════════════════
#  6. Target-GRN panels — Cell Press format (GATA1 / MITF)
# ════════════════════════════════════════════════════════════
def _target_subdf(df, gene, rank_eff_map, top_k=30):
    """Outgoing edges gene→targets, ranked by |rank_eff_map|, top-K."""
    sub = df[df["source"]==gene].copy()
    if sub.empty: return pd.DataFrame()
    edges  = [(r.source,r.target) for _,r in sub.iterrows()]
    mag    = {e: abs(rank_eff_map.get(e,0.0)) for e in edges}
    top    = set(sorted(mag, key=lambda e: mag[e], reverse=True)[:top_k])
    return sub[sub.apply(lambda r:(r.source,r.target) in top, axis=1)]


def _radial_layout(G, center):
    others = sorted([n for n in G.nodes if n!=center])
    pos = {center: np.array([0.0,0.0])}
    n   = max(len(others),1)
    for i,node in enumerate(others):
        θ = 2*np.pi*i/n - np.pi/2
        pos[node] = np.array([np.cos(θ), np.sin(θ)])
    return pos


def save_target_plot(df, gene, cell_line, eff_map, rank_eff_map,
                     mode, top_k=30, panel_letter="a"):
    """
    One Cell Press panel (4.49 in square): gene at centre, targets radially.
    eff_map      : EFF_BEFORE or EFF_AFTER  — values displayed
    rank_eff_map : EFF_AFTER always         — edge selection & ranking
    mode         : "before" | "after"
    """
    sub = _target_subdf(df, gene, rank_eff_map, top_k=top_k)
    if sub.empty: print(f"  {gene}: no outgoing edges in GRN — skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    n_tgt = G.number_of_nodes()-1
    pos   = _radial_layout(G, gene)
    edges = list(G.edges())
    sm    = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev    = {e: eff_map.get(e,0.0) for e in edges}
    e_max = max(abs(v) for v in ev.values() if v)+1e-9

    ecolors = [_before_edge_color(sm.get(e,0)) if mode=="before"
               else _eff_color(ev[e]) for e in edges]
    ewidths = [0.4+2.6*abs(ev[e])/e_max for e in edges]
    estyles = ["solid" if sm.get(e,0)!=0 else (0,(4,2)) for e in edges]
    elabels = {e: f"{ev[e]:+.2f}" for e in edges}

    solid_e  = [e for e,s in zip(edges,estyles) if s=="solid"]
    dashed_e = [e for e,s in zip(edges,estyles) if s!="solid"]
    solid_c  = [c for c,s in zip(ecolors,estyles) if s=="solid"]
    dashed_c = [c for c,s in zip(ecolors,estyles) if s!="solid"]
    solid_w  = [w for w,s in zip(ewidths,estyles) if s=="solid"]
    dashed_w = [w for w,s in zip(ewidths,estyles) if s!="solid"]

    nc = [NODE_CTR if n==gene else NODE_H1 for n in G.nodes]
    ns = [520 if n==gene else 240 for n in G.nodes]

    fig, ax = plt.subplots(figsize=(CP_COL15, CP_COL15*0.96))
    kw = dict(ax=ax, arrows=True, arrowsize=7,
              connectionstyle="arc3,rad=0.08", node_size=ns)
    if solid_e:
        nx.draw_networkx_edges(G, pos, edgelist=solid_e,
                               edge_color=solid_c, width=solid_w,
                               style="solid", alpha=0.90, **kw)
    if dashed_e:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_e,
                               edge_color=dashed_c, width=dashed_w,
                               style=(0,(4,2)), alpha=0.72, **kw)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes),
                           node_color=nc, node_size=ns,
                           edgecolors="white", linewidths=0.9, alpha=0.97)
    for txt in nx.draw_networkx_labels(G, pos, ax=ax, font_size=6,
                                        font_family="Arial",
                                        font_weight="bold").values():
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, ax=ax,
                                  font_size=4.5, font_color="#2a2a2a",
                                  rotate=False,
                                  bbox=dict(facecolor="white",edgecolor="none",
                                            alpha=0.68,pad=0.45))
    ax.set_aspect("equal"); ax.margins(0.20); ax.axis("off")
    ax.text(-0.04, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left",
            fontfamily="Arial")
    mode_str = ("initial weights · DB sign" if mode=="before"
                else "post-training weights · learned sign")
    ax.text(0.5, -0.03, f"{n_tgt} targets · {mode_str}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color="#555", fontfamily="Arial")

    sm_obj = mpl.cm.ScalarMappable(
        cmap=CMAP_DIV, norm=TwoSlopeNorm(vmin=-e_max,vcenter=0,vmax=e_max))
    sm_obj.set_array([])
    cax = fig.add_axes([0.20,-0.02,0.60,0.018])
    cb  = fig.colorbar(sm_obj, cax=cax, orientation="horizontal")
    cb.set_label(r"$\hat{w}_{ij}\!\cdot\!\varphi(x_0)$", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([-e_max,0,e_max])
    cb.set_ticklabels([f"−{e_max:.2f}","0",f"+{e_max:.2f}"])

    handles = [
        Line2D([0],[0],color=ACT_C,lw=1.4,label="activation"),
        Line2D([0],[0],color=REP_C,lw=1.4,label="repression"),
        Line2D([0],[0],color=UNK_C,lw=1.4,label="unknown"),
        Line2D([0],[0],color="#555",lw=1.1,ls="solid",label="DB-constrained"),
        Line2D([0],[0],color="#555",lw=1.1,ls=(0,(4,2)),label="sign learned"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=5.5, frameon=False, bbox_to_anchor=(0.5,-0.11),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(rect=[0,0.04,1,1])
    _save(fig, f"{cell_line}_{gene}_{mode}")
    plt.show(); plt.close(fig)
    print(f"  {gene} ({mode}): {n_tgt} targets")


# ── Cytoscape interactive ─────────────────────────────────────────────────────
_CDN_LOADED = False
def render_cytoscape_gene(html_body):
    global _CDN_LOADED
    cdn = ""
    if not _CDN_LOADED:
        cdn = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/'
               'cytoscape/3.28.1/cytoscape.min.js"></script>')
        _CDN_LOADED = True
    display(HTML(f"{cdn}<div style='margin-bottom:28px'>{html_body}</div>"))

def _cy_html(gene, elements_json, cy_id, title, height=480):
    return f"""
<div style="font-family:Arial,sans-serif;font-size:12px;font-weight:bold;
            margin-bottom:2px">{title}</div>
<div id="{cy_id}" style="width:100%;height:{height}px;border:1px solid #ddd;
     border-radius:6px;background:#fafafa"></div>
<script>
(function(){{
  var cy=cytoscape({{container:document.getElementById('{cy_id}'),
    elements:{elements_json},
    layout:{{name:'cose',animate:false,nodeRepulsion:6000,idealEdgeLength:80}},
    style:[
      {{selector:'node',style:{{'label':'data(label)','font-size':9,
        'font-family':'Arial','text-valign':'center','text-halign':'center',
        'width':'data(size)','height':'data(size)','background-color':'data(color)'}}}},
      {{selector:'node[hop="center"]',style:{{'font-size':12,'font-weight':'bold'}}}},
      {{selector:'edge',style:{{'line-color':'data(color)',
        'target-arrow-color':'data(color)','target-arrow-shape':'triangle',
        'curve-style':'bezier','line-style':'data(linestyle)',
        'width':'data(width)','label':'data(label)','font-size':7,
        'font-family':'Arial','text-rotation':'autorotate',
        'color':'#333','opacity':0.9}}}},
    ],wheelSensitivity:0.3}});
}})();
</script>"""

def _before_cy(df, gene, eff_before, eff_after, top_k=10, height=480):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: return f"<p><b>{gene}</b>: not in GRN.</p>"
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    sm = {(r.source,r.target):int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev_b = {(r.source,r.target):eff_before.get((r.source,r.target),0.0) for _,r in sub.iterrows()}
    me   = max(abs(v) for v in ev_b.values())+1e-9
    nodes=[{"data":{"id":n,"label":n,"hop":_node_hop(n,gene,hop1),
                    "color":NODE_C[_node_hop(n,gene,hop1)],
                    "size":{"center":44,"hop1":30,"hop2":22}[_node_hop(n,gene,hop1)]}}
           for n in G.nodes]
    edges=[{"data":{"id":f"{r.source}__{r.target}","source":r.source,"target":r.target,
            "color":_before_edge_color(sm.get((r.source,r.target),0)),
            "linestyle":"solid" if sm.get((r.source,r.target),0)!=0 else "dashed",
            "width":round(1+4*abs(ev_b.get((r.source,r.target),0))/me,2),
            "label":f"{ev_b.get((r.source,r.target),0):+.2f}"}}
           for _,r in sub.iterrows()]
    return _cy_html(gene,json.dumps(nodes+edges),f"cy_{gene}_pre",
                    f"{gene} — before (solid=DB · dashed=learned · label=init eff)",height)

def _after_cy(df, gene, eff_before, eff_after, model, top_k=10, height=480):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: return f"<p><b>{gene}</b>: not in GRN.</p>"
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    Vm = {g:float(v) for g,v in zip(model.gene_names, model.V.detach().cpu().numpy())}
    Vmx= max(Vm.values())+1e-9
    sm = {(r.source,r.target):int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev_a={e:eff_after.get(e,0.0) for e in [(r.source,r.target) for _,r in sub.iterrows()]}
    me  = max(abs(v) for v in ev_a.values())+1e-9
    nodes=[{"data":{"id":n,"label":n,"hop":_node_hop(n,gene,hop1),
                    "color":NODE_C[_node_hop(n,gene,hop1)],
                    "size":round(20+32*Vm.get(n,1)/Vmx,1)}} for n in G.nodes]
    edges=[{"data":{"id":f"{r.source}__{r.target}","source":r.source,"target":r.target,
            "color":_eff_color(ev_a.get((r.source,r.target),0)),
            "linestyle":"solid" if sm.get((r.source,r.target),0)!=0 else "dashed",
            "width":round(1+7*abs(ev_a.get((r.source,r.target),0))/me,2),
            "label":f"{ev_a.get((r.source,r.target),0):+.2f}"}}
           for _,r in sub.iterrows()]
    return _cy_html(gene,json.dumps(nodes+edges),f"cy_{gene}_post",
                    f"{gene} — after (solid=DB · dashed=learned · label=eff)",height)

def _target_cy(df, gene, eff_map, rank_eff_map,
               mode="after", top_k=30, height=520):
    sub = _target_subdf(df, gene, rank_eff_map, top_k=top_k)
    if sub.empty: return f"<p><b>{gene}</b>: no outgoing edges.</p>"
    G  = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    sm = {(r.source,r.target):int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev = {(r.source,r.target):eff_map.get((r.source,r.target),0.0) for _,r in sub.iterrows()}
    me = max(abs(v) for v in ev.values())+1e-9
    nodes=[{"data":{"id":n,"label":n,
                    "color":NODE_CTR if n==gene else NODE_H1,
                    "size":46 if n==gene else 26}} for n in G.nodes]
    edges=[{"data":{"id":f"{r.source}__{r.target}","source":r.source,"target":r.target,
            "color":(_before_edge_color(sm.get((r.source,r.target),0)) if mode=="before"
                     else _eff_color(ev.get((r.source,r.target),0))),
            "linestyle":"solid" if sm.get((r.source,r.target),0)!=0 else "dashed",
            "width":round(1+5*abs(ev.get((r.source,r.target),0))/me,2),
            "label":f"{ev.get((r.source,r.target),0):+.2f}"}}
           for _,r in sub.iterrows()]
    n_tgt = G.number_of_nodes()-1
    ml = "initial eff·DB sign" if mode=="before" else "post-training eff·learned sign"
    elements=json.dumps(nodes+edges)
    return f"""
<div style="font-family:Arial,sans-serif;font-size:12px;font-weight:bold;
            margin-bottom:3px">{gene} — {ml} · {n_tgt} targets</div>
<div id="cy_{gene}_{mode}_tgt" style="width:100%;height:{height}px;
     border:1px solid #ddd;border-radius:5px;background:#fafafa"></div>
<script>
(function(){{var cy=cytoscape({{container:document.getElementById('cy_{gene}_{mode}_tgt'),
  elements:{elements},
  layout:{{name:'concentric',animate:false,
           concentric:function(n){{return n.data('id')==='{gene}'?2:1;}},
           levelWidth:function(){{return 1;}},minNodeSpacing:26}},
  style:[
    {{selector:'node',style:{{'label':'data(label)','font-size':9,'font-family':'Arial',
      'text-valign':'center','text-halign':'center','width':'data(size)','height':'data(size)',
      'background-color':'data(color)','text-outline-width':1.5,
      'text-outline-color':'#fff','color':'#111'}}}},
    {{selector:'node[id="{gene}"]',style:{{'font-size':12,'font-weight':'bold'}}}},
    {{selector:'edge',style:{{'line-color':'data(color)','target-arrow-color':'data(color)',
      'target-arrow-shape':'triangle','curve-style':'bezier',
      'line-style':'data(linestyle)','width':'data(width)','label':'data(label)',
      'font-size':6.5,'font-family':'Arial','text-rotation':'autorotate',
      'color':'#222','opacity':0.9}}}},
  ],wheelSensitivity:0.3}});}})();
</script>"""


# ════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_VIZ = 10     # ego-graph edges for VIZ_GENES
TOP_K_TGT = 30     # target edges for GATA1 / MITF (raise if < 10 targets appear)

# Build dataset (x0 = logmean of control cells — consistent with GRNN init)
dataset = PerturbseqDataset(adata, GENE_NAMES, PERTURBATION_COL,
                             CONTROL_LABEL, min_cells=5, ensg2sym=ensg2sym)
x0_numpy = dataset.x0.numpy()

model, GRN_DF = grn_tsv_to_grnn(OUT_FILE, GENE_NAMES, x0_numpy)
train_ds, val_ds, test_ds = split_dataset(dataset)

n_unk = int((model.signs==0).sum())
print(f"Model  : {model.N} genes | {model.E} edges "
      f"({n_unk} sign=0/learned, {model.E-n_unk} DB-constrained) | "
      f"{sum(p.numel() for p in model.parameters()):,} params")
print(f"Dataset: {len(dataset)} → train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")
print(f"Device : {DEVICE}")

EFF_BEFORE = model._eff_map()

print("\n=== Heatmap — before ===")
save_weight_heatmap(EFF_BEFORE, "before", CELL_LINE)

print("\n=== Ego-graphs — before ===")
_CDN_LOADED = False
for gene in VIZ_GENES:
    print(f"\n── {gene} ──")
    render_cytoscape_gene(_before_cy(GRN_DF, gene, EFF_BEFORE, EFF_BEFORE, TOP_K_VIZ))
    save_before_jpg(GRN_DF, gene, CELL_LINE, EFF_BEFORE, EFF_BEFORE, TOP_K_VIZ)

# ── Train ─────────────────────────────────────────────────────────────────────
train_history, val_history = train_grnn(
    model, train_ds, val_ds, n_epochs=50, lr=1e-3, top_k_deg=20,
    lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0, lam_deg=2.0,
    lam_balance=0.5, gamma_focus=2.0, device=DEVICE)

plot_loss_curves(train_history, val_history, CELL_LINE)

EFF_AFTER = model._eff_map()

print("\n=== Heatmap — after ===")
save_weight_heatmap(EFF_AFTER, "after", CELL_LINE)

print("\n=== Ego-graphs — after + changed ===")
_CDN_LOADED = False
for gene in VIZ_GENES:
    print(f"\n── {gene} ──")
    render_cytoscape_gene(_after_cy(GRN_DF, gene, EFF_BEFORE, EFF_AFTER, model, TOP_K_VIZ))
    save_after_jpg(GRN_DF, gene, CELL_LINE, EFF_BEFORE, EFF_AFTER, model, TOP_K_VIZ)
    save_changed_jpg(GRN_DF, gene, CELL_LINE, EFF_BEFORE, EFF_AFTER, TOP_K_VIZ)

# ── Convergence diagnostics ───────────────────────────────────────────────────
x0_dev = dataset.x0.to(DEVICE)
mi, fn = _convergence_report(model, test_ds, x0_dev, DEVICE)
print(f"\nConvergence (test): median={mi:.0f} iters | non-converged={100*fn:.1f}%")
if fn > 0.05:
    print("  ⚠  >5% non-converged — consider increasing max_iter.")

# ── Target GRN: GATA1 and MITF (4 individual panels) ─────────────────────────
print("\n=== Target GRN plots ===")

print("\n── GATA1 before ──")
render_cytoscape_gene(_target_cy(GRN_DF,"GATA1",EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"GATA1",CELL_LINE,EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT,panel_letter="a")

print("\n── GATA1 after ──")
render_cytoscape_gene(_target_cy(GRN_DF,"GATA1",EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"GATA1",CELL_LINE,EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT,panel_letter="b")

print("\n── MITF before ──")
render_cytoscape_gene(_target_cy(GRN_DF,"MITF",EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"MITF",CELL_LINE,EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT,panel_letter="c")

print("\n── MITF after ──")
render_cytoscape_gene(_target_cy(GRN_DF,"MITF",EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"MITF",CELL_LINE,EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT,panel_letter="d")

print(f"\n✓ All figures in {PLOT_DIR}/")
for pat in ["*.png","*.pdf"]:
    files = sorted(f.name for f in PLOT_DIR.glob(pat))
    if files: print(f"  {pat}: {len(files)} files")

# ════════════════════════════════════════════════════════════
#  Train ONCE, then random-subsample evaluation 10 times
# ════════════════════════════════════════════════════════════
# ────────────────────────────────────────────────────────────
# CELL 11 — evaluate_all()
#
# Paste this cell BEFORE the training/evaluation cell.
#
# Returns a flat dict with keys consumed by the eval loop:
#   pearson_delta{k}_mean        — mean Pearson r on top-k DEGs
#   pearson_delta_all_mean       — mean Pearson r on all genes
#   centroid_accuracy            — fraction of samples where predicted
#                                  Δ centroid direction matches observed
#   mse_delta                    — MSE on Δ = x_pred − x0 vs x_obs − x0
#   directional_accuracy         — per-gene sign accuracy on genes with
#                                  |LFC| ≥ lfc_threshold
#   directional_lfc_threshold    — the threshold used (echoed for logging)
#   directional_n_gene_pert_pairs— number of (gene, pert) pairs evaluated
# ────────────────────────────────────────────────────────────

import torch
import numpy as np
from torch.utils.data import DataLoader


def evaluate_all(
    model,
    dataset,
    top_k_pearson: int = 20,
    lfc_threshold: float = 0.1,
    device: str = "cpu",
    batch_size: int = 1,
) -> dict:
    """
    Evaluate GRNN on a dataset (or Subset) and return all Systema metrics.

    Parameters
    ----------
    model           : trained GRNN instance
    dataset         : PerturbseqDataset or torch.utils.data.Subset thereof
    top_k_pearson   : k for Pearson Δ-top-k metric
    lfc_threshold   : |Δ| threshold to include a gene in directional accuracy
    device          : "cuda" or "cpu"

    Returns
    -------
    Flat dict of scalar metrics (NaN where undefined).
    """
    model.eval()
    model.to(device)

    # Resolve x0 — works for Subset (wraps the underlying dataset)
    base_ds = dataset
    while not hasattr(base_ds, "x0") and hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset      # unwrap nested Subset(s) to reach the base dataset
    x0 = base_ds.x0.to(device)

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    # Accumulators
    pearson_topk_list  = []   # per-sample Pearson r on top-k DEGs
    pearson_all_list   = []   # per-sample Pearson r on all genes
    centroid_correct   = 0    # samples where sign(mean Δ_pred) == sign(mean Δ_obs)
    mse_delta_list     = []   # per-sample MSE on Δ
    dir_correct        = 0    # gene-pert pairs with correct sign
    dir_total          = 0    # gene-pert pairs above LFC threshold

    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"][0].to(device)          # (N_genes,)
            pidx  = int(batch["perturbed_idx"][0])
            pval  = float(batch["perturbed_value"][0])

            x_pred, _ = model(x0, perturbed_idx=pidx, perturbed_value=pval)

            d_obs  = (x_obs  - x0).cpu().numpy()          # (N_genes,)
            d_pred = (x_pred - x0).cpu().numpy()

            # ── (a) Pearson Δ — all genes ─────────────────────────────────
            if d_obs.std() > 1e-8 and d_pred.std() > 1e-8:
                r_all = float(np.corrcoef(d_pred, d_obs)[0, 1])
            else:
                r_all = float("nan")
            pearson_all_list.append(r_all)

            # ── (a) Pearson Δ — top-k by |d_obs| ─────────────────────────
            k = min(top_k_pearson, len(d_obs))
            topk_idx = np.argpartition(np.abs(d_obs), -k)[-k:]
            d_obs_k  = d_obs[topk_idx]
            d_pred_k = d_pred[topk_idx]
            if d_obs_k.std() > 1e-8 and d_pred_k.std() > 1e-8:
                r_k = float(np.corrcoef(d_pred_k, d_obs_k)[0, 1])
            else:
                r_k = float("nan")
            pearson_topk_list.append(r_k)

            # ── (b) Centroid accuracy ─────────────────────────────────────
            # Correct if the mean direction of Δ_pred matches Δ_obs.
            # Ties (mean == 0) counted as incorrect (conservative).
            if np.sign(d_pred.mean()) == np.sign(d_obs.mean()) and d_obs.mean() != 0:
                centroid_correct += 1

            # ── (c) MSE on Δ ──────────────────────────────────────────────
            mse_delta_list.append(float(np.mean((d_pred - d_obs) ** 2)))

            # ── (d) Directional accuracy ──────────────────────────────────
            mask = np.abs(d_obs) >= lfc_threshold
            if mask.sum() > 0:
                dir_correct += int((np.sign(d_pred[mask]) == np.sign(d_obs[mask])).sum())
                dir_total   += int(mask.sum())

    n = len(pearson_topk_list)
    if n == 0:
        nan = float("nan")
        return {
            f"pearson_delta{top_k_pearson}_mean": nan,
            "pearson_delta_all_mean":             nan,
            "centroid_accuracy":                  nan,
            "mse_delta":                          nan,
            "directional_accuracy":               nan,
            "directional_lfc_threshold":          lfc_threshold,
            "directional_n_gene_pert_pairs":      0,
        }

    def _nanmean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan")

    return {
        f"pearson_delta{top_k_pearson}_mean": _nanmean(pearson_topk_list),
        "pearson_delta_all_mean":             _nanmean(pearson_all_list),
        "centroid_accuracy":                  centroid_correct / n,
        "mse_delta":                          float(np.mean(mse_delta_list)),
        "directional_accuracy":               (dir_correct / dir_total
                                               if dir_total > 0 else float("nan")),
        "directional_lfc_threshold":          lfc_threshold,
        "directional_n_gene_pert_pairs":      dir_total,
    }
N_EVAL_RUNS      = 10
SUBSAMPLE_FRAC   = 0.30   # evaluate on 30% of test set each run
TOP_K_PEARSON    = 20
LFC_THRESHOLD    = 0.1
EVAL_SEEDS       = list(range(N_EVAL_RUNS))

# --- split once ---
torch.manual_seed(0); np.random.seed(0)
train_ds, val_ds, test_ds = split_dataset(dataset, train_frac=0.70, val_frac=0.10, seed=0)

# --- init model once ---
torch.manual_seed(0)
model, _ = grn_tsv_to_grnn(OUT_FILE, GENE_NAMES, x0_numpy)

# --- train once ---
train_grnn(model, train_ds, val_ds,
           n_epochs=50, lr=1e-3,
           top_k_deg=20, lam_deg=2.0, lam_delta=1.0,
           device=DEVICE)

# ════════════════════════════════════════════════════════════
#  Random subsample evaluation loop
# ════════════════════════════════════════════════════════════
all_metrics = []

for seed in EVAL_SEEDS:
    print(f"\n── Eval run {seed+1}/{N_EVAL_RUNS}  (seed={seed}) ──")

    rng = np.random.default_rng(seed)
    P = len(test_ds)
    m = int(SUBSAMPLE_FRAC * P)

    idx = rng.choice(P, size=m, replace=False)
    test_sub = torch.utils.data.Subset(test_ds, idx)

    # evaluate on random subset
    metrics = evaluate_all(model, test_sub,
                           top_k_pearson=TOP_K_PEARSON,
                           lfc_threshold=LFC_THRESHOLD,
                           device=DEVICE)

    metrics["seed"] = seed
    metrics["n_eval_samples"] = m
    all_metrics.append(metrics)

    print(f"  Pearson Δ{TOP_K_PEARSON}={metrics[f'pearson_delta{TOP_K_PEARSON}_mean']:.4f}  "
          f"centroid={metrics['centroid_accuracy']:.4f}  "
          f"MSE={metrics['mse_delta']:.5f}  "
          f"dir={metrics['directional_accuracy']:.4f}")


# ════════════════════════════════════════════════════════════
#  Aggregate mean ± std across evaluation runs
# ════════════════════════════════════════════════════════════
k = TOP_K_PEARSON
metric_keys = [
    (f"pearson_delta{k}_mean",  f"(a) Pearson Δ{k}  "),
    ("pearson_delta_all_mean",  "(a) Pearson Δ-all"),
    ("centroid_accuracy",       "(b) Centroid acc  "),
    ("mse_delta",               "(c) MSE (delta)   "),
    ("directional_accuracy",    "(d) Directional   "),
]

print(f"\n{'='*60}")
print(f"  iPerturb — {CELL_LINE}  ({N_EVAL_RUNS} random subsample eval runs)")
print(f"  Subsample fraction = {SUBSAMPLE_FRAC:.2f}")
print(f"{'='*60}")

for key, label in metric_keys:
    vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
    print(f"  {label} : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# show threshold info
last = all_metrics[-1]
print(f"\n  Directional threshold : {last['directional_lfc_threshold']}")
print(f"  Gene-pert pairs (last): {last['directional_n_gene_pert_pairs']:,}")

# save results
import pandas as _pd
results_df = _pd.DataFrame(all_metrics)
results_path = f"/content/{CELL_LINE}_metrics_subsample_{N_EVAL_RUNS}runs.tsv"
results_df.to_csv(results_path, sep="\t", index=False)
print(f"\n✓ Per-run metrics saved → {results_path}")

# ===== Lambda ablation (Phase F): drop-one-term + lambda-5 sweep on K562 =====
# Opt-in via env IPERTURB_ABLATION=1. Reuses the subsample-eval setup and the SAME
# initial weights (1,1,1,0.5,2) as the baseline ("full"); this only measures each
# term's contribution + lambda_5 robustness. Writes K562_ablation_lambda.tsv, then
# exits before RPE1 (run a normal pass for the main results).
if os.environ.get("IPERTURB_ABLATION"):
    import sys as _sys, pandas as _pd_abl
    ABL_SEEDS = [0, 1]   # smaller study (CPU-feasible); 2 seeds
    BASE = dict(lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0, lam_balance=0.5, lam_deg=2.0)
    CONFIGS = [("full", {}), ("-WMSE", {"lam_wmse": 0.0}), ("-AFDA", {"lam_afda": 0.0}),
               ("-bal", {"lam_balance": 0.0}), ("-Delta", {"lam_delta": 0.0}),
               ("-DEG", {"lam_deg": 0.0}), ("l5=0.5", {"lam_deg": 0.5}),
               ("l5=1", {"lam_deg": 1.0}), ("l5=4", {"lam_deg": 4.0})]
    def _abl_one(weights, seed):
        tr, va, te = split_dataset(dataset, train_frac=0.70, val_frac=0.10, seed=0)
        torch.manual_seed(seed); np.random.seed(seed)
        mdl, _ = grn_tsv_to_grnn(OUT_FILE, GENE_NAMES, x0_numpy)
        train_grnn(mdl, tr, va, n_epochs=30, lr=1e-3, top_k_deg=20, device=DEVICE, **weights)
        r = evaluate_all(mdl, te, top_k_pearson=20, lfc_threshold=0.1, device=DEVICE)
        return r["directional_accuracy"], r["mse_delta"]
    _abl_rows = []
    for _name, _ov in CONFIGS:
        _w = {**BASE, **_ov}; _da, _ms = [], []
        for _s in ABL_SEEDS:
            a, mse = _abl_one(_w, _s); _da.append(a); _ms.append(mse)
            print(f"[ablation] {_name:7} seed={_s} dir={a:.3f} mse={mse:.4f}", flush=True)
        _abl_rows.append(dict(config=_name,
            dir_acc_mean=float(np.mean(_da)), dir_acc_sd=float(np.std(_da)),
            mse_mean=float(np.mean(_ms)), mse_sd=float(np.std(_ms)), **_w))
    _pd_abl.DataFrame(_abl_rows).to_csv("/content/K562_ablation_lambda.tsv", sep="\t", index=False)
    print("✓ Ablation saved → /content/K562_ablation_lambda.tsv", flush=True)
    _sys.exit(0)

import scanpy as sc

DATA_PATH = "/content/RPE1.h5ad"
adata = sc.read_h5ad(DATA_PATH)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

hvg_list = adata.var_names[adata.var["highly_variable"]].tolist()

print(f"Number of HVGs selected: {len(hvg_list)}")
print(hvg_list)

################################################################################
# CELL 0b — HVG gene list + cell-line marker check
# Insert after the existing Cell 0 (HVG selection)
################################################################################
import os

SYMBOL_COL = "gene_name"
CELL_LINE  = "RPE1"          # switch to "RPE1" for the other dataset

hvg_symbols = (adata.var
               .loc[adata.var["highly_variable"], SYMBOL_COL]
               .dropna()
               .tolist())

print(f"HVGs: {len(hvg_symbols)}")

with open("/content/gene_list.txt", "w") as f:
    f.write("\n".join(hvg_symbols))
print("✓ /content/gene_list.txt")

# ── Cell-line marker gene lists ───────────────────────────────
MARKER_GENES = {
    "K562": [
        # --- Original HVG-present markers ---
        "HBZ",      # Hemoglobin zeta
        "GATA1",    # Master erythroid TF
        "KLF1",     # Erythroid-specific TF
        "GFI1B",    # Hematopoietic repressor TF
        "GYPA",     # Erythrocyte membrane marker
        "NFE2",     # Erythroid regulatory factor

        # --- Replacements for missing HVG markers (ABL1, HBB, HBE1, CRKL) ---
        "ALAS2",    # Heme synthesis, strong erythroid marker
        "TFRC",     # CD71, classic K562 / erythroid proliferation marker
        "HBA1",   # Band 3, erythroid membrane marker
        "MYC",     # Alpha hemoglobin stabilizing protein
    ],
    "RPE1": [
    # --- Already passing HVG filter ---
    "MYC",      # Keep — passes ✓
    "MITF",     # Keep — passes ✓ (master RPE/melanocyte TF)
    "OTX2",     # Keep — passes ✓ (RPE developmental TF)
    "CTSD",     # Keep — passes ✓ (lysosomal phagocytosis)
    "ITGAV",    # Keep — passes ✓ (phagocytosis integrin αV)

    # --- Replacements for EHF, ELF3, DCT, LRAT, CDK2AP2 ---
    "ID3",      # Macular RPE TF marker — scRNA-seq confirmed variable
    "CRYAB",    # Peripheral RPE sHSP — stress-responsive, high dispersion
    "IGFBP5",   # Top DE gene in hRPE scRNA-seq — highly dynamic
    "CTGF",     # Macular RPE TGF-β target — transcriptionally variable
    "FST",      # Macular RPE BMP antagonist — stress-responsive
],
}

hvg_set    = set(hvg_symbols)
candidates = MARKER_GENES[CELL_LINE]

print(f"\n=== {CELL_LINE} marker genes in HVG list ===")
found, missing = [], []
for gene in candidates:
    if gene in hvg_set:
        found.append(gene)
        print(f"  ✓  {gene}")
    else:
        print(f"  ✗  {gene}  ← not in HVG set")
        missing.append(gene)

print(f"\nFound : {len(found)} / {len(candidates)}")
print(f"Missing: {missing}")

# Genes to visualise after Cell 10
VIZ_GENES = found   # only genes actually in the network

# ────────────────────────────────────────────────────────────
# CELL 2 — Configuration  ← only edit this cell
# ────────────────────────────────────────────────────────────
import os

# Gene list produced by the HVG cell (one HGNC symbol per line)
GENE_LIST_FILE = "/content/gene_list.txt"

CACHE_DIR  = "/content/grn_cache"
OUT_FILE   = "/content/grn_edges.tsv"

TARGET_EDGES     = 4_600
GREEDY_REWARD    = 0.15
STRING_MIN_SCORE = 700   # 400=permissive, 700=high, 900=very high
COEX_TOPN        = 5     # top-N co-expressed partners per gene

# GeneHancer local files (already downloaded to /content/)
GH_GFF_PATH    = "/content/GeneHancer_v5.26.gff"
GH_TFBS_PATH   = "/content/GeneHancer_TFBSs_v5.26.txt"
GH_TISSUE_PATH = "/content/GeneHancer_Tissues_v5.26.txt"

# COXPRESdb token (free at coxpresdb.jp — leave "" to skip auth)
COXPRESDB_TOKEN = ""

# Filter GeneHancer to K562-active enhancers only (recommended)
TISSUE_FILTER = "RPE1"   # substring match in tissue name; "" = no filter

LEVEL_CONF = {1: 1.0, 2: 0.60, 3: 0.35, 4: 0.20}

# Set to True if COXPRESdb consistently times out from your Colab region
SKIP_COXPRESDB = False

# ────────────────────────────────────────────────────────────
# CELL 3 — Imports & session
# ────────────────────────────────────────────────────────────
# dependencies are installed from requirements.txt (see the Colab notebook)

import logging, time, re, warnings
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress noisy library/network warnings in Colab
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("grn")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (GRN-research)"})
    return s

SESSION = make_session()

def _get(url, params=None, timeout=120):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r

def load_genes() -> set:
    genes = set()
    with open(GENE_LIST_FILE) as f:
        for line in f:
            g = line.strip().split()[0].upper()
            if g:
                genes.add(g)
    log.info("Gene set: %d genes", len(genes))
    return genes

GENE_SET = load_genes()
print(f"✓ {len(GENE_SET)} genes loaded")

# ────────────────────────────────────────────────────────────
# CELL 4 — Level 1a: TRRUST v2
# ────────────────────────────────────────────────────────────
def crawl_trrust(gene_set):
    url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
    log.info("TRRUST → %s", url)
    r = _get(url)
    rows = []
    for line in r.text.strip().split("\n"):
        if line.startswith("#"): continue
        p = line.split("\t")
        if len(p) < 3: continue
        tf, tgt, itype = p[0].upper(), p[1].upper(), p[2]
        if tf not in gene_set or tgt not in gene_set: continue
        sign = 1 if "activation" in itype.lower() else (-1 if "repression" in itype.lower() else 0)
        rows.append({"source": tf, "target": tgt, "sign": sign, "level": 1, "db": "TRRUST"})
    df = pd.DataFrame(rows)
    log.info("  → %d edges", len(df))
    return df

_p = Path(CACHE_DIR)/"l1_trrust.tsv"
L1_TRRUST = pd.read_csv(_p, sep="\t") if _p.exists() else crawl_trrust(GENE_SET)
if not _p.exists(): L1_TRRUST.to_csv(_p, sep="\t", index=False)
print(f"TRRUST: {len(L1_TRRUST)} edges")

# ────────────────────────────────────────────────────────────
# CELL 5 — Level 1b: OmniPath + CollecTRI
# ────────────────────────────────────────────────────────────
def _omnipath_via_lib(gene_set: set, datasets: list[str], label: str) -> pd.DataFrame:
    import omnipath as op
    rows = []
    for ds in datasets:
        try:
            log.info("  omnipath lib: dataset=%s", ds)
            if ds == "collectri":
                df_raw = op.interactions.CollecTRI.get(dorothea_levels=None, genesymbols=True)
            elif ds == "tf_target":
                df_raw = op.interactions.TFtarget.get(genesymbols=True)
            elif ds == "dorothea":
                df_raw = op.interactions.Dorothea.get(dorothea_levels=["A","B"], genesymbols=True)
            else:
                df_raw = op.interactions.AllInteractions.get(datasets=[ds], genesymbols=True)

            for _, row in df_raw.iterrows():
                src = str(row.get("source_genesymbol","")).upper()
                tgt = str(row.get("target_genesymbol","")).upper()
                if src not in gene_set or tgt not in gene_set: continue
                try:
                    sign = int(row.get("consensus_stimulation") or 0) - int(row.get("consensus_inhibition") or 0)
                except: sign = 0
                rows.append({"source": src, "target": tgt, "sign": sign, "level": 1, "db": label})
        except Exception as e:
            log.debug("  omnipath lib %s failed: %s", ds, e)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

def _omnipath_via_http(gene_set: set, datasets: str, label: str) -> pd.DataFrame:
    rows = []
    for ds in datasets.split(","):
        ds = ds.strip()
        for attempt in range(3):
            try:
                time.sleep(2 ** attempt)
                r = SESSION.get("https://omnipathdb.org/interactions",
                    params={"datasets": ds, "genesymbols": "1", "fields": "type,sources,sign", "organism": "9606"},
                    timeout=180)
                r.raise_for_status()
                lines  = r.text.strip().split("\n")
                header = lines[0].split("\t")
                for line in lines[1:]:
                    p = line.split("\t")
                    if len(p) < len(header): continue
                    d   = dict(zip(header, p))
                    src = d.get("source_genesymbol","").upper()
                    tgt = d.get("target_genesymbol","").upper()
                    if src not in gene_set or tgt not in gene_set: continue
                    try: sign = int(d.get("consensus_stimulation") or 0) - int(d.get("consensus_inhibition") or 0)
                    except: sign = 0
                    rows.append({"source": src, "target": tgt, "sign": sign, "level": 1, "db": label})
                break
            except Exception as e:
                log.debug("  HTTP attempt %d %s: %s", attempt+1, ds, e)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

def _enrichr_fallback(gene_set: set, libraries: list[str], label: str) -> pd.DataFrame:
    import gseapy as gp
    rows = []
    for lib in libraries:
        try:
            log.info("  Enrichr fallback: %s", lib)
            enr = gp.get_library(lib, organism="Human")
            for tf, targets in enr.items():
                tf = tf.split(" ")[0].upper()
                if tf not in gene_set: continue
                for tgt in targets:
                    tgt = tgt.upper()
                    if tgt in gene_set and tgt != tf:
                        rows.append({"source": tf, "target": tgt, "sign": 0, "level": 1, "db": f"Enrichr:{lib}"})
        except Exception as e:
            log.debug("  Enrichr %s: %s", lib, e)
    df = pd.DataFrame(rows).drop_duplicates(["source","target"]) if rows else \
         pd.DataFrame(columns=["source","target","sign","level","db"])
    return df

def crawl_omnipath_robust(gene_set: set, datasets: list[str], label: str, cache_name: str) -> pd.DataFrame:
    _p = Path(CACHE_DIR) / f"l1_{cache_name}.tsv"
    if _p.exists():
        df = pd.read_csv(_p, sep="\t")
        log.info("[cache] %s: %d edges", label, len(df))
        return df

    df = pd.DataFrame(columns=["source","target","sign","level","db"])
    # 1. Library
    try: df = _omnipath_via_lib(gene_set, datasets, label)
    except Exception as e: log.warning("Lib path failed: %s", e)
    # 2. HTTP
    if df.empty:
        log.warning("  lib empty → trying raw HTTP")
        try: df = _omnipath_via_http(gene_set, ",".join(datasets), label)
        except Exception as e: log.warning("HTTP path failed: %s", e)
    # 3. Enrichr
    if df.empty:
        log.warning("  HTTP empty → falling back to Enrichr")
        try: df = _enrichr_fallback(gene_set, ["ChEA_2022", "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X", "TF_Perturbations_Followed_by_Expression"], label)
        except Exception as e: log.warning("Enrichr fallback failed: %s", e)

    if not df.empty: df.to_csv(_p, sep="\t", index=False)
    return df

L1_OMNIPATH  = crawl_omnipath_robust(GENE_SET, ["tf_target", "collectri"], "OmniPath",  "omnipath")
L1_COLLECTRI = crawl_omnipath_robust(GENE_SET, ["collectri"], "CollecTRI", "collectri")
print(f"OmniPath: {len(L1_OMNIPATH)} | CollecTRI: {len(L1_COLLECTRI)}")

# ────────────────────────────────────────────────────────────
# CELL 6 — Level 2: GeneHancer (GFF + TFBS + Tissue filter)
# ────────────────────────────────────────────────────────────
import os
os.makedirs(CACHE_DIR, exist_ok=True)
_l2_cache = Path(CACHE_DIR) / "l2_genehancer.tsv"
if _l2_cache.exists(): _l2_cache.unlink()

def parse_gff_gene_associations(gff_path) -> pd.DataFrame:
    records = []
    _skip = re.compile(r'^(ENSG|ENSM|lnc-|piR-|LOC\d|HSALNG|FAM\d)', re.IGNORECASE)
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9: continue
            if parts[2].lower() != "enhancer": continue
            attr_str = parts[8]
            m = re.search(r'genehancer_id=([^;]+)', attr_str)
            if not m: continue
            gh_id = m.group(1).strip()
            for gm in re.finditer(r'connected_gene=([^;]+);score=([\d.]+)', attr_str):
                gene = gm.group(1).strip().upper()
                if not gene or _skip.match(gene): continue
                try: sc = float(gm.group(2))
                except: sc = 0.0
                records.append({"enhancer_id": gh_id, "gene": gene, "score": sc})
    df = pd.DataFrame(records)
    return df

def parse_tfbs(tfbs_path, tissue_filter: str = "") -> pd.DataFrame:
    df = pd.read_csv(tfbs_path, sep="\t", comment="#", dtype=str)
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "ghid": col_map[c] = "ghid"
        elif cl == "tf": col_map[c] = "tf"
        elif "tissue" in cl: col_map[c] = "tissues"
    df = df.rename(columns=col_map)[["ghid", "tf", "tissues"]]
    df["ghid"] = df["ghid"].str.strip(); df["tf"] = df["tf"].str.strip().str.upper()
    df["tissues"] = df["tissues"].fillna("")
    if tissue_filter:
        df = df[df["tissues"].str.contains(tissue_filter, case=False, na=False)]
    return df[["ghid", "tf"]].drop_duplicates()

def parse_tissues(tissue_path, filter_str="") -> set:
    active = set()
    with open(tissue_path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            gh_id, tissue = parts[0].strip(), parts[2].strip()
            if not filter_str or filter_str.lower() in tissue.lower(): active.add(gh_id)
    return active

def build_l2_edges(gene_set, gff_df, tfbs_df, active_enhancers) -> pd.DataFrame:
    if tfbs_df.empty: return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])
    if active_enhancers: gff_df = gff_df[gff_df["enhancer_id"].isin(active_enhancers)]
    tfbs_df = tfbs_df.rename(columns={"ghid": "enhancer_id"})
    merged  = gff_df.merge(tfbs_df, on="enhancer_id", how="inner")
    rows = []
    for _, row in merged.iterrows():
        tf, tgt = row["tf"], row["gene"]
        if tf in gene_set and tgt in gene_set and tf != tgt:
            rows.append({"source": tf, "target": tgt, "sign": 0, "level": 2, "db": "GeneHancer_TFBS"})
    return pd.DataFrame(rows).drop_duplicates(["source", "target"]) if rows else pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

gff_df = parse_gff_gene_associations(GH_GFF_PATH)
tfbs_df = parse_tfbs(GH_TFBS_PATH, tissue_filter=TISSUE_FILTER)
active_enh = parse_tissues(GH_TISSUE_PATH, TISSUE_FILTER)
L2_GH = build_l2_edges(GENE_SET, gff_df, tfbs_df, active_enh)
L2_GH.to_csv(_l2_cache, sep="\t", index=False)
print(f"GeneHancer L2: {len(L2_GH)} edges")

# ────────────────────────────────────────────────────────────
# CELL 7 — Level 3: STRING PPI + transitive closure
# ────────────────────────────────────────────────────────────
def crawl_string_ppi(gene_set, min_score=700):
    url_map = "https://string-db.org/api/json/get_string_ids"
    url_net = "https://string-db.org/api/json/network"
    genes, id_map = sorted(gene_set), {}
    for i in tqdm(range(0, len(genes), 500), desc="STRING id-map"):
        batch = genes[i:i+500]
        try:
            r = SESSION.post(url_map, data={"identifiers": "\r".join(batch), "species": 9606, "limit": 1, "echo_query": 1, "caller_identity": "grn_crawler"}, timeout=60)
            r.raise_for_status()
            for rec in r.json():
                sym, sid = rec.get("queryItem","").upper(), rec.get("stringId","")
                if sym and sid: id_map[sym] = sid
        except Exception as e: log.debug("STRING id_map batch %d: %s", i, e)
        time.sleep(0.5)
    if not id_map: return pd.DataFrame(columns=["source","target","sign","level","db"])
    rev, rows, ids = {v: k for k, v in id_map.items()}, [], list(id_map.values())
    for i in tqdm(range(0, len(ids), 500), desc="STRING network"):
        try:
            r = SESSION.post(url_net, data={"identifiers": "\r".join(ids[i:i+500]), "species": 9606, "required_score": min_score, "caller_identity": "grn_crawler"}, timeout=120)
            r.raise_for_status()
            for itx in r.json():
                a, b = rev.get(itx.get("stringId_A",""),"").upper(), rev.get(itx.get("stringId_B",""),"").upper()
                if a in gene_set and b in gene_set and a != b:
                    rows += [{"source": a, "target": b, "sign": 0, "level": 3, "db": "STRING"},
                             {"source": b, "target": a, "sign": 0, "level": 3, "db": "STRING"}]
        except Exception as e: log.debug("STRING net batch %d: %s", i, e)
        time.sleep(0.5)
    return pd.DataFrame(rows).drop_duplicates(["source","target"]) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

def build_l3_transitive(l2, ppi):
    if l2.empty or ppi.empty: return pd.DataFrame(columns=["source","target","sign","level","db"])
    partners = {}
    for _, row in ppi.iterrows(): partners.setdefault(row["source"], set()).add(row["target"])
    rows = []
    for _, row in l2.iterrows():
        A, B = row["source"], row["target"]
        for C in partners.get(A, set()):
            if C != B: rows.append({"source": C, "target": B, "sign": 0, "level": 3, "db": "L2+PPI"})
    return pd.DataFrame(rows).drop_duplicates(["source","target"]) if rows else pd.DataFrame(columns=["source","target","sign","level","db"])

_pp, _pt = Path(CACHE_DIR)/"l3_ppi_raw.tsv", Path(CACHE_DIR)/"l3_transitive.tsv"
PPI_RAW = pd.read_csv(_pp, sep="\t") if _pp.exists() else crawl_string_ppi(GENE_SET, STRING_MIN_SCORE)
if not _pp.exists(): PPI_RAW.to_csv(_pp, sep="\t", index=False)
L3 = pd.read_csv(_pt, sep="\t") if _pt.exists() else build_l3_transitive(L2_GH, PPI_RAW)
if not _pt.exists(): L3.to_csv(_pt, sep="\t", index=False)
print(f"STRING PPI: {len(PPI_RAW)//2} undirected pairs  |  L3 transitive: {len(L3)} edges")
# ────────────────────────────────────────────────────────────
# CELL 8 — COXPRESdb v8.1: Robust Download + Parse (Colab)
# ✅ Handles: download failures, manual upload fallback,
#    ID format detection, chunked parsing with MR ranking
# ────────────────────────────────────────────────────────────

# ── PREREQUISITES (define in prior cell) ─────────────────────
# GENE_SET = {"TP53", "BRCA1", "EGFR"}  # UPPERCASE symbols
# COEX_TOPN = 5

# ── Imports ─────────────────────────────────────────────────
import os, sys, zipfile, logging, warnings, time, shutil
from pathlib import Path
from typing import Set, Optional, List
import pandas as pd
import requests  # Colab has this pre-installed

# ── Logging ─────────────────────────────────────────────────
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────
CACHE_DIR             = "/content/cache"
COXPRESDB_EXTRACT_DIR = "/content/coxpresdb_extract"
COXPRESDB_CACHE       = Path(CACHE_DIR) / "l4_coex.tsv"

# Zenodo download URLs (v8.1 human union dataset)
# Try these in order if one fails:
COXPRESDB_URLS = [
    # Primary: union (microarray + RNA-seq)
    "https://zenodo.org/records/6861444/files/Hsa-u.v22-05.G16651-S245698.combat_pca.subagging.z.d.zip",
    # Fallback 1: microarray-only (smaller, ~300 MB)
    "https://zenodo.org/records/6861444/files/Hsa-m.v21-06.G20283-S25362.combat_pca.subagging.ls.d.zip",
    # Fallback 2: RNA-seq-only
    "https://zenodo.org/records/6861444/files/Hsa-r.v21-06.G16651-S245698.combat_pca.subagging.ls.d.zip",
]

# Local paths (will be set after successful download)
COXPRESDB_LOCAL_ZIP = None


# ── Helper: Robust Download with Retry/Fallback ─────────────
def download_coxpresdb_with_fallback(output_path: str) -> bool:
    """
    Attempt to download COXPRESdb data with multiple URLs and retry logic.
    Returns True if successful, False otherwise.
    """
    global COXPRESDB_LOCAL_ZIP

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1e6:
        size_mb = os.path.getsize(output_path) / 1e6
        log.info("[cache] Found existing zip: %.1f MB", size_mb)
        COXPRESDB_LOCAL_ZIP = output_path
        return True

    log.info("Attempting to download COXPRESdb human coexpression data...")

    for i, url in enumerate(COXPRESDB_URLS, 1):
        log.info("Try %d/%d: %s", i, len(COXPRESDB_URLS), url[:80] + "...")

        try:
            # Method 1: HTTP stream download (pure Python, no shell magics)
            import urllib.request, shutil as _sh
            _req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GRN-research)"})
            with urllib.request.urlopen(_req, timeout=120) as _resp, open(output_path, "wb") as _out:
                _sh.copyfileobj(_resp, _out)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 1e6:
                size_mb = os.path.getsize(output_path) / 1e6
                log.info("✓ Download successful: %.1f MB", size_mb)
                COXPRESDB_LOCAL_ZIP = output_path
                return True
            else:
                log.warning("Downloaded file too small or missing; cleaning up")
                if os.path.exists(output_path):
                    os.remove(output_path)

        except Exception as e:
            log.warning("Download attempt %d failed: %s", i, e)
            if os.path.exists(output_path):
                os.remove(output_path)
            continue

    # ── Fallback: Manual Upload Instructions ─────────────────
    log.error("❌ All download attempts failed.")
    print("\n" + "⚠️ MANUAL UPLOAD FALLBACK".center(60, "─"))
    print("1. Visit: https://zenodo.org/records/6861444")
    print("2. Download ONE of these files:")
    print("   • Hsa-u.v22-05...zip  (union, ~1.1 GB) ← RECOMMENDED")
    print("   • Hsa-m.v21-06...zip  (microarray, ~300 MB)")
    print("   • Hsa-r.v21-06...zip  (RNA-seq, ~800 MB)")
    print("3. Upload to Colab via folder icon 📁 in left sidebar")
    print("4. Set the path below and re-run:")
    print(f'   COXPRESDB_LOCAL_ZIP = "/content/your-downloaded-file.zip"')
    print("─" * 60 + "\n")
    return False


# ── Helper: Column Detection ─────────────────────────────────
def _detect_col(header: list, patterns: list, fallback: int) -> int:
    for i, col in enumerate(header):
        col_lower = col.lower().strip()
        if any(p.lower() in col_lower for p in patterns):
            return i
    return fallback


# ── Helper: Gene ID Normalization ────────────────────────────
def _normalise_id(raw: str, gene_id_type: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    if gene_id_type == "entrez":
        return raw if raw.isdigit() else None
    return raw.upper()  # symbol / ensembl


# ── Main Parser ──────────────────────────────────────────────
def parse_coxpresdb_bulk(
    gene_set: Set[str],
    top_n: int = 5,
    chunk_limit: Optional[int] = None,
    gene_id_type: str = "symbol",
    zip_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse COXPRESdb v8.1 bulk data with robust error handling.

    Parameters
    ----------
    gene_set : Set[str] - genes to filter (format depends on gene_id_type)
    top_n : int - partners per gene to retain (ranked by MR)
    chunk_limit : Optional[int] - limit chunks for testing
    gene_id_type : str - "symbol" | "entrez" | "ensembl"
    zip_path : Optional[str] - override auto-detected zip path

    Returns
    -------
    pd.DataFrame with columns: source, target, sign, level, db
    """
    global COXPRESDB_LOCAL_ZIP

    if not gene_set:
        log.error("gene_set is empty")
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Set zip path
    if zip_path:
        COXPRESDB_LOCAL_ZIP = zip_path
    elif not COXPRESDB_LOCAL_ZIP:
        COXPRESDB_LOCAL_ZIP = "/content/Hsa_union_coex.zip"

    # Download if needed
    if not os.path.exists(COXPRESDB_LOCAL_ZIP):
        if not download_coxpresdb_with_fallback(COXPRESDB_LOCAL_ZIP):
            return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Extract if needed
    extract_dir = Path(COXPRESDB_EXTRACT_DIR)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting zip (this may take 1-3 min)...")
        try:
            with zipfile.ZipFile(COXPRESDB_LOCAL_ZIP, "r") as zf:
                zf.extractall(extract_dir)
            log.info("✓ Extraction complete")
        except zipfile.BadZipFile:
            log.error("❌ Invalid zip file — re-download or upload manually")
            return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Find chunk files (.d extension or numeric names)
    chunk_files = sorted(
        p for p in extract_dir.iterdir()
        if p.suffix == ".d" and p.stem.isdigit()
    )
    if not chunk_files:
        chunk_files = sorted(
            p for p in extract_dir.iterdir()
            if p.is_file() and p.name.isdigit()
        )
    if not chunk_files:
        log.error("No chunk files found in %s", extract_dir)
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    log.info("Found %d chunk files | gene_id_type=%s", len(chunk_files), gene_id_type)

    # Collect all valid pairs with MR scores
    collected: List[dict] = []
    files_processed = 0

    for path in chunk_files:
        if chunk_limit and files_processed >= chunk_limit:
            break
        files_processed += 1

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header_line = f.readline().strip()
                if not header_line:
                    continue
                header = header_line.split("\t")

                q_idx = _detect_col(header, ["gene1","query","source","gene_a","symbol1","id1"], 0)
                t_idx = _detect_col(header, ["gene2","target","partner","gene_b","symbol2","id2"], 1)
                mr_idx = _detect_col(header, ["mr","mutual_rank","mutualrank","rank","score"], -1)

                if q_idx == t_idx:
                    t_idx = q_idx + 1

                for row_num, line in enumerate(f):
                    parts = line.strip().split("\t")
                    if len(parts) <= max(q_idx, t_idx):
                        continue

                    src = _normalise_id(parts[q_idx], gene_id_type)
                    tgt = _normalise_id(parts[t_idx], gene_id_type)

                    if not src or not tgt or src == tgt:
                        continue
                    if src not in gene_set or tgt not in gene_set:
                        continue

                    # Parse MR (lower = stronger); fallback to row order
                    if mr_idx != -1 and mr_idx < len(parts):
                        try:
                            mr = float(parts[mr_idx])
                        except ValueError:
                            mr = float(row_num)
                    else:
                        mr = float(row_num)

                    collected.append({"source": src, "target": tgt, "mr": mr})

        except Exception as e:
            log.warning("Error reading %s: %s", path.name, e)
            continue

    if not collected:
        log.warning("No valid gene pairs found — check gene_id_type and GENE_SET")
        return pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

    # Rank by MR (ascending) and keep top_n per source
    raw_df = pd.DataFrame(collected).sort_values("mr")
    top_df = raw_df.groupby("source", sort=False).head(top_n).reset_index(drop=True)

    # Build bidirectional edges, deduplicate
    fwd = top_df[["source", "target"]].copy()
    rev = top_df.rename(columns={"source": "target", "target": "source"})[["source", "target"]].copy()

    edges = (pd.concat([fwd, rev], ignore_index=True)
               .drop_duplicates(subset=["source", "target"])
               .reset_index(drop=True))

    edges["sign"] = 0
    edges["level"] = 4
    edges["db"] = "COXPRESdb"

    log.info("COXPRESdb: %d edges | %d unique genes | top-%d by MR",
             len(edges), edges[["source","target"]].stack().nunique(), top_n)
    return edges


# ── Main Execution ───────────────────────────────────────────
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Check cache first
if COXPRESDB_CACHE.exists() and COXPRESDB_CACHE.stat().st_size > 100:
    try:
        L4 = pd.read_csv(COXPRESDB_CACHE, sep="\t")
        if not L4.empty:
            log.info("[cache] Loaded %d edges from %s", len(L4), COXPRESDB_CACHE)
        else:
            log.warning("Cached file is empty — will re-parse")
            L4 = None
    except Exception as e:
        log.warning("Cache read error: %s — will re-parse", e)
        L4 = None
else:
    L4 = None

# Parse if not cached
if L4 is None:
    if "GENE_SET" not in globals() or "COEX_TOPN" not in globals():
        raise NameError("Define GENE_SET (set) and COEX_TOPN (int) before running")

    # Normalize gene set for symbol matching
    gene_set_norm = {g.strip().upper() for g in GENE_SET if g}

    L4 = parse_coxpresdb_bulk(
        gene_set_norm,
        top_n=COEX_TOPN,
        chunk_limit=None,          # Set to 50 for quick testing
        gene_id_type="symbol",     # Try "entrez" if no matches
        zip_path=None,             # Or set manually: "/content/my-file.zip"
    )

    if not L4.empty:
        L4.to_csv(COXPRESDB_CACHE, sep="\t", index=False)
        log.info("✓ Saved to %s", COXPRESDB_CACHE)

# ── Output Summary ───────────────────────────────────────────
print(f"\n📊 COXPRESdb L4: {len(L4):,} edges" if L4 is not None else "\n📊 COXPRESdb L4: (parse failed)")

if L4 is not None and not L4.empty:
    print(f"   • Unique genes: {L4[['source','target']].stack().nunique():,}")
    print(f"   • Columns: {list(L4.columns)}")
    print(f"   • Sample:\n{L4.head(3).to_string(index=False)}")
elif L4 is not None:
    print("  ⚠️  0 edges — troubleshooting:")
    print("     1. Try gene_id_type='entrez' (COXPRESdb often uses numeric IDs)")
    print("     2. Verify GENE_SET matches file's ID format")
    print("     3. Test with chunk_limit=10 to inspect first chunk")
    print("     4. Run diagnostic: check header/column names in .d files")
else:
    print("  ❌ Parsing failed — check logs above for download/extract errors")

# ────────────────────────────────────────────────────────────
# 💡 Quick Test Mode (uncomment to debug):
# ────────────────────────────────────────────────────────────
# L4_test = parse_coxpresdb_bulk(
#     {"TP53", "BRCA1", "EGFR"},
#     top_n=3,
#     chunk_limit=20,      # Only scan first 20 chunks
#     gene_id_type="symbol"  # Try "entrez" if this returns 0 edges
# )
# print(f"Test run: {len(L4_test)} edges")
# if not L4_test.empty:
#     print(L4_test.head())
# ────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────
# CELL 9 — Merge all levels
# ────────────────────────────────────────────────────────────
def merge_edges(dfs):
    valid_dfs = [d for d in dfs if not d.empty]
    if not valid_dfs: return pd.DataFrame(columns=["source","target","sign","level","db"])
    all_e = pd.concat(valid_dfs, ignore_index=True)

    # Pandas 2.2+ compatible aggregation (replaces deprecated .apply())
    agg_dict = {
        "level": "min",
        "sign": lambda x: int(x.dropna().iloc[0]) if x.dropna().nunique() == 1 else 0,
        "db": lambda x: ",".join(sorted(x.unique()))
    }
    merged = all_e.groupby(["source", "target"], as_index=False).agg(agg_dict)
    log.info("Merged pool: %d unique (source,target) pairs", len(merged))
    return merged

POOL = merge_edges([L1_TRRUST, L1_OMNIPATH, L1_COLLECTRI, L2_GH, L3, L4])

print("\n=== Edge pool ===")
print(POOL.groupby("level")["source"].count().rename("edges").to_string())
print(f"\nPool total  : {len(POOL):,}")
print(f"Unique src  : {POOL['source'].nunique():,}")
print(f"Unique tgt  : {POOL['target'].nunique():,}")
print(f"\nSigned (L1) : {(POOL['sign'] != 0).sum():,}")
print(f"Unsigned    : {(POOL['sign'] == 0).sum():,}")

# Save final pool
POOL.to_csv(OUT_FILE, sep="\t", index=False)
print(f"\n✓ Saved to {OUT_FILE}")
# ── GATA1 diagnostic: POOL ───────────────────────────────────
_g1_pool_out = POOL[POOL["source"] == "GATA1"]
_g1_pool_in  = POOL[POOL["target"] == "GATA1"]
print(f"\n── GATA1 in POOL ──")
print(f"  Outgoing edges : {len(_g1_pool_out):,}")
print(f"  Incoming edges : {len(_g1_pool_in):,}")
if not _g1_pool_out.empty:
    print(f"  Out by level   :\n{_g1_pool_out.groupby('level')['target'].count().rename('edges').to_string()}")
    print(f"  Out by db      :\n{_g1_pool_out['db'].value_counts().to_string()}")
else:
    print("  ⚠️  GATA1 has NO outgoing edges in POOL — not present as source")

# ────────────────────────────────────────────────────────────
# CELL 10 — Rule-of-10 budget + greedy (param-count stopping)
# ────────────────────────────────────────────────────────────
import networkx as nx
from tqdm import tqdm

# ── Parameter budget ─────────────────────────────────────────
#
#  Sign treatment:
#    sign ∈ {+1, −1}  (L1 known)  →  fixed constant, NOT learned
#                                     edge costs 2 params (K_d, n)
#    sign = 0         (L2–L4)     →  learnable s_ij ∈ [−1, +1]
#                                     edge costs 3 params (K_d, n, s)
#
#  Total params:
#    gene params  :  3 × 2000                      =  6 000
#    edge params  :  2 × E_signed + 3 × E_unsigned  ≤  22 000
#
#  Worst case (all unsigned):  E ≤ 22 000 / 3 ≈  7 333  →  z ≈ 3.7
#  Best case  (all signed  ):  E ≤ 22 000 / 2 = 11 000  →  z = 5.5
#  Typical    (f_signed ≈ 0.3):
#    E ≤ 22 000 / (3 − f) = 22 000 / 2.7 ≈  8 148
#
#  → Stop greedily when cumulative param cost hits 22 000.
#    This is tighter and more honest than a fixed edge count.

N_GENES             = 2_000
PARAMS_GENE         = 3          # V_j, α_j, b_j  (per gene, always learned)
PARAMS_EDGE_SIGNED  = 2          # K_d, n          (sign is a known constant)
PARAMS_EDGE_UNSIGNED= 3          # K_d, n, s_ij    (sign is learned ∈ [−1,+1])
N_CONSTRAINTS       = 280_000

PARAM_BUDGET        = N_CONSTRAINTS // 10                        # 28 000
GENE_PARAM_TOTAL    = PARAMS_GENE * N_GENES                      #  6 000
EDGE_PARAM_BUDGET   = PARAM_BUDGET - GENE_PARAM_TOTAL            # 22 000

# For reporting: estimate z under typical pool composition
# (computed precisely inside the algorithm when pool sign-mix is known)
_z_worst = EDGE_PARAM_BUDGET / PARAMS_EDGE_UNSIGNED / N_GENES    # ≈ 3.67
_z_best  = EDGE_PARAM_BUDGET / PARAMS_EDGE_SIGNED   / N_GENES    # = 5.5

print(f"Parameter budget    : {PARAM_BUDGET:,}")
print(f"Gene params         : {GENE_PARAM_TOTAL:,}")
print(f"Edge param budget   : {EDGE_PARAM_BUDGET:,}")
print(f"z  (all unsigned)   : {_z_worst:.2f}  →  E_max ≈ {int(_z_worst*N_GENES):,}")
print(f"z  (all signed)     : {_z_best:.2f}   →  E_max ≈ {int(_z_best *N_GENES):,}")
print("Stopping criterion  : cumulative edge-param cost ≤ EDGE_PARAM_BUDGET")


# ── Helper: param cost per edge ───────────────────────────────
def _edge_cost(sign: int) -> int:
    """2 if sign is known (±1), 3 if unknown (0)."""
    return PARAMS_EDGE_SIGNED if sign != 0 else PARAMS_EDGE_UNSIGNED


# ── Main algorithm ────────────────────────────────────────────
def greedy_select_connected(edges,
                             edge_param_budget = EDGE_PARAM_BUDGET,
                             reward            = GREEDY_REWARD,
                             level_conf        = LEVEL_CONF):
    """
    Select edges until the GRNN parameter budget is exhausted.
    Stopping criterion: sum(param_cost per selected edge) ≤ edge_param_budget.

    Each edge costs:
      2 params  if sign ∈ {+1, −1}  (K_d, n;  sign fixed from database)
      3 params  if sign = 0          (K_d, n, s_ij learnable ∈ [−1, +1])

    Phase 0: Pool → LCC (drop unreachable nodes)
    Phase 1: Kruskal spanning tree (connectivity guarantee)
    Phase 2: Reward-penalty greedy fill up to param budget
    """
    edges  = edges.copy().reset_index(drop=True)
    signs  = edges["sign"].to_numpy(dtype=int)
    costs  = np.array([_edge_cost(s) for s in signs], dtype=int)

    # ── Phase 0: LCC restriction ─────────────────────────────
    G_pool = nx.from_pandas_edgelist(
                 edges, "source", "target", create_using=nx.Graph())
    comps  = sorted(nx.connected_components(G_pool), key=len, reverse=True)

    if len(comps) > 1:
        dropped = set().union(*comps[1:])
        log.warning(
            "Pool: %d components → restricting to LCC (%d nodes). "
            "Dropping %d unreachable nodes: %s …",
            len(comps), len(comps[0]), len(dropped), sorted(dropped)[:8],
        )
        lcc   = comps[0]
        edges = edges[
            edges["source"].isin(lcc) & edges["target"].isin(lcc)
        ].reset_index(drop=True)
        signs = edges["sign"].to_numpy(dtype=int)
        costs = np.array([_edge_cost(s) for s in signs], dtype=int)
        log.warning("Pool after LCC: %d edges, %d nodes.", len(edges), len(lcc))
    else:
        log.info("Pool connected (%d nodes). No repair needed.", len(comps[0]))

    # ── Arrays ───────────────────────────────────────────────
    conf_arr = edges["level"].map(level_conf).fillna(0.1).to_numpy(dtype=float)
    src_arr  = edges["source"].to_numpy()
    tgt_arr  = edges["target"].to_numpy()

    all_nodes = sorted(set(src_arr) | set(tgt_arr))
    N         = len(all_nodes)
    node2id   = {n: i for i, n in enumerate(all_nodes)}
    src_ids   = np.array([node2id[s] for s in src_arr], dtype=int)
    tgt_ids   = np.array([node2id[t] for t in tgt_arr], dtype=int)

    # Check spanning-tree cost fits budget
    span_cost = sum(
        sorted([_edge_cost(s) for s in signs])[: N - 1]
    )
    if span_cost > edge_param_budget:
        log.warning(
            "Spanning tree alone costs %d params > budget %d. "
            "Output may not be fully connected.", span_cost, edge_param_budget)

    # ── Union-Find ───────────────────────────────────────────
    parent = np.arange(N, dtype=int)
    rnk    = np.zeros(N, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return False
        if rnk[rx] < rnk[ry]: rx, ry = ry, rx
        parent[ry] = rx
        if rnk[rx] == rnk[ry]: rnk[rx] += 1
        return True

    remaining      = np.ones(len(edges), dtype=bool)
    selected       = []
    params_used    = 0

    # ── Phase 1: Kruskal spanning tree ───────────────────────
    n_comp   = N
    bridged  = 0
    p1_order = np.argsort(-conf_arr)   # best-confidence first

    for idx in p1_order:
        if n_comp <= 1: break
        cost = costs[idx]
        if params_used + cost > edge_param_budget: continue  # too expensive
        if union(src_ids[idx], tgt_ids[idx]):
            selected.append(int(idx))
            remaining[int(idx)] = False
            params_used += cost
            n_comp      -= 1
            bridged     += 1

    if n_comp > 1:
        log.error("Phase 1: %d components remain after spanning tree.", n_comp)
    else:
        log.info("Phase 1: %d spanning-tree edges (%d params used) → connected ✓",
                 bridged, params_used)

    # ── Phase 2: reward-penalty greedy ───────────────────────
    accum = np.zeros(len(edges), dtype=float)

    pbar = tqdm(desc="Greedy phase 2", total=edge_param_budget - params_used)
    while params_used < edge_param_budget and remaining.any():

        # (1) reward all remaining
        accum[remaining] += reward

        # (2) pick argmax
        best = int(np.argmax(np.where(remaining, conf_arr + accum, -np.inf)))
        cost = costs[best]

        if params_used + cost > edge_param_budget:
            # Try to fill remaining budget with cheaper (signed) edges only
            signed_remaining = remaining & (costs == PARAMS_EDGE_SIGNED)
            if not signed_remaining.any():
                break
            best = int(np.argmax(
                np.where(signed_remaining, conf_arr + accum, -np.inf)))
            cost = costs[best]
            if params_used + cost > edge_param_budget:
                break

        selected.append(best)
        remaining[best] = False
        params_used     += cost
        pbar.update(cost)

        # (3) penalise edges sharing source u or target v
        accum[remaining & (src_arr == src_arr[best])] -= reward  # (u, v')
        accum[remaining & (tgt_arr == tgt_arr[best])] -= reward  # (u', v)

    pbar.close()

    # ── Build result ─────────────────────────────────────────
    result = edges.iloc[selected].drop(columns=["conf"], errors="ignore").copy()

    # ── Hard connectivity assert ─────────────────────────────
    U         = nx.from_pandas_edgelist(result, "source", "target",
                                        create_using=nx.Graph())
    out_comps = sorted(nx.connected_components(U), key=len, reverse=True)
    if len(out_comps) > 1:
        raise RuntimeError(
            f"Output NOT connected ({len(out_comps)} components). "
            f"Sizes: {[len(c) for c in out_comps[:10]]}"
        )

    n_signed   = (result["sign"] != 0).sum()
    n_unsigned = (result["sign"] == 0).sum()
    z_actual   = len(result) / N_GENES

    log.info(
        "✓  %d edges | %d nodes | 1 component | %d params used / %d budget\n"
        "   signed (%d×2=%d params)  unsigned (%d×3=%d params)  z=%.2f",
        len(result), U.number_of_nodes(), params_used, edge_param_budget,
        n_signed,   n_signed   * PARAMS_EDGE_SIGNED,
        n_unsigned, n_unsigned * PARAMS_EDGE_UNSIGNED,
        z_actual,
    )
    return result, params_used


# ── Run ───────────────────────────────────────────────────────
SELECTED, PARAMS_USED = greedy_select_connected(POOL)
SELECTED.to_csv(OUT_FILE, sep="\t", index=False)

n_signed   = (SELECTED["sign"] != 0).sum()
n_unsigned = (SELECTED["sign"] == 0).sum()
# ── GATA1 diagnostic: SELECTED ───────────────────────────────
_g1_sel_out = SELECTED[SELECTED["source"] == "GATA1"]
_g1_sel_in  = SELECTED[SELECTED["target"] == "GATA1"]
print(f"\n── GATA1 in SELECTED ──")
print(f"  Outgoing edges : {len(_g1_sel_out):,}  (was {len(_g1_pool_out):,} in POOL)")
print(f"  Incoming edges : {len(_g1_sel_in):,}  (was {len(_g1_pool_in):,} in POOL)")
print(f"  Δ outgoing     : {len(_g1_sel_out) - len(_g1_pool_out):+,}")
if not _g1_sel_out.empty:
    print(f"  Out by level   :\n{_g1_sel_out.groupby('level')['target'].count().rename('edges').to_string()}")
    print(f"  Retained targets: {sorted(_g1_sel_out['target'].tolist())}")
else:
    print("  ⚠️  GATA1 pruned entirely from outgoing edges in SELECTED")

# Dropped edges (in POOL but not SELECTED)
_g1_dropped = _g1_pool_out[
    ~_g1_pool_out["target"].isin(_g1_sel_out["target"])
]
print(f"  Dropped targets ({len(_g1_dropped)}): {sorted(_g1_dropped['target'].tolist())}")

print(f"\n=== Final GRN ===")
print(f"Total edges         : {len(SELECTED):,}")
print(f"  signed  (±1, 2p)  : {n_signed:,}   → {n_signed*2:,} params")
print(f"  unsigned (0, 3p)  : {n_unsigned:,}  → {n_unsigned*3:,} params")
print(f"Edge params used    : {PARAMS_USED:,}  /  {EDGE_PARAM_BUDGET:,}")
print(f"Gene params         : {GENE_PARAM_TOTAL:,}")
print(f"Total params        : {PARAMS_USED + GENE_PARAM_TOTAL:,}  /  {PARAM_BUDGET:,}")
print(f"z edges/gene        : {len(SELECTED)/N_GENES:.2f}")
print(f"\n✓ Saved → {OUT_FILE}")

gata1_pool = POOL[POOL['source'] == 'GATA1']
gata1_selected = SELECTED[SELECTED['source'] == 'GATA1']

print(f"GATA1 outgoing edges in RPE1 POOL: {len(gata1_pool)}")
if not gata1_pool.empty:
    display(gata1_pool.head())

print(f"\nGATA1 outgoing edges in RPE1 SELECTED: {len(gata1_selected)}")
if not gata1_selected.empty:
    display(gata1_selected.head())

genes_to_check = ['OTX2', 'PAX6', 'MITF']

present_genes = set()
for gene in genes_to_check:
    if (SELECTED['source'] == gene).any() or (SELECTED['target'] == gene).any():
        present_genes.add(gene)

print(f"Genes present in RPE1 GRN: {list(present_genes)}")
missing_genes = set(genes_to_check) - present_genes
print(f"Genes not present in RPE1 GRN: {list(missing_genes)}")

# Extract full gene list from the GRN graph
grn_genes = set(SELECTED['source']).union(set(SELECTED['target']))
grn_gene_list = sorted(list(grn_genes))

print(f"Total unique genes in the GRN graph: {len(grn_gene_list)}")
print(grn_gene_list)

# Log/save the GRN
log_path = 'RPE1_grn_logged.tsv'
SELECTED.to_csv(log_path, sep='\t', index=False)
print(f"\nLogged RPE1 GRN to '{log_path}'")

import pandas as pd

# Read the saved TSV files
k562_grn = pd.read_csv('K562_grn_logged.tsv', sep='\t')
rpe1_grn = pd.read_csv('RPE1_grn_logged.tsv', sep='\t')

# Extract unique genes
k562_genes = set(k562_grn['source']).union(set(k562_grn['target']))
rpe1_genes = set(rpe1_grn['source']).union(set(rpe1_grn['target']))

# Compute intersection
intersection_genes = k562_genes.intersection(rpe1_genes)

# Sort lists for consistent output
k562_list = sorted(list(k562_genes))
rpe1_list = sorted(list(rpe1_genes))
intersection_list = sorted(list(intersection_genes))

# Write to a text file
output_path = 'grn_gene_lists.txt'
with open(output_path, 'w') as f:
    f.write(f"=== (i) K562 GRN Genes ({len(k562_list)}) ===\n")
    f.write(", ".join(k562_list) + "\n\n")

    f.write(f"=== (ii) RPE1 GRN Genes ({len(rpe1_list)}) ===\n")
    f.write(", ".join(rpe1_list) + "\n\n")

    f.write(f"=== (iii) Intersection of K562 and RPE1 GRN Genes ({len(intersection_list)}) ===\n")
    f.write(", ".join(intersection_list) + "\n")

print(f"Lists successfully saved to '{output_path}'")
print(f"K562 genes: {len(k562_list)}")
print(f"RPE1 genes: {len(rpe1_list)}")
print(f"Intersection: {len(intersection_list)}")

################################################################################
# CELL 12 — GRNN: build → plots before → train → plots after → target GRN
#
# Biology fixes vs. naive GRNN
# ─────────────────────────────
#   [1] x0: log1p(mean(expm1(X))) — bias-free baseline (Jensen's gap fixed)
#   [2] perturbed_value: observed expression of the KO'd gene per experiment
#   [3] Kd_raw: inv_softplus(x0_src) → Hill occupancy φ(x0) ≈ 0.5 at init
#   [4] Dual-channel _step: h_act in numerator / h_rep in denominator
#       (non-zero gradients for both channels regardless of act/rep ratio)
#
# Weight parameterisation
# ───────────────────────
#   sign ∈ {+1,−1} → w_hat = sign · sigmoid(w_raw)   direction fixed by DB
#   sign = 0       → w_hat = tanh(w_raw)              direction learned
#
# Visual conventions (all panels)
# ────────────────────────────────
#   Red    = activation  (eff > 0)     Blue  = repression (eff < 0)
#   Grey   = unknown / near-zero       RdBu_r palette (CVD-safe)
#   Solid  = DB-constrained direction  Dashed = sign=0 (direction learned)
#
# Outputs
# ───────
#   /content/grn_plots/{CELL_LINE}_heatmap_{before,after}.png/.pdf
#   /content/grn_plots/{CELL_LINE}_{GENE}_{before,after,changed}.png/.pdf
#   /content/grn_plots/{CELL_LINE}_{GENE}_{before,after}.png/.pdf  ← targets
#   (GENE = VIZ_GENES + GATA1 + MITF)
################################################################################

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import pandas as pd, numpy as np, json, networkx as nx, logging
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from IPython.display import display, HTML
from tqdm import tqdm

log = logging.getLogger("grn")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ════════════════════════════════════════════════════════════
#  PUBLICATION STYLE
#  Q1 journal column widths (Nature Methods / Cell / Cell Systems)
#  NeurIPS: COL_1=3.46  Cell Press: CP_COL1=3.35, CP_COL15=4.49, CP_COL2=6.85
# ════════════════════════════════════════════════════════════
COL_1,  COL_15,  COL_2  = 3.46, 5.00, 7.20    # NeurIPS / Nature Methods
CP_COL1, CP_COL15, CP_COL2 = 3.35, 4.49, 6.85 # Cell Press

mpl.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":    "dejavusans",
    "font.size":           7,   "axes.titlesize":     8,
    "axes.labelsize":      7,   "xtick.labelsize":    6.5,
    "ytick.labelsize":     6.5, "legend.fontsize":    6.5,
    "legend.frameon":      False,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.5, "ytick.major.width":  0.5,
    "xtick.major.size":    2.5, "ytick.major.size":   2.5,
    "lines.linewidth":     1.0, "patch.linewidth":    0.5,
    "axes.spines.top":     False, "axes.spines.right": False,
    "axes.grid":           False,
    "savefig.dpi":         600,   # Cell Press line-art standard
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "figure.dpi":          150,
    "pdf.fonttype":        42,  # TrueType embedded — editable in Illustrator
    "ps.fonttype":         42,
    "svg.fonttype":        "none",
})

# CVD-safe RdBu palette (Nature/Cell figure convention)
ACT_C    = "#B2182B"   # deep red    — activation
REP_C    = "#2166AC"   # deep blue   — repression
UNK_C    = "#BDBDBD"   # neutral grey
NODE_CTR = "#F4A261"   # warm amber  — centre / perturbed gene
NODE_H1  = "#457B9D"   # slate blue  — direct neighbours
NODE_H2  = "#D9D9D9"   # cool grey   — 2nd hop
EDGE_LBL = "#3A3A3A"
CMAP_DIV = plt.get_cmap("RdBu_r")
EDGE_C_SIGN = {1: ACT_C, -1: REP_C, 0: UNK_C}
NODE_C   = {"center": NODE_CTR, "hop1": NODE_H1, "hop2": NODE_H2}
NODE_S   = {"center": 360,      "hop1": 220,     "hop2": 110}

PLOT_DIR = Path("/content/grn_plots")
PLOT_DIR.mkdir(exist_ok=True)

ensg2sym         = adata.var["gene_name"].to_dict()
GENE_NAMES       = [ensg2sym.get(e, e) for e in hvg_list]
PERTURBATION_COL = "gene"
CONTROL_LABEL    = "non-targeting"


# ════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════
def _logmean(X):
    """log1p(mean(expm1(X))) — bias-free baseline in log1p count space."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.log1p(np.expm1(X).mean(axis=0)).ravel()


def _inv_softplus(y, eps=1e-6):
    """Inverse of F.softplus so softplus(_inv_softplus(y)) ≈ y."""
    y = torch.as_tensor(y, dtype=torch.float32).clamp(min=eps)
    return torch.log(torch.expm1(y))


def _save(fig, stem):
    """Save PNG (600 dpi) and PDF (vector TrueType) side by side."""
    for suffix in (".png", ".pdf"):
        kw = dict(facecolor="white", bbox_inches="tight")
        if suffix == ".png": kw["dpi"] = 600
        fig.savefig(PLOT_DIR / f"{stem}{suffix}", **kw)
    print(f"  ✓ {stem}.png + .pdf")


def _eff_color(w, thr=0.02):
    return ACT_C if w > thr else REP_C if w < -thr else UNK_C


def _before_edge_color(sign):
    return EDGE_C_SIGN.get(int(sign), UNK_C)


# ════════════════════════════════════════════════════════════
#  1. GRNN
# ════════════════════════════════════════════════════════════
class GRNN(nn.Module):
    """
    Gene Regulatory Neural Network — biophysical Hill-kinetics model.

    Key design decisions
    ────────────────────
    w_hat: sign ≠ 0 → sign·sigmoid(w_raw)   (DB-constrained direction)
           sign = 0 → tanh(w_raw)            (direction learned from data)

    Kd_raw = inv_softplus(x0_src) so Hill occupancy φ(x0) ≈ 0.5 at init,
    placing each TF in the responsive (unsaturated) regime.

    Dual-channel _step: separate scatter_add for activation (h_act) and
    repression (h_rep) guarantees non-zero gradients for both channels.
      x_ss = (V·(1+h_act) + b) / (1 + α + h_rep)

    w_raw init (inv_sigmoid of desired |w_hat| magnitude):
      Activation L1→0.85  L2→0.65  L3→0.45  L4→0.25
      Repression L1→0.90  L2→0.70  L3→0.50  L4→0.30
      Unknown    → N(0,0.1)
    """
    def __init__(self, gene_names, src_idx, tgt_idx, signs, levels,
                 x0_init, max_iter=100, eps=1e-5):
        super().__init__()
        self.gene_names = gene_names
        self.gene2idx   = {g: i for i, g in enumerate(gene_names)}
        self.N = len(gene_names); self.E = src_idx.numel()
        self.max_iter = max_iter; self.eps = eps

        self.register_buffer("src_idx", src_idx.long())
        self.register_buffer("tgt_idx", tgt_idx.long())
        self.register_buffer("signs",   signs.float())
        self.register_buffer("levels",  levels.float())

        self.log_V     = nn.Parameter(torch.log(x0_init.clamp(min=1e-6)))
        self.log_alpha = nn.Parameter(torch.full((self.N,), -2.3))
        self.b_raw     = nn.Parameter(torch.full((self.N,), -4.0))
        self.Kd_raw    = nn.Parameter(_inv_softplus(x0_init[src_idx].clamp(min=1e-3)))
        self.n_raw     = nn.Parameter(torch.zeros(self.E))

        def inv_sig(p): return float(np.log(p / (1 - p)))
        w = torch.zeros(self.E)
        act = signs == 1; rep = signs == -1; unk = signs == 0
        l1 = levels==1; l2 = levels==2; l3 = levels==3; l4 = levels==4
        w[act&l1]=inv_sig(0.85); w[act&l2]=inv_sig(0.65)
        w[act&l3]=inv_sig(0.45); w[act&l4]=inv_sig(0.25)
        w[rep&l1]=inv_sig(0.90); w[rep&l2]=inv_sig(0.70)
        w[rep&l3]=inv_sig(0.50); w[rep&l4]=inv_sig(0.30)
        w[unk]   =torch.randn(unk.sum()) * 0.1
        self.w_raw = nn.Parameter(w)

    @property
    def V(self):     return torch.exp(self.log_V)
    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def b(self):     return F.softplus(self.b_raw)
    @property
    def Kd(self):    return F.softplus(self.Kd_raw) + 1e-6
    @property
    def n(self):     return 1.0 + 3.0 * torch.sigmoid(self.n_raw)
    @property
    def w_hat(self):
        return torch.where(self.signs != 0,
                           self.signs * torch.sigmoid(self.w_raw),
                           torch.tanh(self.w_raw))

    @property
    def eff_weight(self):
        """eff_ij = ŵ_ij · φ_ij(x0_i) — regulatory drive at baseline."""
        with torch.no_grad():
            x0  = self.V[self.src_idx] * 0.8
            xn  = x0.pow(self.n); Kdn = self.Kd.pow(self.n)
            phi = xn / (Kdn + xn + 1e-12)
            return (self.w_hat * phi).cpu().numpy()

    def _eff_map(self):
        arr = self.eff_weight
        return {(self.gene_names[int(s)], self.gene_names[int(t)]): float(w)
                for s, t, w in zip(self.src_idx.cpu(), self.tgt_idx.cpu(), arr)}

    def _step(self, x):
        """Dual-channel thermodynamic update (non-zero gradients guaranteed)."""
        phi   = (x[self.src_idx].pow(self.n)
                 / (self.Kd.pow(self.n) + x[self.src_idx].pow(self.n) + 1e-12))
        drive = self.w_hat * phi
        h_act = torch.zeros(self.N, device=x.device, dtype=x.dtype)
        h_rep = torch.zeros(self.N, device=x.device, dtype=x.dtype)
        h_act.scatter_add_(0, self.tgt_idx, drive.clamp(min=0))
        h_rep.scatter_add_(0, self.tgt_idx, (-drive).clamp(min=0))
        return (self.V * (1.0 + h_act) + self.b) / (1.0 + self.alpha + h_rep)

    def forward(self, x0, perturbed_idx=None, perturbed_value=None):
        x = x0.clone()
        if perturbed_idx is not None and perturbed_value is not None:
            x[perturbed_idx] = perturbed_value
        for t in range(self.max_iter):
            x_new = self._step(x)
            if perturbed_idx is not None and perturbed_value is not None:
                x_new[perturbed_idx] = perturbed_value
            if (x_new - x).norm() < self.eps:
                return x_new, t + 1
            x = x_new
        return x, self.max_iter


# ════════════════════════════════════════════════════════════
#  2. TSV → model
# ════════════════════════════════════════════════════════════
def grn_tsv_to_grnn(tsv_path, gene_names, x0, max_iter=100, eps=1e-5):
    df = pd.read_csv(tsv_path, sep="\t")
    g2i = {g: i for i, g in enumerate(gene_names)}
    df  = df[df["source"].isin(g2i) & df["target"].isin(g2i)].copy()
    src = torch.tensor([g2i[g] for g in df["source"]], dtype=torch.long)
    tgt = torch.tensor([g2i[g] for g in df["target"]], dtype=torch.long)
    signs  = torch.tensor(df["sign"].fillna(0).astype(int).values,  dtype=torch.float32)
    levels = torch.tensor(df["level"].fillna(4).astype(int).values, dtype=torch.float32)
    model  = GRNN(gene_names, src, tgt, signs, levels,
                  torch.tensor(x0, dtype=torch.float32), max_iter, eps)
    log.info("GRNN: %d genes | %d edges (%d learned-sign) | %d params",
             len(gene_names), len(df), int((signs==0).sum()),
             sum(p.numel() for p in model.parameters()))
    return model, df


# ════════════════════════════════════════════════════════════
#  3. Dataset — FIX [1] logmean baseline, FIX [2] observed pv
# ════════════════════════════════════════════════════════════
class PerturbseqDataset(Dataset):
    def __init__(self, adata, gene_names, perturbation_col="gene",
                 control_label="non-targeting", min_cells=5,
                 ensg2sym=None, verbose=True):
        if ensg2sym is None: ensg2sym = {}
        var_ids = adata.var_names.tolist()
        sym2pos = {}
        for pos, ensg in enumerate(var_ids):
            sym = ensg2sym.get(ensg, ensg)
            if sym not in sym2pos: sym2pos[sym] = pos

        hvg_pos, valid_genes = [], []
        for sym in gene_names:
            if sym in sym2pos:
                hvg_pos.append(sym2pos[sym]); valid_genes.append(sym)
        if not valid_genes: raise ValueError("No valid HVG genes found.")

        self.gene_names = valid_genes
        self.gene2idx   = {g: i for i, g in enumerate(valid_genes)}
        self.hvg_pos    = np.array(hvg_pos, dtype=np.int64)

        ctrl_mask = adata.obs[perturbation_col].values == control_label
        self.x0   = torch.tensor(
            _logmean(adata.X[ctrl_mask])[self.hvg_pos], dtype=torch.float32)

        self.samples = []
        for gene_sym, grp in adata.obs.groupby(perturbation_col):
            if gene_sym == control_label or gene_sym not in self.gene2idx: continue
            if len(grp) < min_cells: continue
            gene_idx   = self.gene2idx[gene_sym]
            x_obs_full = _logmean(adata[grp.index].X)[self.hvg_pos]
            pv         = float(x_obs_full[gene_idx])   # FIX [2]: observed KD level
            self.samples.append((gene_idx, pv,
                                  torch.tensor(x_obs_full, dtype=torch.float32)))
        if not self.samples: raise ValueError("No perturbation samples.")
        if verbose:
            print(f"PerturbseqDataset: {len(self.samples)} experiments "
                  f"| {len(valid_genes)} HVGs")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, p, x = self.samples[idx]
        return {"perturbed_idx": int(g), "perturbed_value": float(p),
                "x_obs": x, "x0": self.x0}


def split_dataset(dataset, train_frac=0.70, val_frac=0.10, seed=42):
    n = len(dataset); n_tr = int(train_frac*n); n_v = int(val_frac*n)
    return random_split(dataset, [n_tr, n_v, n-n_tr-n_v],
                        generator=torch.Generator().manual_seed(seed))


# ════════════════════════════════════════════════════════════
#  4. Loss functions
# ════════════════════════════════════════════════════════════
def grnn_loss(x_pred, x_obs, x0, top_k=20,
              lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0,
              lam_deg=2.0,  lam_balance=0.5, gamma_focus=2.0):
    """
    WMSE     : weighted MSE — weight ∝ |Δobs|, prevents mode collapse
    AFDA     : autofocus direction-aware — focal weight on hard examples,
               cosine-based direction agreement, applied to |Δobs| ≥ 0.1
    Balance  : soft up-fraction alignment — prevents systematic act/rep bias
    Delta    : MSE on Δ = x_pred − x0  (avoids Δ=0 shortcut)
    DEG      : top-k importance-weighted MSE
    """
    d_obs = x_obs - x0; d_pred = x_pred - x0

    # WMSE
    w = d_obs.abs() + 1e-6; w = w / w.sum()
    loss_wmse = (w * (x_pred - x_obs).pow(2)).sum()

    # AFDA
    mask = d_obs.abs() >= 0.1
    if mask.any():
        dp = d_pred[mask]; do = d_obs[mask]
        cos    = (dp * do) / ((dp.abs() + 1e-8) * (do.abs() + 1e-8))
        agree  = (1.0 + cos) / 2.0
        fw     = (1.0 - agree).clamp(0, 1).pow(gamma_focus)
        loss_afda = (fw * (dp - do).pow(2)).mean()
    else:
        loss_afda = x_pred.new_tensor(0.0)

    # Balance
    tau = 0.05
    loss_balance = (torch.sigmoid(d_pred/tau).mean()
                    - torch.sigmoid(d_obs/tau).mean()).pow(2)

    # Delta + DEG
    loss_delta = F.mse_loss(d_pred, d_obs)
    k = min(top_k, len(d_obs))
    topk = d_obs.abs().topk(k).indices
    wts  = d_obs.abs()[topk] / (d_obs.abs()[topk].sum() + 1e-12)
    loss_deg = (wts * (x_pred[topk] - x_obs[topk]).pow(2)).sum()

    return (lam_wmse * loss_wmse + lam_afda * loss_afda
            + lam_balance * loss_balance + lam_delta * loss_delta
            + lam_deg * loss_deg)


@torch.no_grad()
def _eval_loss(model, subset, x0, device,
               top_k, lam_wmse, lam_afda, lam_delta,
               lam_deg, lam_balance, gamma_focus):
    model.eval(); total = 0.0
    for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
        x_obs = batch["x_obs"][0].to(device)
        xp, _ = model(x0, int(batch["perturbed_idx"][0]),
                      float(batch["perturbed_value"][0]))
        total += grnn_loss(xp, x_obs, x0, top_k, lam_wmse, lam_afda,
                           lam_delta, lam_deg, lam_balance, gamma_focus).item()
    return total / max(len(subset), 1)


@torch.no_grad()
def _sign_balance_report(model, subset, x0, device):
    model.eval(); pu, ou = [], []
    for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
        x_obs = batch["x_obs"][0].to(device)
        xp, _ = model(x0, int(batch["perturbed_idx"][0]),
                      float(batch["perturbed_value"][0]))
        pu.append((xp - x0 > 0).float().mean().item())
        ou.append((x_obs - x0 > 0).float().mean().item())
    return float(np.mean(pu)), float(np.mean(ou))


@torch.no_grad()
def _convergence_report(model, subset, x0, device):
    model.eval(); iters = []
    for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
        _, n = model(x0, int(batch["perturbed_idx"][0]),
                     float(batch["perturbed_value"][0]))
        iters.append(n)
    iters = np.array(iters)
    return float(np.median(iters)), float((iters == model.max_iter).mean())


def train_grnn(model, train_ds, val_ds, n_epochs=50, lr=1e-3,
               top_k_deg=20, lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0,
               lam_deg=2.0, lam_balance=0.5, gamma_focus=2.0,
               device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    x0    = train_ds.dataset.x0.to(device)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    train_h, val_h = [], []
    for epoch in range(1, n_epochs + 1):
        model.train(); el = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False):
            x_obs = batch["x_obs"][0].to(device)
            opt.zero_grad()
            xp, _ = model(x0, int(batch["perturbed_idx"][0]),
                          float(batch["perturbed_value"][0]))
            loss = grnn_loss(xp, x_obs, x0, top_k_deg, lam_wmse, lam_afda,
                             lam_delta, lam_deg, lam_balance, gamma_focus)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += loss.item()
        sched.step()
        train_h.append(el / len(loader))
        val_h.append(_eval_loss(model, val_ds, x0, device, top_k_deg,
                                lam_wmse, lam_afda, lam_delta, lam_deg,
                                lam_balance, gamma_focus))
        if epoch % 5 == 0 or epoch == 1:
            pu, ou = _sign_balance_report(model, val_ds, x0, device)
            mi, fn = _convergence_report(model, val_ds, x0, device)
            log.info("Ep %3d  tr=%.4f  val=%.4f  bias=%.2f  "
                     "med_it=%.0f  non_conv=%.1f%%",
                     epoch, train_h[-1], val_h[-1], pu-ou, mi, 100*fn)
    return train_h, val_h


def plot_loss_curves(train_h, val_h, cell_line):
    fig, ax = plt.subplots(figsize=(COL_1, COL_1 * 0.62))
    ep = np.arange(1, len(train_h)+1)
    ax.plot(ep, train_h, color="#2A6F97", lw=1.1, label="train")
    ax.plot(ep, val_h,   color="#E07A5F", lw=1.1, label="validation")
    ax.set_xlabel("Epoch", labelpad=2); ax.set_ylabel("Loss", labelpad=2)
    ax.set_yscale("log"); ax.legend(loc="upper right")
    ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout(pad=0.2)
    _save(fig, f"{cell_line}_loss_curves")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════
#  5. Visualisation — ego-graph panels (VIZ_GENES)
# ════════════════════════════════════════════════════════════
def _top_changed_subdf(df, gene, eff_before, eff_after, top_k=10):
    """Ego-graph: edges where gene is source OR target, ranked by |Δeff|."""
    direct = df[(df["source"]==gene)|(df["target"]==gene)].copy()
    if direct.empty: return pd.DataFrame(), set()
    edges  = [(r.source, r.target) for _, r in direct.iterrows()]
    deltas = {e: abs(eff_after.get(e,0)-eff_before.get(e,0)) for e in edges}
    top    = set(sorted(deltas, key=lambda e: deltas[e], reverse=True)[:top_k])
    filt   = direct[direct.apply(lambda r:(r.source,r.target) in top, axis=1)]
    G      = nx.from_pandas_edgelist(filt,"source","target",create_using=nx.DiGraph())
    hop1   = (set(G.successors(gene))|set(G.predecessors(gene)))-{gene}
    return filt, hop1


def _node_hop(n, center, hop1):
    return "center" if n==center else "hop1" if n in hop1 else "hop2"


def _layout(G, gene, hop1):
    hop2   = set(G.nodes)-{gene}-hop1
    shells = [[gene], sorted(hop1&set(G.nodes))]
    if hop2: shells.append(sorted(hop2&set(G.nodes)))
    try:    return nx.shell_layout(G, nlist=[s for s in shells if s])
    except: return nx.kamada_kawai_layout(G)


def _draw_network(G, gene, hop1, *, sign_map, edge_color_map,
                  edge_width_map, edge_label_map,
                  node_sizes_override=None, panel_label, stem):
    pos   = _layout(G, gene, hop1)
    nodes = list(G.nodes())
    sizes = (node_sizes_override if node_sizes_override is not None
             else [NODE_S[_node_hop(n,gene,hop1)] for n in nodes])
    smap  = dict(zip(nodes, sizes))
    edges = list(G.edges())
    solid_e  = [e for e in edges if sign_map.get(e,0)!=0]
    dashed_e = [e for e in edges if sign_map.get(e,0)==0]

    fig, ax = plt.subplots(figsize=(COL_1, COL_1*0.95))
    kw = dict(ax=ax, arrows=True, arrowsize=8,
              connectionstyle="arc3,rad=0.10",
              node_size=[smap[n] for n in nodes])
    if solid_e:
        nx.draw_networkx_edges(G, pos, edgelist=solid_e,
                               edge_color=[edge_color_map[e] for e in solid_e],
                               width=[edge_width_map[e] for e in solid_e],
                               style="solid", alpha=0.92, **kw)
    if dashed_e:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_e,
                               edge_color=[edge_color_map[e] for e in dashed_e],
                               width=[edge_width_map[e] for e in dashed_e],
                               style=(0,(3,2)), alpha=0.75, **kw)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                           node_color=[NODE_C[_node_hop(n,gene,hop1)] for n in nodes],
                           node_size=sizes, edgecolors="white",
                           linewidths=0.8, alpha=0.95)
    for txt in nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5,
                                        font_family="Arial",
                                        font_weight="bold").values():
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_map, ax=ax,
                                  font_size=5.5, font_color=EDGE_LBL,
                                  bbox=dict(facecolor="white",edgecolor="none",
                                            alpha=0.75,pad=0.6), rotate=False)
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.axis("off"); ax.margins(0.10)
    fig.tight_layout(pad=0.2)
    _save(fig, stem); plt.show(); plt.close(fig)


def save_before_jpg(df, gene, cell_line, eff_before, eff_after, top_k=10):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: print(f"  {gene}: no changed edges, skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    edges = list(G.edges())
    sm = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    _draw_network(G, gene, hop1,
                  sign_map      ={e: sm.get(e,0) for e in edges},
                  edge_color_map={e: _before_edge_color(sm.get(e,0)) for e in edges},
                  edge_width_map={e: 1.4 for e in edges},
                  edge_label_map={e: f"{eff_before.get(e,0.0):+.2f}" for e in edges},
                  panel_label="a", stem=f"{cell_line}_{gene}_before")


def save_after_jpg(df, gene, cell_line, eff_before, eff_after, model, top_k=10):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: print(f"  {gene}: no changed edges, skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    V_map = {g: float(v) for g,v in zip(model.gene_names, model.V.detach().cpu().numpy())}
    V_max = max(V_map.values())+1e-9
    edges = list(G.edges())
    em    = {e: eff_after.get(e,0.0) for e in edges}
    e_max = max(abs(v) for v in em.values())+1e-9
    sm = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    ns = [380 if _node_hop(n,gene,hop1)=="center"
          else 80+320*V_map.get(n,1.0)/V_max for n in G.nodes()]
    _draw_network(G, gene, hop1,
                  sign_map           ={e: sm.get(e,0) for e in edges},
                  edge_color_map     ={e: _eff_color(em[e]) for e in edges},
                  edge_width_map     ={e: 0.6+3.0*abs(em[e])/e_max for e in edges},
                  edge_label_map     ={e: f"{em[e]:+.2f}" for e in edges},
                  node_sizes_override=ns,
                  panel_label="b", stem=f"{cell_line}_{gene}_after")


def save_changed_jpg(df, gene, cell_line, eff_before, eff_after, top_k=10):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: print(f"  {gene}: no changed edges, skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    edges = list(G.edges())
    delta = {e: eff_after.get(e,0)-eff_before.get(e,0) for e in edges}
    d_max = max(abs(v) for v in delta.values())+1e-9
    sm = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    _draw_network(G, gene, hop1,
                  sign_map      ={e: sm.get(e,0) for e in edges},
                  edge_color_map={e: _eff_color(eff_after.get(e,0)) for e in edges},
                  edge_width_map={e: 0.6+3.6*abs(delta[e])/d_max for e in edges},
                  edge_label_map={e: f"Δ{delta[e]:+.2f}" for e in edges},
                  panel_label="c", stem=f"{cell_line}_{gene}_changed")


def save_weight_heatmap(eff_weights, label, cell_line, max_label=60):
    if not eff_weights: print("  ⚠ Empty weight map."); return
    rec = [(s,t,w) for (s,t),w in eff_weights.items()]
    df  = pd.DataFrame(rec, columns=["source","target","weight"])
    src = sorted(df["source"].unique()); tgt = sorted(df["target"].unique())
    M   = (df.pivot_table(index="source",columns="target",
                           values="weight",aggfunc="first")
              .reindex(index=src,columns=tgt).fillna(0.0).values)
    ns, nt = len(src), len(tgt)
    vmax = float(np.quantile(np.abs(M),0.98)) or 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    w = min(COL_2, max(COL_1, 0.045*nt+1.4))
    h = min(COL_2, max(COL_1, 0.045*ns+1.0))
    fig, ax = plt.subplots(figsize=(w,h))
    im = ax.imshow(M, aspect="auto", cmap=CMAP_DIV, norm=norm,
                   interpolation="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, aspect=18, pad=0.015)
    cb.set_label(r"$\hat{w}\!\cdot\!\varphi(x_0)$", fontsize=6.5, labelpad=4)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2)
    thr = max_label
    if nt<=thr: ax.set_xticks(range(nt)); ax.set_xticklabels(tgt,rotation=90,fontsize=max(4,380//nt))
    else:       ax.set_xticks([]); ax.text(0.5,-0.04,f"{nt} target genes",transform=ax.transAxes,ha="center",va="top",fontsize=6)
    if ns<=thr: ax.set_yticks(range(ns)); ax.set_yticklabels(src,fontsize=max(4,380//ns))
    else:       ax.set_yticks([]); ax.text(-0.04,0.5,f"{ns} source genes",transform=ax.transAxes,ha="right",va="center",rotation=90,fontsize=6)
    ax.set_xlabel("Target gene",labelpad=3); ax.set_ylabel("Source gene",labelpad=3)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    fig.tight_layout(pad=0.3)
    _save(fig, f"{cell_line}_heatmap_{label}")
    plt.show(); plt.close(fig)
    print(f"  ({ns}×{nt})")


# ════════════════════════════════════════════════════════════
#  6. Target-GRN panels — Cell Press format (GATA1 / MITF)
# ════════════════════════════════════════════════════════════
def _target_subdf(df, gene, rank_eff_map, top_k=30):
    """Outgoing edges gene→targets, ranked by |rank_eff_map|, top-K."""
    sub = df[df["source"]==gene].copy()
    if sub.empty: return pd.DataFrame()
    edges  = [(r.source,r.target) for _,r in sub.iterrows()]
    mag    = {e: abs(rank_eff_map.get(e,0.0)) for e in edges}
    top    = set(sorted(mag, key=lambda e: mag[e], reverse=True)[:top_k])
    return sub[sub.apply(lambda r:(r.source,r.target) in top, axis=1)]


def _radial_layout(G, center):
    others = sorted([n for n in G.nodes if n!=center])
    pos = {center: np.array([0.0,0.0])}
    n   = max(len(others),1)
    for i,node in enumerate(others):
        θ = 2*np.pi*i/n - np.pi/2
        pos[node] = np.array([np.cos(θ), np.sin(θ)])
    return pos


def save_target_plot(df, gene, cell_line, eff_map, rank_eff_map,
                     mode, top_k=30, panel_letter="a"):
    """
    One Cell Press panel (4.49 in square): gene at centre, targets radially.
    eff_map      : EFF_BEFORE or EFF_AFTER  — values displayed
    rank_eff_map : EFF_AFTER always         — edge selection & ranking
    mode         : "before" | "after"
    """
    sub = _target_subdf(df, gene, rank_eff_map, top_k=top_k)
    if sub.empty: print(f"  {gene}: no outgoing edges in GRN — skipped."); return
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    n_tgt = G.number_of_nodes()-1
    pos   = _radial_layout(G, gene)
    edges = list(G.edges())
    sm    = {(r.source,r.target): int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev    = {e: eff_map.get(e,0.0) for e in edges}
    e_max = max(abs(v) for v in ev.values() if v)+1e-9

    ecolors = [_before_edge_color(sm.get(e,0)) if mode=="before"
               else _eff_color(ev[e]) for e in edges]
    ewidths = [0.4+2.6*abs(ev[e])/e_max for e in edges]
    estyles = ["solid" if sm.get(e,0)!=0 else (0,(4,2)) for e in edges]
    elabels = {e: f"{ev[e]:+.2f}" for e in edges}

    solid_e  = [e for e,s in zip(edges,estyles) if s=="solid"]
    dashed_e = [e for e,s in zip(edges,estyles) if s!="solid"]
    solid_c  = [c for c,s in zip(ecolors,estyles) if s=="solid"]
    dashed_c = [c for c,s in zip(ecolors,estyles) if s!="solid"]
    solid_w  = [w for w,s in zip(ewidths,estyles) if s=="solid"]
    dashed_w = [w for w,s in zip(ewidths,estyles) if s!="solid"]

    nc = [NODE_CTR if n==gene else NODE_H1 for n in G.nodes]
    ns = [520 if n==gene else 240 for n in G.nodes]

    fig, ax = plt.subplots(figsize=(CP_COL15, CP_COL15*0.96))
    kw = dict(ax=ax, arrows=True, arrowsize=7,
              connectionstyle="arc3,rad=0.08", node_size=ns)
    if solid_e:
        nx.draw_networkx_edges(G, pos, edgelist=solid_e,
                               edge_color=solid_c, width=solid_w,
                               style="solid", alpha=0.90, **kw)
    if dashed_e:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_e,
                               edge_color=dashed_c, width=dashed_w,
                               style=(0,(4,2)), alpha=0.72, **kw)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes),
                           node_color=nc, node_size=ns,
                           edgecolors="white", linewidths=0.9, alpha=0.97)
    for txt in nx.draw_networkx_labels(G, pos, ax=ax, font_size=6,
                                        font_family="Arial",
                                        font_weight="bold").values():
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, ax=ax,
                                  font_size=4.5, font_color="#2a2a2a",
                                  rotate=False,
                                  bbox=dict(facecolor="white",edgecolor="none",
                                            alpha=0.68,pad=0.45))
    ax.set_aspect("equal"); ax.margins(0.20); ax.axis("off")
    ax.text(-0.04, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left",
            fontfamily="Arial")
    mode_str = ("initial weights · DB sign" if mode=="before"
                else "post-training weights · learned sign")
    ax.text(0.5, -0.03, f"{n_tgt} targets · {mode_str}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color="#555", fontfamily="Arial")

    sm_obj = mpl.cm.ScalarMappable(
        cmap=CMAP_DIV, norm=TwoSlopeNorm(vmin=-e_max,vcenter=0,vmax=e_max))
    sm_obj.set_array([])
    cax = fig.add_axes([0.20,-0.02,0.60,0.018])
    cb  = fig.colorbar(sm_obj, cax=cax, orientation="horizontal")
    cb.set_label(r"$\hat{w}_{ij}\!\cdot\!\varphi(x_0)$", fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([-e_max,0,e_max])
    cb.set_ticklabels([f"−{e_max:.2f}","0",f"+{e_max:.2f}"])

    handles = [
        Line2D([0],[0],color=ACT_C,lw=1.4,label="activation"),
        Line2D([0],[0],color=REP_C,lw=1.4,label="repression"),
        Line2D([0],[0],color=UNK_C,lw=1.4,label="unknown"),
        Line2D([0],[0],color="#555",lw=1.1,ls="solid",label="DB-constrained"),
        Line2D([0],[0],color="#555",lw=1.1,ls=(0,(4,2)),label="sign learned"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=5.5, frameon=False, bbox_to_anchor=(0.5,-0.11),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(rect=[0,0.04,1,1])
    _save(fig, f"{cell_line}_{gene}_{mode}")
    plt.show(); plt.close(fig)
    print(f"  {gene} ({mode}): {n_tgt} targets")


# ── Cytoscape interactive ─────────────────────────────────────────────────────
_CDN_LOADED = False
def render_cytoscape_gene(html_body):
    global _CDN_LOADED
    cdn = ""
    if not _CDN_LOADED:
        cdn = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/'
               'cytoscape/3.28.1/cytoscape.min.js"></script>')
        _CDN_LOADED = True
    display(HTML(f"{cdn}<div style='margin-bottom:28px'>{html_body}</div>"))

def _cy_html(gene, elements_json, cy_id, title, height=480):
    return f"""
<div style="font-family:Arial,sans-serif;font-size:12px;font-weight:bold;
            margin-bottom:2px">{title}</div>
<div id="{cy_id}" style="width:100%;height:{height}px;border:1px solid #ddd;
     border-radius:6px;background:#fafafa"></div>
<script>
(function(){{
  var cy=cytoscape({{container:document.getElementById('{cy_id}'),
    elements:{elements_json},
    layout:{{name:'cose',animate:false,nodeRepulsion:6000,idealEdgeLength:80}},
    style:[
      {{selector:'node',style:{{'label':'data(label)','font-size':9,
        'font-family':'Arial','text-valign':'center','text-halign':'center',
        'width':'data(size)','height':'data(size)','background-color':'data(color)'}}}},
      {{selector:'node[hop="center"]',style:{{'font-size':12,'font-weight':'bold'}}}},
      {{selector:'edge',style:{{'line-color':'data(color)',
        'target-arrow-color':'data(color)','target-arrow-shape':'triangle',
        'curve-style':'bezier','line-style':'data(linestyle)',
        'width':'data(width)','label':'data(label)','font-size':7,
        'font-family':'Arial','text-rotation':'autorotate',
        'color':'#333','opacity':0.9}}}},
    ],wheelSensitivity:0.3}});
}})();
</script>"""

def _before_cy(df, gene, eff_before, eff_after, top_k=10, height=480):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: return f"<p><b>{gene}</b>: not in GRN.</p>"
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    sm = {(r.source,r.target):int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev_b = {(r.source,r.target):eff_before.get((r.source,r.target),0.0) for _,r in sub.iterrows()}
    me   = max(abs(v) for v in ev_b.values())+1e-9
    nodes=[{"data":{"id":n,"label":n,"hop":_node_hop(n,gene,hop1),
                    "color":NODE_C[_node_hop(n,gene,hop1)],
                    "size":{"center":44,"hop1":30,"hop2":22}[_node_hop(n,gene,hop1)]}}
           for n in G.nodes]
    edges=[{"data":{"id":f"{r.source}__{r.target}","source":r.source,"target":r.target,
            "color":_before_edge_color(sm.get((r.source,r.target),0)),
            "linestyle":"solid" if sm.get((r.source,r.target),0)!=0 else "dashed",
            "width":round(1+4*abs(ev_b.get((r.source,r.target),0))/me,2),
            "label":f"{ev_b.get((r.source,r.target),0):+.2f}"}}
           for _,r in sub.iterrows()]
    return _cy_html(gene,json.dumps(nodes+edges),f"cy_{gene}_pre",
                    f"{gene} — before (solid=DB · dashed=learned · label=init eff)",height)

def _after_cy(df, gene, eff_before, eff_after, model, top_k=10, height=480):
    sub, hop1 = _top_changed_subdf(df, gene, eff_before, eff_after, top_k)
    if sub.empty: return f"<p><b>{gene}</b>: not in GRN.</p>"
    G = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    Vm = {g:float(v) for g,v in zip(model.gene_names, model.V.detach().cpu().numpy())}
    Vmx= max(Vm.values())+1e-9
    sm = {(r.source,r.target):int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev_a={e:eff_after.get(e,0.0) for e in [(r.source,r.target) for _,r in sub.iterrows()]}
    me  = max(abs(v) for v in ev_a.values())+1e-9
    nodes=[{"data":{"id":n,"label":n,"hop":_node_hop(n,gene,hop1),
                    "color":NODE_C[_node_hop(n,gene,hop1)],
                    "size":round(20+32*Vm.get(n,1)/Vmx,1)}} for n in G.nodes]
    edges=[{"data":{"id":f"{r.source}__{r.target}","source":r.source,"target":r.target,
            "color":_eff_color(ev_a.get((r.source,r.target),0)),
            "linestyle":"solid" if sm.get((r.source,r.target),0)!=0 else "dashed",
            "width":round(1+7*abs(ev_a.get((r.source,r.target),0))/me,2),
            "label":f"{ev_a.get((r.source,r.target),0):+.2f}"}}
           for _,r in sub.iterrows()]
    return _cy_html(gene,json.dumps(nodes+edges),f"cy_{gene}_post",
                    f"{gene} — after (solid=DB · dashed=learned · label=eff)",height)

def _target_cy(df, gene, eff_map, rank_eff_map,
               mode="after", top_k=30, height=520):
    sub = _target_subdf(df, gene, rank_eff_map, top_k=top_k)
    if sub.empty: return f"<p><b>{gene}</b>: no outgoing edges.</p>"
    G  = nx.from_pandas_edgelist(sub,"source","target",create_using=nx.DiGraph())
    sm = {(r.source,r.target):int(r.get("sign",0)) for _,r in sub.iterrows()}
    ev = {(r.source,r.target):eff_map.get((r.source,r.target),0.0) for _,r in sub.iterrows()}
    me = max(abs(v) for v in ev.values())+1e-9
    nodes=[{"data":{"id":n,"label":n,
                    "color":NODE_CTR if n==gene else NODE_H1,
                    "size":46 if n==gene else 26}} for n in G.nodes]
    edges=[{"data":{"id":f"{r.source}__{r.target}","source":r.source,"target":r.target,
            "color":(_before_edge_color(sm.get((r.source,r.target),0)) if mode=="before"
                     else _eff_color(ev.get((r.source,r.target),0))),
            "linestyle":"solid" if sm.get((r.source,r.target),0)!=0 else "dashed",
            "width":round(1+5*abs(ev.get((r.source,r.target),0))/me,2),
            "label":f"{ev.get((r.source,r.target),0):+.2f}"}}
           for _,r in sub.iterrows()]
    n_tgt = G.number_of_nodes()-1
    ml = "initial eff·DB sign" if mode=="before" else "post-training eff·learned sign"
    elements=json.dumps(nodes+edges)
    return f"""
<div style="font-family:Arial,sans-serif;font-size:12px;font-weight:bold;
            margin-bottom:3px">{gene} — {ml} · {n_tgt} targets</div>
<div id="cy_{gene}_{mode}_tgt" style="width:100%;height:{height}px;
     border:1px solid #ddd;border-radius:5px;background:#fafafa"></div>
<script>
(function(){{var cy=cytoscape({{container:document.getElementById('cy_{gene}_{mode}_tgt'),
  elements:{elements},
  layout:{{name:'concentric',animate:false,
           concentric:function(n){{return n.data('id')==='{gene}'?2:1;}},
           levelWidth:function(){{return 1;}},minNodeSpacing:26}},
  style:[
    {{selector:'node',style:{{'label':'data(label)','font-size':9,'font-family':'Arial',
      'text-valign':'center','text-halign':'center','width':'data(size)','height':'data(size)',
      'background-color':'data(color)','text-outline-width':1.5,
      'text-outline-color':'#fff','color':'#111'}}}},
    {{selector:'node[id="{gene}"]',style:{{'font-size':12,'font-weight':'bold'}}}},
    {{selector:'edge',style:{{'line-color':'data(color)','target-arrow-color':'data(color)',
      'target-arrow-shape':'triangle','curve-style':'bezier',
      'line-style':'data(linestyle)','width':'data(width)','label':'data(label)',
      'font-size':6.5,'font-family':'Arial','text-rotation':'autorotate',
      'color':'#222','opacity':0.9}}}},
  ],wheelSensitivity:0.3}});}})();
</script>"""


# ════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_VIZ = 10     # ego-graph edges for VIZ_GENES
TOP_K_TGT = 30     # target edges for GATA1 / MITF (raise if < 10 targets appear)

# Build dataset (x0 = logmean of control cells — consistent with GRNN init)
dataset = PerturbseqDataset(adata, GENE_NAMES, PERTURBATION_COL,
                             CONTROL_LABEL, min_cells=5, ensg2sym=ensg2sym)
x0_numpy = dataset.x0.numpy()

model, GRN_DF = grn_tsv_to_grnn(OUT_FILE, GENE_NAMES, x0_numpy)
train_ds, val_ds, test_ds = split_dataset(dataset)

n_unk = int((model.signs==0).sum())
print(f"Model  : {model.N} genes | {model.E} edges "
      f"({n_unk} sign=0/learned, {model.E-n_unk} DB-constrained) | "
      f"{sum(p.numel() for p in model.parameters()):,} params")
print(f"Dataset: {len(dataset)} → train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")
print(f"Device : {DEVICE}")

EFF_BEFORE = model._eff_map()

print("\n=== Heatmap — before ===")
save_weight_heatmap(EFF_BEFORE, "before", CELL_LINE)

print("\n=== Ego-graphs — before ===")
_CDN_LOADED = False
for gene in VIZ_GENES:
    print(f"\n── {gene} ──")
    render_cytoscape_gene(_before_cy(GRN_DF, gene, EFF_BEFORE, EFF_BEFORE, TOP_K_VIZ))
    save_before_jpg(GRN_DF, gene, CELL_LINE, EFF_BEFORE, EFF_BEFORE, TOP_K_VIZ)

# ── Train ─────────────────────────────────────────────────────────────────────
train_history, val_history = train_grnn(
    model, train_ds, val_ds, n_epochs=50, lr=1e-3, top_k_deg=20,
    lam_wmse=1.0, lam_afda=1.0, lam_delta=1.0, lam_deg=2.0,
    lam_balance=0.5, gamma_focus=2.0, device=DEVICE)

plot_loss_curves(train_history, val_history, CELL_LINE)

EFF_AFTER = model._eff_map()

print("\n=== Heatmap — after ===")
save_weight_heatmap(EFF_AFTER, "after", CELL_LINE)

print("\n=== Ego-graphs — after + changed ===")
_CDN_LOADED = False
for gene in VIZ_GENES:
    print(f"\n── {gene} ──")
    render_cytoscape_gene(_after_cy(GRN_DF, gene, EFF_BEFORE, EFF_AFTER, model, TOP_K_VIZ))
    save_after_jpg(GRN_DF, gene, CELL_LINE, EFF_BEFORE, EFF_AFTER, model, TOP_K_VIZ)
    save_changed_jpg(GRN_DF, gene, CELL_LINE, EFF_BEFORE, EFF_AFTER, TOP_K_VIZ)

# ── Convergence diagnostics ───────────────────────────────────────────────────
x0_dev = dataset.x0.to(DEVICE)
mi, fn = _convergence_report(model, test_ds, x0_dev, DEVICE)
print(f"\nConvergence (test): median={mi:.0f} iters | non-converged={100*fn:.1f}%")
if fn > 0.05:
    print("  ⚠  >5% non-converged — consider increasing max_iter.")

# ── Target GRN: GATA1 and MITF (4 individual panels) ─────────────────────────
print("\n=== Target GRN plots ===")

print("\n── GATA1 before ──")
render_cytoscape_gene(_target_cy(GRN_DF,"GATA1",EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"GATA1",CELL_LINE,EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT,panel_letter="a")

print("\n── GATA1 after ──")
render_cytoscape_gene(_target_cy(GRN_DF,"GATA1",EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"GATA1",CELL_LINE,EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT,panel_letter="b")

print("\n── MITF before ──")
render_cytoscape_gene(_target_cy(GRN_DF,"MITF",EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"MITF",CELL_LINE,EFF_BEFORE,EFF_AFTER,mode="before",top_k=TOP_K_TGT,panel_letter="c")

print("\n── MITF after ──")
render_cytoscape_gene(_target_cy(GRN_DF,"MITF",EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT))
save_target_plot(GRN_DF,"MITF",CELL_LINE,EFF_AFTER,EFF_AFTER,mode="after",top_k=TOP_K_TGT,panel_letter="d")

print(f"\n✓ All figures in {PLOT_DIR}/")
for pat in ["*.png","*.pdf"]:
    files = sorted(f.name for f in PLOT_DIR.glob(pat))
    if files: print(f"  {pat}: {len(files)} files")

# ════════════════════════════════════════════════════════════
#  Train ONCE, then random-subsample evaluation 10 times
# ════════════════════════════════════════════════════════════

N_EVAL_RUNS      = 10
SUBSAMPLE_FRAC   = 0.30   # evaluate on 30% of test set each run
TOP_K_PEARSON    = 20
LFC_THRESHOLD    = 0.1
EVAL_SEEDS       = list(range(N_EVAL_RUNS))

# --- split once ---
torch.manual_seed(0); np.random.seed(0)
train_ds, val_ds, test_ds = split_dataset(dataset, train_frac=0.70, val_frac=0.10, seed=0)

# --- init model once ---
torch.manual_seed(0)
model, _ = grn_tsv_to_grnn(OUT_FILE, GENE_NAMES, x0_numpy)

# --- train once ---
train_grnn(model, train_ds, val_ds,
           n_epochs=50, lr=1e-3,
           top_k_deg=20, lam_deg=2.0, lam_delta=1.0,
           device=DEVICE)

# ════════════════════════════════════════════════════════════
#  Random subsample evaluation loop
# ════════════════════════════════════════════════════════════
all_metrics = []

for seed in EVAL_SEEDS:
    print(f"\n── Eval run {seed+1}/{N_EVAL_RUNS}  (seed={seed}) ──")

    rng = np.random.default_rng(seed)
    P = len(test_ds)
    m = int(SUBSAMPLE_FRAC * P)

    idx = rng.choice(P, size=m, replace=False)
    test_sub = torch.utils.data.Subset(test_ds, idx)

    # evaluate on random subset
    metrics = evaluate_all(model, test_sub,
                           top_k_pearson=TOP_K_PEARSON,
                           lfc_threshold=LFC_THRESHOLD,
                           device=DEVICE)

    metrics["seed"] = seed
    metrics["n_eval_samples"] = m
    all_metrics.append(metrics)

    print(f"  Pearson Δ{TOP_K_PEARSON}={metrics[f'pearson_delta{TOP_K_PEARSON}_mean']:.4f}  "
          f"centroid={metrics['centroid_accuracy']:.4f}  "
          f"MSE={metrics['mse_delta']:.5f}  "
          f"dir={metrics['directional_accuracy']:.4f}")


# ════════════════════════════════════════════════════════════
#  Aggregate mean ± std across evaluation runs
# ════════════════════════════════════════════════════════════
k = TOP_K_PEARSON
metric_keys = [
    (f"pearson_delta{k}_mean",  f"(a) Pearson Δ{k}  "),
    ("pearson_delta_all_mean",  "(a) Pearson Δ-all"),
    ("centroid_accuracy",       "(b) Centroid acc  "),
    ("mse_delta",               "(c) MSE (delta)   "),
    ("directional_accuracy",    "(d) Directional   "),
]

print(f"\n{'='*60}")
print(f"  iPerturb — {CELL_LINE}  ({N_EVAL_RUNS} random subsample eval runs)")
print(f"  Subsample fraction = {SUBSAMPLE_FRAC:.2f}")
print(f"{'='*60}")

for key, label in metric_keys:
    vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
    print(f"  {label} : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# show threshold info
last = all_metrics[-1]
print(f"\n  Directional threshold : {last['directional_lfc_threshold']}")
print(f"  Gene-pert pairs (last): {last['directional_n_gene_pert_pairs']:,}")

# save results
import pandas as _pd
results_df = _pd.DataFrame(all_metrics)
results_path = f"/content/{CELL_LINE}_metrics_subsample_{N_EVAL_RUNS}runs.tsv"
results_df.to_csv(results_path, sep="\t", index=False)
print(f"\n✓ Per-run metrics saved → {results_path}")

# Results (figures in /content/grn_plots, metrics TSVs and GRN tables in /content)
# are collected/zipped by notebooks/iPerturb_Colab.ipynb (Colab) or retrieved directly.
print("\n✓ iPerturb pipeline complete — all outputs written under /content/.")