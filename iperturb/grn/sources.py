"""Evidence-level crawlers for GRN construction.

Level 1 — curated TF-target databases (TRRUST, OmniPath, CollecTRI)
Level 2 — enhancer-based edges (GeneHancer TFBS)
Level 3 — PPI transitive closure (STRING)
Level 4 — co-expression (COXPRESdb v8 bulk)
"""

from __future__ import annotations

import logging
import re
import time
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils import make_session, robust_get

log = logging.getLogger("iperturb.grn")
warnings.filterwarnings("ignore")

_EMPTY = pd.DataFrame(columns=["source", "target", "sign", "level", "db"])

_SESSION = make_session()


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1a — TRRUST v2
# ──────────────────────────────────────────────────────────────────────────────

def crawl_trrust(gene_set: set) -> pd.DataFrame:
    """Download and filter the TRRUST v2 human TF-target database."""
    url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
    log.info("TRRUST → %s", url)
    r = robust_get(_SESSION, url)
    rows = []
    for line in r.text.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        tf, tgt, itype = parts[0].upper(), parts[1].upper(), parts[2]
        if tf not in gene_set or tgt not in gene_set:
            continue
        sign = (1  if "activation" in itype.lower() else
                -1 if "repression" in itype.lower() else 0)
        rows.append({"source": tf, "target": tgt, "sign": sign, "level": 1, "db": "TRRUST"})
    df = pd.DataFrame(rows)
    log.info("  → %d edges", len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1b — OmniPath / CollecTRI (library + Enrichr fallback)
# ──────────────────────────────────────────────────────────────────────────────

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
                df_raw = op.interactions.Dorothea.get(dorothea_levels=["A", "B"], genesymbols=True)
            else:
                df_raw = op.interactions.AllInteractions.get(datasets=[ds], genesymbols=True)

            for _, row in df_raw.iterrows():
                src = str(row.get("source_genesymbol", "")).upper()
                tgt = str(row.get("target_genesymbol", "")).upper()
                if src not in gene_set or tgt not in gene_set:
                    continue
                try:
                    sign = int(row.get("consensus_stimulation") or 0) - \
                           int(row.get("consensus_inhibition")  or 0)
                except Exception:
                    sign = 0
                rows.append({"source": src, "target": tgt, "sign": sign, "level": 1, "db": label})
        except Exception as exc:
            log.debug("  omnipath lib %s failed: %s", ds, exc)
    return pd.DataFrame(rows) if rows else _EMPTY.copy()


def _enrichr_fallback(gene_set: set, libraries: list[str], label: str) -> pd.DataFrame:
    import gseapy as gp
    rows = []
    for lib in libraries:
        try:
            log.info("  Enrichr fallback: %s", lib)
            enr = gp.get_library(lib, organism="Human")
            for tf, targets in enr.items():
                tf = tf.split(" ")[0].upper()
                if tf not in gene_set:
                    continue
                for tgt in targets:
                    tgt = tgt.upper()
                    if tgt in gene_set and tgt != tf:
                        rows.append({"source": tf, "target": tgt, "sign": 0,
                                     "level": 1, "db": f"Enrichr:{lib}"})
        except Exception as exc:
            log.debug("  Enrichr %s: %s", lib, exc)
    if not rows:
        return _EMPTY.copy()
    return pd.DataFrame(rows).drop_duplicates(["source", "target"])


def crawl_omnipath_robust(
    gene_set: set,
    datasets: list[str],
    label: str,
    cache_dir: str,
) -> pd.DataFrame:
    """
    Fetch TF-target interactions from OmniPath (omnipath library).
    Falls back to Enrichr if the library path fails or returns nothing.
    """
    df = _EMPTY.copy()
    try:
        df = _omnipath_via_lib(gene_set, datasets, label)
    except Exception as exc:
        log.warning("Lib path failed: %s", exc)

    if df.empty:
        log.warning("  lib empty → falling back to Enrichr")
        try:
            df = _enrichr_fallback(
                gene_set,
                ["ChEA_2022",
                 "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
                 "TF_Perturbations_Followed_by_Expression"],
                label,
            )
        except Exception as exc:
            log.warning("Enrichr fallback failed: %s", exc)

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2 — GeneHancer (GFF + TFBS + Tissue filter)
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_RE = re.compile(r"^(ENSG|ENSM|lnc-|piR-|LOC\d|HSALNG|FAM\d)", re.IGNORECASE)


def _parse_gff_gene_associations(gff_path: str) -> pd.DataFrame:
    records = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9 or parts[2].lower() != "enhancer":
                continue
            attr_str = parts[8]
            m = re.search(r"genehancer_id=([^;]+)", attr_str)
            if not m:
                continue
            gh_id = m.group(1).strip()
            for gm in re.finditer(r"connected_gene=([^;]+);score=([\d.]+)", attr_str):
                gene = gm.group(1).strip().upper()
                if not gene or _SKIP_RE.match(gene):
                    continue
                try:
                    sc = float(gm.group(2))
                except Exception:
                    sc = 0.0
                records.append({"enhancer_id": gh_id, "gene": gene, "score": sc})
    return pd.DataFrame(records)


def _parse_tfbs(tfbs_path: str, tissue_filter: str = "") -> pd.DataFrame:
    df = pd.read_csv(tfbs_path, sep="\t", comment="#", dtype=str)
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "ghid":
            col_map[c] = "ghid"
        elif cl == "tf":
            col_map[c] = "tf"
        elif "tissue" in cl:
            col_map[c] = "tissues"
    df = df.rename(columns=col_map)[["ghid", "tf", "tissues"]]
    df["ghid"]    = df["ghid"].str.strip()
    df["tf"]      = df["tf"].str.strip().str.upper()
    df["tissues"] = df["tissues"].fillna("")
    if tissue_filter:
        df = df[df["tissues"].str.contains(tissue_filter, case=False, na=False)]
    return df[["ghid", "tf"]].drop_duplicates()


def _parse_tissues(tissue_path: str, filter_str: str = "") -> set:
    active = set()
    with open(tissue_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gh_id, tissue = parts[0].strip(), parts[2].strip()
            if not filter_str or filter_str.lower() in tissue.lower():
                active.add(gh_id)
    return active


def parse_genehancer(
    gene_set: set,
    gff_path: str,
    tfbs_path: str,
    tissue_path: str,
    tissue_filter: str = "",
) -> pd.DataFrame:
    """Build Level-2 edges from GeneHancer GFF + TFBS + Tissue files."""
    gff_df  = _parse_gff_gene_associations(gff_path)
    tfbs_df = _parse_tfbs(tfbs_path, tissue_filter)
    active  = _parse_tissues(tissue_path, tissue_filter)

    if tfbs_df.empty:
        return _EMPTY.copy()

    gff_sub = gff_df[gff_df["enhancer_id"].isin(active)] if active else gff_df
    tfbs_df = tfbs_df.rename(columns={"ghid": "enhancer_id"})
    merged  = gff_sub.merge(tfbs_df, on="enhancer_id", how="inner")

    rows = []
    for _, row in merged.iterrows():
        tf, tgt = row["tf"], row["gene"]
        if tf in gene_set and tgt in gene_set and tf != tgt:
            rows.append({"source": tf, "target": tgt, "sign": 0, "level": 2, "db": "GeneHancer_TFBS"})

    if not rows:
        return _EMPTY.copy()
    return pd.DataFrame(rows).drop_duplicates(["source", "target"])


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3 — STRING PPI + transitive closure
# ──────────────────────────────────────────────────────────────────────────────

def crawl_string_ppi(gene_set: set, min_score: int = 700) -> pd.DataFrame:
    """Retrieve undirected PPI edges from STRING for genes in gene_set."""
    from tqdm import tqdm

    url_map = "https://string-db.org/api/json/get_string_ids"
    url_net = "https://string-db.org/api/json/network"
    genes, id_map = sorted(gene_set), {}

    for i in tqdm(range(0, len(genes), 500), desc="STRING id-map"):
        batch = genes[i:i + 500]
        try:
            r = _SESSION.post(
                url_map,
                data={"identifiers": "\r".join(batch), "species": 9606,
                      "limit": 1, "echo_query": 1, "caller_identity": "iperturb"},
                timeout=60,
            )
            r.raise_for_status()
            for rec in r.json():
                sym, sid = rec.get("queryItem", "").upper(), rec.get("stringId", "")
                if sym and sid:
                    id_map[sym] = sid
        except Exception as exc:
            log.debug("STRING id_map batch %d: %s", i, exc)
        time.sleep(0.5)

    if not id_map:
        return _EMPTY.copy()

    rev, rows, ids = {v: k for k, v in id_map.items()}, [], list(id_map.values())

    for i in tqdm(range(0, len(ids), 500), desc="STRING network"):
        try:
            r = _SESSION.post(
                url_net,
                data={"identifiers": "\r".join(ids[i:i + 500]), "species": 9606,
                      "required_score": min_score, "caller_identity": "iperturb"},
                timeout=120,
            )
            r.raise_for_status()
            for itx in r.json():
                a = rev.get(itx.get("stringId_A", ""), "").upper()
                b = rev.get(itx.get("stringId_B", ""), "").upper()
                if a in gene_set and b in gene_set and a != b:
                    rows += [
                        {"source": a, "target": b, "sign": 0, "level": 3, "db": "STRING"},
                        {"source": b, "target": a, "sign": 0, "level": 3, "db": "STRING"},
                    ]
        except Exception as exc:
            log.debug("STRING net batch %d: %s", i, exc)
        time.sleep(0.5)

    if not rows:
        return _EMPTY.copy()
    return pd.DataFrame(rows).drop_duplicates(["source", "target"])


def build_l3_transitive(l2: pd.DataFrame, ppi: pd.DataFrame) -> pd.DataFrame:
    """
    Level-3 transitive edges: for each L2 edge A→B and PPI A–C,
    add C→B (a PPI partner of an L2 TF can regulate the same targets).
    """
    if l2.empty or ppi.empty:
        return _EMPTY.copy()
    partners: dict[str, set] = {}
    for _, row in ppi.iterrows():
        partners.setdefault(row["source"], set()).add(row["target"])
    rows = []
    for _, row in l2.iterrows():
        A, B = row["source"], row["target"]
        for C in partners.get(A, set()):
            if C != B:
                rows.append({"source": C, "target": B, "sign": 0, "level": 3, "db": "L2+PPI"})
    if not rows:
        return _EMPTY.copy()
    return pd.DataFrame(rows).drop_duplicates(["source", "target"])


# ──────────────────────────────────────────────────────────────────────────────
#  Level 4 — COXPRESdb v8 bulk (co-expression)
# ──────────────────────────────────────────────────────────────────────────────

_COL_PATTERNS = {
    "source": ["gene1", "query",  "source",  "gene_a", "symbol1", "id1"],
    "target": ["gene2", "target", "partner", "gene_b", "symbol2", "id2"],
    "mr":     ["mr", "mutual_rank", "mutualrank", "rank", "score"],
}


def _detect_col(header: list[str], patterns: list[str], fallback: int) -> int:
    for i, col in enumerate(header):
        if any(p.lower() in col.lower().strip() for p in patterns):
            return i
    return fallback


def _normalise_id(raw: str, gene_id_type: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    if gene_id_type == "entrez":
        return raw if raw.isdigit() else None
    return raw.upper()


def parse_coxpresdb_bulk(
    gene_set: set,
    top_n: int = 5,
    chunk_limit: Optional[int] = None,
    gene_id_type: str = "symbol",
    zip_path: str = "",
    extract_dir: str = "",
) -> pd.DataFrame:
    """
    Parse COXPRESdb v8 bulk data and return top-N co-expressed partners per
    gene, ranked by mutual rank (MR; lower = stronger co-expression).

    Parameters
    ----------
    gene_set      : set of gene symbols (uppercase)
    top_n         : strongest partners to keep per source gene
    chunk_limit   : stop after this many .d files (None = all)
    gene_id_type  : "symbol" | "entrez" | "ensembl"
    zip_path      : path to Hsa_union_coex.zip
    extract_dir   : where to extract; defaults to <zip_path>_extract/
    """
    if not gene_set:
        log.error("gene_set is empty.")
        return _EMPTY.copy()

    if not zip_path or not Path(zip_path).exists():
        log.warning("COXPRESdb zip not found at '%s' — skipping L4.", zip_path)
        return _EMPTY.copy()

    out_dir = Path(extract_dir) if extract_dir else Path(zip_path).with_suffix("")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        log.info("✓ Extraction complete.")

    chunk_files = sorted(p for p in out_dir.iterdir() if p.suffix == ".d" and p.stem.isdigit())
    if not chunk_files:
        chunk_files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name.isdigit())
    if not chunk_files:
        log.error("No numeric .d files found in %s", out_dir)
        return _EMPTY.copy()

    log.info("COXPRESdb: %d chunk files; gene_id_type=%s", len(chunk_files), gene_id_type)

    collected: list[dict] = []
    for files_done, path in enumerate(chunk_files):
        if chunk_limit is not None and files_done >= chunk_limit:
            break
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header_line = f.readline().strip()
                if not header_line:
                    continue
                header = header_line.split("\t")
                q_idx  = _detect_col(header, _COL_PATTERNS["source"], fallback=0)
                t_idx  = _detect_col(header, _COL_PATTERNS["target"], fallback=1)
                mr_idx = _detect_col(header, _COL_PATTERNS["mr"],     fallback=-1)
                if q_idx == t_idx:
                    t_idx = q_idx + 1

                for row_num, line in enumerate(f):
                    parts = line.strip().split("\t")
                    if len(parts) < max(q_idx, t_idx) + 1:
                        continue
                    src = _normalise_id(parts[q_idx], gene_id_type)
                    tgt = _normalise_id(parts[t_idx], gene_id_type)
                    if src is None or tgt is None or src == tgt:
                        continue
                    if src not in gene_set or tgt not in gene_set:
                        continue
                    mr = (float(parts[mr_idx])
                          if mr_idx != -1 and mr_idx < len(parts)
                          else float(row_num))
                    try:
                        mr = float(mr)
                    except ValueError:
                        mr = float(row_num)
                    collected.append({"source": src, "target": tgt, "mr": mr})
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)

    if not collected:
        return _EMPTY.copy()

    raw_df = pd.DataFrame(collected).sort_values("mr")
    raw_df = raw_df.groupby("source", sort=False).head(top_n).reset_index(drop=True)

    fwd  = raw_df[["source", "target"]].copy()
    rev  = raw_df.rename(columns={"source": "target", "target": "source"})[["source", "target"]].copy()
    edges = (pd.concat([fwd, rev], ignore_index=True)
               .drop_duplicates(subset=["source", "target"])
               .reset_index(drop=True))
    edges["sign"]  = 0
    edges["level"] = 4
    edges["db"]    = "COXPRESdb"
    log.info("COXPRESdb: %d final edges | top-%d by MR", len(edges), top_n)
    return edges
