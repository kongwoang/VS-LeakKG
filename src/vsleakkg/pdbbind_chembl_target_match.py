"""PDBBind → ChEMBL target sequence matching.

Steps:
  1. Extract ChEMBL component_sequences for SINGLE PROTEIN targets, writing
     a FASTA where the header is the canonical target_chembl_id.
  2. Run `mmseqs search` against the PDBBind protein FASTA.
  3. Convert to TSV via `convertalis`.
  4. Bucket matches into confirmed_high / confirmed_medium / candidate_family.
  5. Materialise PDBBind→ChEMBL nodes/edges.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables" / "final"
TABLES.mkdir(parents=True, exist_ok=True)
CHEMBL_DB = ROOT / "data" / "raw" / "chembl" / "extracted" / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"
MMSEQS_BAT = Path("C:/Tools/mmseqs2/mmseqs/mmseqs.bat")
TMP_DIR = PROCESSED / "tmp_mmseqs_chembl"
TMP_DIR.mkdir(exist_ok=True)


def dump_chembl_component_sequences(out_fasta: Path) -> int:
    """Pull (target_chembl_id, sequence) for SINGLE-PROTEIN ChEMBL targets.

    A target can have >1 component; we emit one FASTA record per
    (target_chembl_id, component_id) so all sequences are searchable. The
    PDBBind matcher will collapse hits back to target_chembl_id.
    """
    conn = sqlite3.connect(f"file:{CHEMBL_DB.as_posix()}?mode=ro", uri=True)
    q = """
    SELECT td.chembl_id   AS target_chembl_id,
           tc.component_id,
           cs.sequence    AS seq
    FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences cs ON tc.component_id = cs.component_id
    WHERE td.target_type = 'SINGLE PROTEIN'
      AND cs.sequence IS NOT NULL AND length(cs.sequence) > 30
    """
    n = 0
    with out_fasta.open("w") as fh:
        for tcid, cid, seq in conn.execute(q):
            seq = seq.strip().replace(" ", "").replace("\n", "")
            if not seq:
                continue
            fh.write(f">{tcid}|{cid}\n")
            # wrap at 80
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")
            n += 1
    conn.close()
    return n


def run_mmseqs_search(query_fasta: Path, target_fasta: Path,
                      out_tsv: Path) -> None:
    """mmseqs createdb + search + convertalis. Sensitive enough to find
    distant homologs but not so sensitive it explodes runtime."""
    qdb = TMP_DIR / "qdb"
    tdb = TMP_DIR / "tdb"
    resdb = TMP_DIR / "result"
    cmds = [
        [str(MMSEQS_BAT), "createdb", str(query_fasta), str(qdb)],
        [str(MMSEQS_BAT), "createdb", str(target_fasta), str(tdb)],
        [str(MMSEQS_BAT), "search", str(qdb), str(tdb), str(resdb), str(TMP_DIR / "mmtmp"),
         "-s", "5.0", "--threads", "4", "-e", "1e-5", "--max-seqs", "20"],
        [str(MMSEQS_BAT), "convertalis", str(qdb), str(tdb), str(resdb), str(out_tsv),
         "--format-output",
         "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen"],
    ]
    for c in cmds:
        print("[match]   $", " ".join(c[2:5]))
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            print("STDERR:", r.stderr[-1000:])
            raise RuntimeError(f"mmseqs command failed: {c}")


def classify_match(pident: float, qcov: float, tcov: float, evalue: float) -> str:
    cov = min(qcov, tcov)
    if pident >= 0.90 and cov >= 0.80:
        return "confirmed_high"
    if pident >= 0.50 and cov >= 0.80:
        return "confirmed_medium"
    if pident >= 0.30 and cov >= 0.60:
        return "candidate_family"
    if evalue <= 1e-10:
        return "low_confidence_homolog"
    return "discard"


def parse_tsv(tsv: Path) -> pl.DataFrame:
    if not tsv.exists():
        return pl.DataFrame()
    df = pl.read_csv(tsv, separator="\t", has_header=False, new_columns=[
        "query", "target", "pident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits", "qlen", "tlen"
    ])
    if df.height == 0:
        return df
    # ChEMBL FASTA headers are "<target_chembl_id>|<component_id>" — split.
    df = df.with_columns([
        pl.col("target").str.split("|").list.first().alias("target_chembl_id"),
        pl.col("target").str.split("|").list.last().alias("component_id"),
        (pl.col("alnlen") / pl.col("qlen")).alias("qcov"),
        (pl.col("alnlen") / pl.col("tlen")).alias("tcov"),
    ])
    df = df.with_columns(
        pl.struct(["pident", "qcov", "tcov", "evalue"])
          .map_elements(lambda r: classify_match(r["pident"]/100.0 if r["pident"] > 1 else r["pident"],
                                                 r["qcov"], r["tcov"], r["evalue"]),
                        return_dtype=pl.Utf8)
          .alias("confidence")
    )
    return df


def collapse_to_best(df: pl.DataFrame) -> pl.DataFrame:
    """Keep best hit per (query, target_chembl_id) by bits."""
    return (df.sort("bits", descending=True)
              .group_by(["query", "target_chembl_id"])
              .first()
              .sort("query", "bits", descending=[False, True]))


def main():
    pdb_fasta = PROCESSED / "pdbbind_proteins.fasta"
    chembl_fasta = PROCESSED / "chembl_component_sequences.fasta"
    out_tsv = PROCESSED / "pdbbind_to_chembl_alignments.tsv"
    out_match = PROCESSED / "pdbbind_to_chembl_sequence_matches.parquet"
    out_targetmap = PROCESSED / "pdbbind_to_chembl_target_map.parquet"

    if not chembl_fasta.exists() or chembl_fasta.stat().st_size < 1_000_000:
        n = dump_chembl_component_sequences(chembl_fasta)
        print(f"[match] ChEMBL FASTA: {n} sequences  ({chembl_fasta.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"[match] reusing existing {chembl_fasta} ({chembl_fasta.stat().st_size/1e6:.1f} MB)")

    if not out_tsv.exists() or out_tsv.stat().st_size < 1000:
        run_mmseqs_search(pdb_fasta, chembl_fasta, out_tsv)
    else:
        print(f"[match] reusing existing {out_tsv}")
    print(f"[match] TSV size: {out_tsv.stat().st_size/1e6:.1f} MB")

    df = parse_tsv(out_tsv)
    print(f"[match] raw alignments: {df.height}")
    if df.height == 0:
        return

    # Collapse: keep ONE row per (query=PDBBind seq, target_chembl_id), best bits.
    best = collapse_to_best(df)
    best.write_parquet(out_match)
    print(f"[match] best per (pdb_seq, chembl_target): {best.height}")

    # confidence-bucket breakdown
    print(best.group_by("confidence").len().sort("len", descending=True))

    # Target map: best ChEMBL target per PDBBind sequence
    tm = (best.sort("bits", descending=True)
              .group_by("query")
              .first()
              .select("query", "target_chembl_id", "component_id",
                      "pident", "qcov", "tcov", "evalue", "bits", "confidence")
              .rename({"query": "pdbbind_seq_id"}))
    tm.write_parquet(out_targetmap)
    print(f"[match] target map rows: {tm.height}")

    # ------ Materialise graph nodes/edges ------
    nodes = []
    edges = []
    # PDBBindProtein nodes already exist in MVP-2; we don't duplicate.
    # ChEMBLComponent nodes
    for cid in best["component_id"].unique().drop_nulls().to_list():
        nodes.append({"node_id": f"chembl_comp:{cid}", "node_type": "ChEMBLComponent"})
    # Edges: PDBBindProtein -> ChEMBLComponent + PDBBindProtein -> ChEMBLTarget
    for r in best.iter_rows(named=True):
        if r["confidence"] in ("discard",):
            continue
        edges.append({
            "src": f"pdbbind_prot:{r['query']}",
            "dst": f"chembl_comp:{r['component_id']}",
            "edge_type": "matches_chembl_component",
            "confidence": r["confidence"],
            "pident": r["pident"],
        })
        edges.append({
            "src": f"pdbbind_prot:{r['query']}",
            "dst": f"chembl_tgt:{r['target_chembl_id']}",
            "edge_type": "maps_to_chembl_target",
            "confidence": r["confidence"],
            "pident": r["pident"],
        })

    # PDBBind complex -> ChEMBL target via best-protein map
    if (PROCESSED / "pdbbind_complexes.parquet").exists():
        complexes = pl.read_parquet(PROCESSED / "pdbbind_complexes.parquet").select(
            "pdb_id", "protein_sequence_concat")
        prot = pl.read_parquet(PROCESSED / "pdbbind_proteins.parquet").with_columns(
            pl.col("seq_sha256").str.slice(0, 16).alias("seq_id"))
        complexes = complexes.join(prot.select("sequence_concat", "seq_id"),
                                    left_on="protein_sequence_concat",
                                    right_on="sequence_concat", how="left")
        complexes = complexes.join(tm, left_on="seq_id", right_on="pdbbind_seq_id", how="left")
        confirmed_complex = complexes.filter(
            pl.col("confidence").is_in(["confirmed_high", "confirmed_medium"])
        )
        print(f"[match] PDBBind complexes with confirmed ChEMBL target: "
              f"{confirmed_complex.height} / {complexes.height}")
        for r in confirmed_complex.iter_rows(named=True):
            edges.append({
                "src": f"pdbbind_complex:{r['pdb_id']}",
                "dst": f"chembl_tgt:{r['target_chembl_id']}",
                "edge_type": "candidate_from_chembl_target",
                "confidence": r["confidence"],
                "pident": r["pident"],
            })

    pl.DataFrame(nodes).write_parquet(PROCESSED / "mvp2_plus_pdbbind_chembl_target_nodes.parquet")
    pl.DataFrame(edges).write_parquet(PROCESSED / "mvp2_plus_pdbbind_chembl_target_edges.parquet")
    print(f"[match] new nodes: {len(nodes)}, new edges: {len(edges)}")

    # Table 16
    summary = (best.group_by("confidence")
               .agg([
                   pl.len().alias("n_pairs"),
                   pl.col("query").n_unique().alias("n_pdbbind_seqs"),
                   pl.col("target_chembl_id").n_unique().alias("n_chembl_targets"),
                   pl.col("pident").mean().alias("mean_pident"),
                   pl.col("evalue").min().alias("min_evalue"),
               ]).sort("n_pairs", descending=True))
    summary.write_csv(TABLES / "table16_pdbbind_chembl_target_match_summary.csv")
    print("[match] table16 written")


if __name__ == "__main__":
    main()
