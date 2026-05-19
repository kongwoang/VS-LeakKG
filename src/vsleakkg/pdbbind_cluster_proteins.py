"""PDBBind protein clustering — MMseqs2 path (and exact-sequence fallback).

The orchestrator that runs `mmseqs easy-cluster` lives outside this module
(see scripts/cluster_pdbbind.ps1 or run-only invocation in
src/vsleakkg/build_protein_clusters.py); this module is responsible for
parsing the `*_cluster.tsv` outputs into the parquets the rest of the pipeline
expects.

TSV format (mmseqs easy-cluster):
    representative_id  TAB  member_id   (one row per member, repeated reps)

Parquet schema (per threshold):
    seq_id        : str  — chain-concat sha256 prefix (matches FASTA header)
    cluster_id    : str  — representative id, prefixed with `pclu_<pct>_`
    cluster_size  : i64
    rep_seq_id    : str  — same content as cluster_id sans prefix
    seq_len       : i64  — null when no FASTA is loaded
    method        : str  — "mmseqs2_easy_cluster" or "exact_sequence_fallback"
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha1
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FASTA = PROCESSED / "pdbbind_proteins.fasta"


def read_fasta_lengths(p: Path) -> dict[str, int]:
    """Returns seq_id -> sequence length."""
    out: dict[str, int] = {}
    cur_id = None
    cur_len = 0
    with p.open() as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    out[cur_id] = cur_len
                cur_id = line[1:].split()[0]
                cur_len = 0
            else:
                cur_len += len(line)
    if cur_id is not None:
        out[cur_id] = cur_len
    return out


def parse_cluster_tsv(tsv_path: Path, pct: int) -> pl.DataFrame:
    """Reads representative<TAB>member TSV. Cluster id is derived from
    representative id so the column is stable across re-runs."""
    df = pl.read_csv(
        tsv_path, separator="\t", has_header=False,
        new_columns=["rep_seq_id", "seq_id"], infer_schema_length=0,
    )
    sizes = df.group_by("rep_seq_id").len().rename({"len": "cluster_size"})
    df = df.join(sizes, on="rep_seq_id")
    df = df.with_columns([
        (pl.lit(f"pclu_{pct}_") + pl.col("rep_seq_id")).alias("cluster_id"),
        pl.lit("mmseqs2_easy_cluster").alias("method"),
    ])
    return df.select("seq_id", "cluster_id", "cluster_size", "rep_seq_id", "method")


def main() -> None:
    lens = read_fasta_lengths(FASTA)
    print(f"[cluster] fasta sequences: {len(lens)}")

    written = {}
    for pct in (90, 50, 30):
        tsv = PROCESSED / f"pdbbind_clu_{pct}_cluster.tsv"
        if not tsv.exists():
            print(f"[cluster]   SKIP {pct}%: {tsv.name} not found")
            continue
        df = parse_cluster_tsv(tsv, pct)
        df = df.with_columns(
            pl.col("seq_id").replace(lens, default=None).cast(pl.Int64).alias("seq_len")
        )
        out = PROCESSED / f"pdbbind_protein_clusters_{pct}.parquet"
        df.write_parquet(out)
        n_clu = df["cluster_id"].n_unique()
        n_ns = df.filter(pl.col("cluster_size") > 1).height
        largest = int(df["cluster_size"].max())
        print(f"[cluster]   {pct}%: rows={df.height}  clusters={n_clu}  "
              f"non_singleton={n_ns}  largest={largest}")
        written[pct] = df

    # Combined wide parquet for convenience.
    if written:
        base = next(iter(written.values())).select("seq_id").unique()
        combined = base.clone()
        for pct, df in sorted(written.items()):
            combined = combined.join(
                df.select(
                    "seq_id",
                    pl.col("cluster_id").alias(f"cluster_id_{pct}"),
                    pl.col("cluster_size").alias(f"cluster_size_{pct}"),
                    pl.col("rep_seq_id").alias(f"rep_seq_id_{pct}"),
                ),
                on="seq_id", how="left",
            )
        combined = combined.with_columns(pl.lit("mmseqs2_easy_cluster").alias("method"))
        combined.write_parquet(PROCESSED / "pdbbind_protein_clusters_combined.parquet")
        print(f"[cluster] combined parquet rows={combined.height}")


if __name__ == "__main__":
    main()
