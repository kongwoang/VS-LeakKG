"""Multi-axis cold split generator (polars-bulk).

Strategy: instead of iterating rows or groups in Python, we:
  1. compute unique group keys + sizes via group_by().agg()
  2. assign partitions to groups deterministically using `hash(seed,key) % 100`
     thresholds (no Python loop over groups — vectorised)
  3. left-join the group→partition table back to the row table

The hash-based assignment is approximate (it does not actively balance label
ratio), but for cold splits we mainly care that no group spans two partitions,
and the train/val/test fractions match overall ~70/15/15 by the law of large
numbers when the number of groups is large.

If a benchmark has very few groups (e.g. PDBBind protein clusters ~3,300),
the law-of-large-numbers approximation can drift; we report the achieved
partition sizes so the user can inspect.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "generated_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "outputs" / "tables" / "final"

SEED = 17
TRAIN_PCT, VAL_PCT = 70, 15  # rest -> test


def _hash_assign_expr(group_col: str, name: str) -> pl.Expr:
    """Return a polars Expr that maps `group_col` -> 'train'/'validation'/'test'
    deterministically based on hash(seed+name+key) % 100."""
    salt = f"{SEED}|{name}"
    # polars hash() with deterministic seeds
    return (
        (pl.col(group_col).cast(pl.Utf8).fill_null("__NULL__") + "|" + pl.lit(salt))
        .hash(seed=0)
        .mod(100)
        .alias("_hash100")
    )


def _make_split(df: pl.DataFrame, group_col: str, name: str) -> pl.DataFrame:
    df = df.with_columns(_hash_assign_expr(group_col, name))
    df = df.with_columns(
        pl.when(pl.col("_hash100") < TRAIN_PCT).then(pl.lit("train"))
          .when(pl.col("_hash100") < TRAIN_PCT + VAL_PCT).then(pl.lit("validation"))
          .otherwise(pl.lit("test"))
          .alias("generated_split")
    ).drop("_hash100")
    return df


def _summary(df: pl.DataFrame, group_col: str) -> dict:
    parts = df.group_by("generated_split").agg([
        pl.len().alias("n_rows"),
        pl.col("label").mean().alias("active_ratio") if "label" in df.columns
            else pl.lit(None).alias("active_ratio"),
        pl.col(group_col).n_unique().alias("n_groups"),
        pl.col("target").n_unique().alias("n_targets") if "target" in df.columns
            else pl.lit(None).alias("n_targets"),
    ]).sort("generated_split")
    return {r["generated_split"]: r for r in parts.iter_rows(named=True)}


def generate_litpcba_splits() -> dict[str, dict]:
    ex = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet").select(
        "target", "split", "label", "inchikey", "smiles_canonical",
        "scaffold_smiles", "source_file"
    )
    print(f"[gen] LIT-PCBA AVE total rows: {ex.height}")
    out = {}

    df = _make_split(ex, "scaffold_smiles", "litpcba_scaffold_cold")
    df.write_parquet(OUT_DIR / "litpcba_scaffold_cold.parquet")
    out["litpcba_scaffold_cold"] = _summary(df, "scaffold_smiles")
    print("[gen]   scaffold_cold done")

    df = _make_split(ex, "inchikey", "litpcba_ligand_cold")
    df.write_parquet(OUT_DIR / "litpcba_ligand_cold.parquet")
    out["litpcba_ligand_cold"] = _summary(df, "inchikey")
    print("[gen]   ligand_cold done")

    ex2 = ex.with_columns((pl.col("scaffold_smiles").cast(pl.Utf8).fill_null("") + "|" +
                           pl.col("target")).alias("_strict_key"))
    df = _make_split(ex2, "_strict_key", "litpcba_strict_cold")
    df.write_parquet(OUT_DIR / "litpcba_strict_cold.parquet")
    out["litpcba_strict_cold"] = _summary(df, "_strict_key")
    print("[gen]   strict_cold done")
    return out


def generate_dude_splits() -> dict[str, dict]:
    ex = pl.read_parquet(PROCESSED / "dude_examples.parquet").select(
        "target", "label", "inchikey", "smiles_canonical", "scaffold_smiles"
    ).with_columns(pl.lit("dude").alias("source_file"))
    print(f"[gen] DUD-E rows: {ex.height}")
    df = _make_split(ex, "scaffold_smiles", "dude_scaffold_cold")
    df.write_parquet(OUT_DIR / "dude_scaffold_cold.parquet")
    return {"dude_scaffold_cold": _summary(df, "scaffold_smiles")}


def generate_dekois_splits() -> dict[str, dict]:
    ex = pl.read_parquet(PROCESSED / "dekois_examples.parquet").select(
        "target", "label", "inchikey", "smiles_canonical", "scaffold_smiles", "source_file"
    )
    print(f"[gen] DEKOIS rows: {ex.height}")
    df = _make_split(ex, "scaffold_smiles", "dekois_scaffold_cold")
    df.write_parquet(OUT_DIR / "dekois_scaffold_cold.parquet")
    return {"dekois_scaffold_cold": _summary(df, "scaffold_smiles")}


def generate_pdbbind_splits() -> dict[str, dict]:
    pdb = pl.read_parquet(PROCESSED / "pdbbind_complexes.parquet").select(
        "pdb_id", "ligand_inchikey", "ligand_scaffold_smiles", "protein_sequence_concat"
    )
    clu30 = pl.read_parquet(PROCESSED / "pdbbind_protein_clusters_30.parquet").select(
        "seq_id", pl.col("cluster_id").alias("protein_cluster_30")
    )
    prot = pl.read_parquet(PROCESSED / "pdbbind_proteins.parquet").with_columns(
        pl.col("seq_sha256").str.slice(0, 16).alias("seq_id"))
    pdb = pdb.join(prot.select("sequence_concat", "seq_id"),
                   left_on="protein_sequence_concat", right_on="sequence_concat",
                   how="left").join(clu30, on="seq_id", how="left")
    pdb = pdb.with_columns(
        pl.lit(0, dtype=pl.Int8).alias("label"),
        pl.col("ligand_inchikey").alias("inchikey"),
        pl.col("ligand_scaffold_smiles").alias("scaffold_smiles"),
        pl.lit("pdbbind").alias("source_file"),
        pl.col("pdb_id").alias("target"))
    print(f"[gen] PDBBind rows: {pdb.height}")
    df = _make_split(pdb, "protein_cluster_30", "pdbbind_protein_cold")
    df.write_parquet(OUT_DIR / "pdbbind_ligand_scaffold_protein_cold.parquet")
    return {"pdbbind_protein_cold_30": _summary(df, "protein_cluster_30")}


def generate_bayesbind_splits() -> dict[str, dict]:
    bb = pl.read_parquet(PROCESSED / "bayesbind_examples.parquet").select(
        "target", "split", "label", "inchikey", "smiles_canonical",
        "scaffold_smiles", "source_file"
    )
    print(f"[gen] BayesBind rows: {bb.height}")
    df = _make_split(bb, "scaffold_smiles", "bayesbind_scaffold_cold")
    df.write_parquet(OUT_DIR / "bayesbind_scaffold_cold.parquet")
    return {"bayesbind_scaffold_cold": _summary(df, "scaffold_smiles")}


def main():
    summaries = {}
    summaries.update(generate_litpcba_splits())
    summaries.update(generate_dude_splits())
    summaries.update(generate_dekois_splits())
    summaries.update(generate_pdbbind_splits())
    summaries.update(generate_bayesbind_splits())

    rows = []
    for split_name, parts in summaries.items():
        for part_name, r in parts.items():
            rows.append({"split_name": split_name, "partition": part_name, **r})
    pl.DataFrame(rows).write_csv(TABLES / "table13_generated_split_summary.csv")
    print(f"[gen] table13 written ({len(rows)} rows)")

    for split_name, parts in summaries.items():
        print(f"\n[{split_name}]")
        for part_name, r in parts.items():
            ar = r.get('active_ratio')
            ar_s = f"{ar:.4f}" if ar is not None else "-"
            print(f"  {part_name:>12}: n={r['n_rows']:>8,}  groups={r['n_groups']:>7,}  active_ratio={ar_s}")


if __name__ == "__main__":
    main()
