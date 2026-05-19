"""Weighted path-product contamination score for LIT-PCBA AVE.

Implements the proposal's S(π) = prod_e w_r(e), C_axis = max_{train} S(π),
and C_total = max over axes.

A val example is considered to have a path on axis A if it shares the
relevant entity (ligand InChIKey / scaffold / ChEMBL/BDB/PDBBind ligand /
confirmed assay / confirmed document / candidate assay / candidate
document / source / time-bin) with ANY training example.

The weight table lives in `data/processed/edge_type_weights.parquet` and
is intentionally tiny so it can be hand-edited.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


DEFAULT_WEIGHTS: dict[str, float] = {
    "ligand_identity":          1.00,
    "same_inchikey":            0.98,
    "same_scaffold":            0.60,
    "analog_tanimoto_ge_0.9":   0.90,
    "analog_tanimoto_ge_0.8":   0.80,
    "analog_tanimoto_ge_0.6":   0.60,
    "protein_cluster_90":       0.90,
    "protein_cluster_50":       0.70,
    "protein_cluster_30":       0.50,
    "pocket_cluster_fine":      0.85,
    "pocket_cluster_mid":       0.70,
    "pocket_cluster_coarse":    0.50,
    "same_chembl_ligand":       0.85,
    "same_bindingdb_ligand":    0.80,
    "same_pdbbind_ligand":      0.85,
    "confirmed_same_assay":     0.90,
    "confirmed_same_document":  0.75,
    "candidate_assay":          0.50,
    "candidate_document":       0.40,
    "same_source":              0.20,
    "same_decoy_protocol":      0.50,
    "same_timebin":             0.30,
}


def write_weight_table() -> pl.DataFrame:
    df = pl.DataFrame([{"edge_or_axis": k, "weight": v} for k, v in DEFAULT_WEIGHTS.items()])
    df.write_parquet(PROCESSED / "edge_type_weights.parquet")
    df.write_csv(ROOT / "outputs" / "tables" / "final" / "edge_type_weights.csv")
    return df


def compute_litpcba() -> pl.DataFrame:
    print("[wc] loading inputs…")
    pf = pl.read_parquet(PROCESSED / "mvp2_path_features_litpcba.parquet")
    print(f"[wc] LIT-PCBA AVE val rows: {pf.height}")

    # Confirmed assay/document join: which val ligands have a confirmed assay/doc
    # link that is shared with some train ligand for the SAME target?
    conf = pl.read_parquet(PROCESSED / "benchmark_chembl_confirmed_provenance.parquet")
    conf_litpcba = conf.filter(pl.col("benchmark_dataset") == "LIT-PCBA AVE")
    print(f"[wc] confirmed LIT-PCBA rows: {conf_litpcba.height}")

    # For each (target, inchikey) val pair, count confirmed assays/documents
    conf_counts = (conf_litpcba
        .group_by("inchikey")
        .agg([
            pl.col("assay_chembl_id").n_unique().alias("path_confirmed_assay_count"),
            pl.col("document_chembl_id").n_unique().alias("path_confirmed_document_count"),
        ]))

    pf = pf.join(conf_counts, on="inchikey", how="left").with_columns([
        pl.col("path_confirmed_assay_count").fill_null(0),
        pl.col("path_confirmed_document_count").fill_null(0),
    ])

    # Weight lookup
    W = DEFAULT_WEIGHTS

    # For each axis: hit-flag * weight (binary OR semantics; max over train is
    # implicitly 1 if any path exists for that axis).
    def hit(col: str, w: float) -> pl.Expr:
        return (pl.when(pl.col(col) > 0).then(pl.lit(w)).otherwise(pl.lit(0.0)))

    pf = pf.with_columns([
        hit("path_identity_train_count",          W["ligand_identity"]).alias("C_identity_weighted"),
        hit("path_scaffold_train_count",          W["same_scaffold"]).alias("C_scaffold_weighted"),
        hit("path_analog_train_max",              W["analog_tanimoto_ge_0.6"]).alias("C_analog_weighted"),
        hit("path_chembl_ligand_train_count",     W["same_chembl_ligand"]).alias("C_chembl_weighted"),
        hit("path_bindingdb_ligand_train_count",  W["same_bindingdb_ligand"]).alias("C_bindingdb_weighted"),
        hit("path_pdbbind_same_ligand_count",     W["same_pdbbind_ligand"]).alias("C_pdbbind_lig_weighted"),
        hit("path_pdbbind_same_scaffold_count",   W["same_scaffold"] * W["same_pdbbind_ligand"]).alias("C_pdbbind_scf_weighted"),
        hit("path_candidate_assay_train_count",   W["candidate_assay"]).alias("C_assay_candidate_weighted"),
        hit("path_candidate_document_train_count", W["candidate_document"]).alias("C_doc_candidate_weighted"),
        hit("path_confirmed_assay_count",         W["confirmed_same_assay"]).alias("C_assay_confirmed_weighted"),
        hit("path_confirmed_document_count",      W["confirmed_same_document"]).alias("C_doc_confirmed_weighted"),
        # Source: every LIT-PCBA val example trivially shares the LIT-PCBA source
        # with every LIT-PCBA train example. We assign the same_source weight.
        pl.lit(W["same_source"]).alias("C_source_weighted"),
        # Decoy protocol: all LIT-PCBA AVE examples share the AVE-debiased
        # decoy protocol. Constant.
        pl.lit(W["same_decoy_protocol"]).alias("C_decoy_protocol_weighted"),
        # Time: all LIT-PCBA examples share the LIT-PCBA2020 release bin.
        pl.lit(W["same_timebin"]).alias("C_time_weighted"),
    ])

    # Protein-cluster axis is left empty for now: the val→train protein-cluster
    # path would require linking LIT-PCBA target names to PDBBind protein
    # clusters, which we did not materialise (the curated target dictionary
    # maps to ChEMBL targets, not PDBBind chains). Conservative 0.0.
    pf = pf.with_columns(pl.lit(0.0).alias("C_protein_weighted"),
                         pl.lit(0.0).alias("C_pocket_weighted"))

    # Strict bundle: axes the proposal calls "high-confidence evidence":
    # identity, confirmed-assay, confirmed-document, scaffold, pdbbind-ligand.
    strict_axes = ["C_identity_weighted", "C_scaffold_weighted",
                   "C_pdbbind_lig_weighted",
                   "C_assay_confirmed_weighted", "C_doc_confirmed_weighted"]

    # Candidate bundle: + cross-source ligand overlaps + candidate assay/doc.
    candidate_axes = strict_axes + [
        "C_analog_weighted",
        "C_chembl_weighted", "C_bindingdb_weighted",
        "C_pdbbind_scf_weighted",
        "C_assay_candidate_weighted", "C_doc_candidate_weighted",
        "C_source_weighted", "C_decoy_protocol_weighted", "C_time_weighted",
    ]

    # Max aggregation
    pf = pf.with_columns([
        pl.max_horizontal(strict_axes).alias("C_total_weighted_strict"),
        pl.max_horizontal(candidate_axes).alias("C_total_weighted_candidate"),
    ])

    # Weighted-sum aggregation (rough alternative: average of non-zero axes,
    # capped at 1.0)
    pf = pf.with_columns([
        pl.sum_horizontal(strict_axes).alias("_strict_sum"),
        pl.sum_horizontal(candidate_axes).alias("_cand_sum"),
    ]).with_columns([
        (pl.col("_strict_sum") / float(len(strict_axes))).clip(0.0, 1.0).alias("C_total_weighted_strict_avg"),
        (pl.col("_cand_sum") / float(len(candidate_axes))).clip(0.0, 1.0).alias("C_total_weighted_candidate_avg"),
    ]).drop("_strict_sum", "_cand_sum")

    return pf


def main():
    wt = write_weight_table()
    print(f"[wc] weight table rows: {wt.height}")
    df = compute_litpcba()
    out = PROCESSED / "litpcba_weighted_contamination_scores.parquet"
    df.write_parquet(out)
    print(f"[wc] wrote {out} ({df.height} rows × {len(df.columns)} cols)")

    # Per-target summary table (table7)
    agg = (df.group_by("target")
           .agg([
               pl.col("C_total_weighted_strict").mean().alias("mean_strict"),
               pl.col("C_total_weighted_candidate").mean().alias("mean_candidate"),
               pl.col("C_total_weighted_strict_avg").mean().alias("mean_strict_avg"),
               pl.col("C_total_weighted_candidate_avg").mean().alias("mean_candidate_avg"),
               (pl.col("C_total_weighted_strict") > 0.5).cast(pl.Float64).mean().alias("frac_strict_gt_0.5"),
               (pl.col("C_total_weighted_candidate") > 0.5).cast(pl.Float64).mean().alias("frac_candidate_gt_0.5"),
               pl.len().alias("n_val"),
           ]).sort("mean_strict"))
    agg.write_csv(ROOT / "outputs" / "tables" / "final" / "table7_weighted_contamination_by_target.csv")
    print("[wc] table7 written")

    # Aggregate one-liner
    print(f"[wc] overall strict={float(df['C_total_weighted_strict'].mean()):.4f}  "
          f"candidate={float(df['C_total_weighted_candidate'].mean()):.4f}  "
          f"strict_avg={float(df['C_total_weighted_strict_avg'].mean()):.4f}  "
          f"candidate_avg={float(df['C_total_weighted_candidate_avg'].mean()):.4f}")


if __name__ == "__main__":
    main()
