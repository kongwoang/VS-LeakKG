"""Contamination-NN baseline.

For each LIT-PCBA AVE val example, score it by the mean label of the train
neighbors connected through the highest-weight available axis.

For each axis A with weight w_A:
  score_A(val) = w_A × mean(train_label | train shares A-feature with val,
                                          same benchmark target)

Final prediction:
  score(val) = max over A of score_A(val)

This is a tractable surrogate for the proposal's
  arg max_{x_i ∈ D_train} C(x_t, {x_i}) → transfer label.
Picking the train_i that maximises C reduces to: pick the axis with highest
w_A that has any matching train, and transfer the matching train's label
(or the mean over multiple matches).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from vsleakkg.metrics import all_metrics
from vsleakkg.weighted_contamination import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_CSV_PER_TARGET = ROOT / "outputs" / "tables" / "final" / "table8_contamination_nn_metrics.csv"


def per_axis_train_label_lookup(train: pl.DataFrame,
                                 key_col: str,
                                 target_col: str = "target") -> pl.DataFrame:
    """Returns one row per (target, key) with mean train label."""
    return (train.filter(pl.col(key_col).is_not_null())
            .group_by([target_col, key_col])
            .agg([
                pl.col("label").mean().alias(f"plabel_{key_col}"),
                pl.len().alias(f"ntrain_{key_col}"),
            ]))


def axis_score(val: pl.DataFrame, train: pl.DataFrame,
               key_col: str, weight: float) -> pl.DataFrame:
    """Return val with two new columns: score_<key>, hit_<key>."""
    lut = per_axis_train_label_lookup(train, key_col)
    val2 = val.join(lut, on=["target", key_col], how="left")
    val2 = val2.with_columns([
        (pl.col(f"plabel_{key_col}").fill_null(0.0) * weight).alias(f"score_{key_col}"),
        pl.col(f"ntrain_{key_col}").fill_null(0).alias(f"ntrain_{key_col}_filled"),
    ])
    return val2.drop(f"plabel_{key_col}", f"ntrain_{key_col}").rename({
        f"ntrain_{key_col}_filled": f"ntrain_{key_col}"
    })


def main() -> None:
    print("[cnn] loading LIT-PCBA AVE examples…")
    ex = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet").select(
        "target", "split", "label", "inchikey", "smiles_canonical", "scaffold_smiles"
    )
    train = ex.filter(pl.col("split") == "train")
    val = ex.filter(pl.col("split") == "validation")
    print(f"[cnn] train={train.height}  val={val.height}")

    # ---- Axis 1: same inchikey ----
    val = axis_score(val, train, "inchikey", DEFAULT_WEIGHTS["same_inchikey"])

    # ---- Axis 2: same scaffold ----
    val = axis_score(val, train, "scaffold_smiles", DEFAULT_WEIGHTS["same_scaffold"])

    # ---- Axis 3: same canonical SMILES (identity) ----
    val = axis_score(val, train, "smiles_canonical", DEFAULT_WEIGHTS["ligand_identity"])

    # ---- Axis 4: same ChEMBL ligand (molregno) — for both train + val ----
    print("[cnn] loading ChEMBL ligand map…")
    cmap = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet").filter(
        pl.col("benchmark_dataset") == "LIT-PCBA AVE"
    ).select("inchikey", "molregno").unique()
    train_with_chembl = train.join(cmap, on="inchikey", how="left")
    val_with_chembl = val.join(cmap, on="inchikey", how="left")
    val = axis_score(val_with_chembl, train_with_chembl, "molregno",
                     DEFAULT_WEIGHTS["same_chembl_ligand"])

    # ---- Axis 5: same BindingDB ligand (inchikey -> bdb existence flag) ----
    # The val inchikey already maps; we just flag whether the train side has
    # the same inchikey AND it's in BindingDB.
    bdbmap = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet").filter(
        pl.col("benchmark_dataset") == "LIT-PCBA AVE"
    ).select("inchikey").unique().with_columns(pl.lit(True).alias("in_bdb"))
    train_with_bdb = train.join(bdbmap, on="inchikey", how="left").with_columns(
        pl.col("in_bdb").fill_null(False))
    val_with_bdb = val.join(bdbmap, on="inchikey", how="left").with_columns(
        pl.col("in_bdb").fill_null(False))
    # Use "in_bdb" as the axis key.
    val = axis_score(val_with_bdb.with_columns(pl.col("in_bdb").cast(pl.Utf8).alias("bdb_key")),
                     train_with_bdb.with_columns(pl.col("in_bdb").cast(pl.Utf8).alias("bdb_key")),
                     "bdb_key",
                     DEFAULT_WEIGHTS["same_bindingdb_ligand"])

    # ---- Axis 6: same confirmed assay ----
    conf = pl.read_parquet(PROCESSED / "benchmark_chembl_confirmed_provenance.parquet").filter(
        pl.col("benchmark_dataset") == "LIT-PCBA AVE"
    ).select("inchikey", "assay_chembl_id").unique()
    # For each (target, inchikey) we attach the set of confirmed assays.
    # Train rows with assay X have label l. Mean label = neighbor probability.
    train_assay = train.join(conf, on="inchikey", how="inner")
    val_assay = val.join(conf, on="inchikey", how="inner")
    if train_assay.height and val_assay.height:
        lut = (train_assay.group_by(["target", "assay_chembl_id"])
               .agg(pl.col("label").mean().alias("plabel_assay"),
                    pl.len().alias("ntrain_assay")))
        val_assay = val_assay.join(lut, on=["target", "assay_chembl_id"], how="left")
        # For each val row, max plabel × weight across its confirmed assays.
        assay_per_row = (val_assay.group_by(["target", "inchikey"])
                         .agg([
                             (pl.col("plabel_assay").fill_null(0.0).max()
                              * DEFAULT_WEIGHTS["confirmed_same_assay"]).alias("score_assay_conf"),
                             pl.col("ntrain_assay").fill_null(0).sum().alias("ntrain_assay_conf"),
                         ]))
        val = val.join(assay_per_row, on=["target", "inchikey"], how="left").with_columns([
            pl.col("score_assay_conf").fill_null(0.0),
            pl.col("ntrain_assay_conf").fill_null(0),
        ])
    else:
        val = val.with_columns(pl.lit(0.0).alias("score_assay_conf"),
                               pl.lit(0).alias("ntrain_assay_conf"))

    # ---- Axis 7: same confirmed document ----
    conf_doc = pl.read_parquet(PROCESSED / "benchmark_chembl_confirmed_provenance.parquet").filter(
        pl.col("benchmark_dataset") == "LIT-PCBA AVE"
    ).select("inchikey", "document_chembl_id").unique()
    train_doc = train.join(conf_doc, on="inchikey", how="inner")
    val_doc = val.join(conf_doc, on="inchikey", how="inner")
    if train_doc.height and val_doc.height:
        lut = (train_doc.group_by(["target", "document_chembl_id"])
               .agg(pl.col("label").mean().alias("plabel_doc"),
                    pl.len().alias("ntrain_doc")))
        val_doc = val_doc.join(lut, on=["target", "document_chembl_id"], how="left")
        doc_per_row = (val_doc.group_by(["target", "inchikey"])
                       .agg([
                           (pl.col("plabel_doc").fill_null(0.0).max()
                            * DEFAULT_WEIGHTS["confirmed_same_document"]).alias("score_doc_conf"),
                           pl.col("ntrain_doc").fill_null(0).sum().alias("ntrain_doc_conf"),
                       ]))
        val = val.join(doc_per_row, on=["target", "inchikey"], how="left").with_columns([
            pl.col("score_doc_conf").fill_null(0.0),
            pl.col("ntrain_doc_conf").fill_null(0),
        ])
    else:
        val = val.with_columns(pl.lit(0.0).alias("score_doc_conf"),
                               pl.lit(0).alias("ntrain_doc_conf"))

    # ---- Combine: cnn score = max of axis scores; best axis label ----
    axis_score_cols = [c for c in val.columns if c.startswith("score_")]
    val = val.with_columns(pl.max_horizontal(axis_score_cols).alias("cnn_score"))
    # Determine which axis fired (for reporting)
    val = val.with_columns(
        pl.struct(axis_score_cols).map_elements(
            lambda r: max(r.items(), key=lambda kv: (kv[1] or 0.0))[0],
            return_dtype=pl.Utf8,
        ).alias("cnn_best_axis")
    )

    # Persist
    out = PROCESSED / "litpcba_contamination_nn_predictions.parquet"
    val.write_parquet(out)
    print(f"[cnn] wrote {out} (rows={val.height}, cols={len(val.columns)})")

    # Per-target metrics
    metric_rows = []
    for tgt in sorted(val["target"].unique().to_list()):
        sub = val.filter(pl.col("target") == tgt)
        m = all_metrics(sub["label"].to_numpy(), sub["cnn_score"].to_numpy())
        metric_rows.append({"target": tgt, "diagnostic": "contamination_nn", **m})
    # Aggregate row
    m_all = all_metrics(val["label"].to_numpy(), val["cnn_score"].to_numpy())
    metric_rows.append({"target": "ALL", "diagnostic": "contamination_nn", **m_all})

    pl.DataFrame(metric_rows).write_csv(OUT_CSV_PER_TARGET)
    print(f"[cnn] wrote {OUT_CSV_PER_TARGET}")

    # Headline aggregate
    aucs = [r["auroc"] for r in metric_rows if r["target"] != "ALL" and not np.isnan(r["auroc"])]
    print(f"[cnn] mean AUROC across {len(aucs)} targets: {np.mean(aucs):.4f}")
    print(f"[cnn] median AUROC: {np.median(aucs):.4f}")
    print(f"[cnn] overall (concatenated): AUROC={m_all['auroc']:.4f}  AP={m_all['ap']:.4f}  BEDROC={m_all['bedroc']:.4f}")


if __name__ == "__main__":
    main()
