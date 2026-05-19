"""Contamination decile and worst-group reports for LIT-PCBA AVE."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from vsleakkg.metrics import all_metrics

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables" / "final"


def main() -> None:
    wc = pl.read_parquet(PROCESSED / "litpcba_weighted_contamination_scores.parquet")
    cnn = pl.read_parquet(PROCESSED / "litpcba_contamination_nn_predictions.parquet")
    print(f"[dwg] wc rows: {wc.height}, cnn rows: {cnn.height}")

    # Join on (target, inchikey) — both keyed the same way
    df = wc.join(cnn.select("target", "inchikey", "cnn_score", "cnn_best_axis"),
                 on=["target", "inchikey"], how="left")
    print(f"[dwg] joined rows: {df.height}")

    # ---- Decile analysis ----
    decile_rows = []
    for score_col in ("C_total_weighted_strict",
                      "C_total_weighted_candidate",
                      "cnn_score"):
        if score_col not in df.columns:
            continue
        # qcut into deciles (10 bins, label 1..10)
        # Use polars quantile-based binning via rank approach
        s = df.with_columns(
            (pl.col(score_col).rank(method="ordinal", descending=False) /
             pl.col(score_col).count() * 10.0).ceil().cast(pl.Int64).clip(1, 10).alias("decile")
        )
        agg = (s.group_by("decile")
               .agg([
                   pl.col("label").mean().alias("active_ratio"),
                   pl.col(score_col).mean().alias("mean_score"),
                   pl.col(score_col).min().alias("min_score"),
                   pl.col(score_col).max().alias("max_score"),
                   pl.col("label").sum().alias("n_active"),
                   pl.len().alias("n_examples"),
               ])
               .sort("decile"))
        for r in agg.iter_rows(named=True):
            decile_rows.append({"score_axis": score_col, **r})
    pl.DataFrame(decile_rows).write_csv(TABLES / "table11_contamination_decile_report.csv")
    print(f"[dwg] table11 written ({len(decile_rows)} rows)")

    # ---- Worst-group analysis ----
    worst_rows = []

    # 1. By target
    for tgt in sorted(df["target"].unique().to_list()):
        sub = df.filter(pl.col("target") == tgt)
        m_cnn = all_metrics(sub["label"].to_numpy(), sub["cnn_score"].to_numpy())
        m_str = all_metrics(sub["label"].to_numpy(), sub["C_total_weighted_strict"].to_numpy())
        worst_rows.append({"group_type": "target", "group_id": tgt,
                           "n_examples": sub.height,
                           "active_ratio": float(sub["label"].mean()),
                           "cnn_auroc": m_cnn["auroc"],
                           "cnn_bedroc": m_cnn["bedroc"],
                           "cnn_ef1pct": m_cnn["ef1pct"],
                           "strict_auroc": m_str["auroc"]})

    # 2. By scaffold frequency bucket: rare/medium/frequent
    sc_counts = df.group_by("scaffold_smiles").len().rename({"len": "scf_freq"})
    df2 = df.join(sc_counts, on="scaffold_smiles", how="left")
    df2 = df2.with_columns(
        pl.when(pl.col("scf_freq") <= 5).then(pl.lit("rare<=5"))
        .when(pl.col("scf_freq") <= 50).then(pl.lit("medium6_50"))
        .otherwise(pl.lit("frequent>50")).alias("scf_bucket")
    )
    for b in ("rare<=5", "medium6_50", "frequent>50"):
        sub = df2.filter(pl.col("scf_bucket") == b)
        if sub.is_empty():
            continue
        m_cnn = all_metrics(sub["label"].to_numpy(), sub["cnn_score"].to_numpy())
        worst_rows.append({"group_type": "scaffold_frequency", "group_id": b,
                           "n_examples": sub.height,
                           "active_ratio": float(sub["label"].mean()),
                           "cnn_auroc": m_cnn["auroc"],
                           "cnn_bedroc": m_cnn["bedroc"],
                           "cnn_ef1pct": m_cnn["ef1pct"],
                           "strict_auroc": float("nan")})

    # 3. By contamination decile (strict)
    s = df.with_columns(
        (pl.col("C_total_weighted_strict").rank(method="ordinal", descending=False) /
         pl.col("C_total_weighted_strict").count() * 10.0).ceil().cast(pl.Int64).clip(1, 10).alias("strict_decile")
    )
    for d in range(1, 11):
        sub = s.filter(pl.col("strict_decile") == d)
        if sub.is_empty():
            continue
        m_cnn = all_metrics(sub["label"].to_numpy(), sub["cnn_score"].to_numpy())
        worst_rows.append({"group_type": "strict_decile", "group_id": f"D{d}",
                           "n_examples": sub.height,
                           "active_ratio": float(sub["label"].mean()),
                           "cnn_auroc": m_cnn["auroc"],
                           "cnn_bedroc": m_cnn["bedroc"],
                           "cnn_ef1pct": m_cnn["ef1pct"],
                           "strict_auroc": float("nan")})

    # 4. By best-axis the CNN baseline used
    for ax in df["cnn_best_axis"].unique().drop_nulls().to_list():
        sub = df.filter(pl.col("cnn_best_axis") == ax)
        if sub.is_empty():
            continue
        m_cnn = all_metrics(sub["label"].to_numpy(), sub["cnn_score"].to_numpy())
        worst_rows.append({"group_type": "cnn_best_axis", "group_id": ax,
                           "n_examples": sub.height,
                           "active_ratio": float(sub["label"].mean()),
                           "cnn_auroc": m_cnn["auroc"],
                           "cnn_bedroc": m_cnn["bedroc"],
                           "cnn_ef1pct": m_cnn["ef1pct"],
                           "strict_auroc": float("nan")})

    pl.DataFrame(worst_rows).write_csv(TABLES / "table12_worst_group_report.csv")
    print(f"[dwg] table12 written ({len(worst_rows)} rows)")

    # Persist enriched parquet too
    s.write_parquet(PROCESSED / "litpcba_contamination_deciles.parquet")
    print("[dwg] decile parquet written")


if __name__ == "__main__":
    main()
