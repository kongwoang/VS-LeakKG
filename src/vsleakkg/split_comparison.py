"""Compare shortcut baselines on original vs generated cold splits.

For each (dataset, split variant) we compute:
  - ligand-KNN (same-inchikey hit) AUROC
  - scaffold memorization AUROC
  - contamination-NN AUROC (LIT-PCBA only; reuses confirmed-provenance axes)

The hypothesis: scaffold-cold and strict-cold splits should reduce ligand-KNN
and scaffold-memorization AUROC compared to the original train/val partition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from vsleakkg.metrics import all_metrics

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SPLITS = PROCESSED / "generated_splits"
TABLES = ROOT / "outputs" / "tables" / "final"


def axis_score(val: pl.DataFrame, train: pl.DataFrame, key_col: str) -> np.ndarray:
    """AUROC-friendly score: per-target mean train label for matching key."""
    lut = (train.filter(pl.col(key_col).is_not_null())
                .group_by(["target", key_col])
                .agg(pl.col("label").mean().alias("p_train")))
    v = val.join(lut, on=["target", key_col], how="left")
    return v["p_train"].fill_null(0.0).to_numpy()


def diag_on_split(ex: pl.DataFrame, train_name: str = "train",
                  val_name: str = "validation") -> dict:
    train = ex.filter(pl.col(_split_col(ex)) == train_name)
    val   = ex.filter(pl.col(_split_col(ex)) == val_name)
    if train.is_empty() or val.is_empty():
        return {"ligand_knn_auroc": float("nan"),
                "scaffold_knn_auroc": float("nan"),
                "n_val": val.height,
                "n_pos": int((val["label"] == 1).sum()) if "label" in val.columns else 0}
    y = val["label"].to_numpy().astype(np.int8)
    s_lig = axis_score(val, train, "inchikey")
    s_scf = axis_score(val, train, "scaffold_smiles") if "scaffold_smiles" in val.columns else np.zeros(val.height)
    out = {
        "n_val": val.height,
        "n_pos": int(y.sum()),
        "ligand_knn_auroc":  all_metrics(y, s_lig)["auroc"],
        "ligand_knn_bedroc": all_metrics(y, s_lig)["bedroc"],
        "scaffold_knn_auroc": all_metrics(y, s_scf)["auroc"],
        "scaffold_knn_bedroc": all_metrics(y, s_scf)["bedroc"],
        "ligand_knn_ef1pct": all_metrics(y, s_lig)["ef1pct"],
        "scaffold_knn_ef1pct": all_metrics(y, s_scf)["ef1pct"],
    }
    return out


def _split_col(df: pl.DataFrame) -> str:
    for c in ("generated_split", "split"):
        if c in df.columns:
            return c
    raise ValueError("no split column found")


def main():
    rows = []

    # ---- LIT-PCBA AVE: original vs generated ----
    orig = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet").select(
        "target", "split", "label", "inchikey", "scaffold_smiles")
    m = diag_on_split(orig)
    rows.append({"dataset": "LIT-PCBA AVE", "split": "original_AVE", **m})

    for split_file, label in (("litpcba_scaffold_cold.parquet", "generated_scaffold_cold"),
                              ("litpcba_ligand_cold.parquet",   "generated_ligand_cold"),
                              ("litpcba_strict_cold.parquet",   "generated_strict_cold")):
        df = pl.read_parquet(SPLITS / split_file)
        m = diag_on_split(df)
        rows.append({"dataset": "LIT-PCBA AVE", "split": label, **m})

    # ---- DUD-E: actives-vs-decoys "original" baseline = within-target
    # active/decoy AUROC; vs generated scaffold-cold ----
    dude_orig = pl.read_parquet(PROCESSED / "dude_examples.parquet").select(
        "target", "label", "inchikey", "scaffold_smiles")
    # Treat the whole DUD-E set as "validation" for ligand-knn (no original train/val
    # for the active vs decoy classifier — use the existing shortcut numbers).
    rows.append({"dataset": "DUD-E", "split": "original_actives_vs_decoys",
                 "n_val": dude_orig.height,
                 "n_pos": int((dude_orig["label"] == 1).sum()),
                 "ligand_knn_auroc": float("nan"),
                 "ligand_knn_bedroc": float("nan"),
                 "scaffold_knn_auroc": float("nan"),
                 "scaffold_knn_bedroc": float("nan"),
                 "ligand_knn_ef1pct": float("nan"),
                 "scaffold_knn_ef1pct": float("nan"),
                 "note": "see outputs/tables/dude_shortcut_results.csv for legacy AUROC"})
    df = pl.read_parquet(SPLITS / "dude_scaffold_cold.parquet")
    m = diag_on_split(df)
    rows.append({"dataset": "DUD-E", "split": "generated_scaffold_cold", **m})

    # ---- DEKOIS ----
    rows.append({"dataset": "DEKOIS", "split": "original_actives_vs_decoys",
                 "n_val": pl.read_parquet(PROCESSED / "dekois_examples.parquet").height,
                 "n_pos": int((pl.read_parquet(PROCESSED / "dekois_examples.parquet")["label"] == 1).sum()),
                 "ligand_knn_auroc": float("nan"),
                 "ligand_knn_bedroc": float("nan"),
                 "scaffold_knn_auroc": float("nan"),
                 "scaffold_knn_bedroc": float("nan"),
                 "ligand_knn_ef1pct": float("nan"),
                 "scaffold_knn_ef1pct": float("nan"),
                 "note": "see outputs/tables/dekois_shortcut_results.csv for legacy AUROC"})
    df = pl.read_parquet(SPLITS / "dekois_scaffold_cold.parquet")
    m = diag_on_split(df)
    rows.append({"dataset": "DEKOIS", "split": "generated_scaffold_cold", **m})

    # ---- BayesBind ----
    df = pl.read_parquet(SPLITS / "bayesbind_scaffold_cold.parquet")
    m = diag_on_split(df)
    rows.append({"dataset": "BayesBind V1.5", "split": "generated_scaffold_cold", **m})

    out = pl.DataFrame(rows)
    out.write_csv(TABLES / "table14_original_vs_generated_split_diagnostics.csv")
    print(f"[cmp] wrote table14 ({out.height} rows)")
    print(out.to_pandas().to_string())


if __name__ == "__main__":
    main()
