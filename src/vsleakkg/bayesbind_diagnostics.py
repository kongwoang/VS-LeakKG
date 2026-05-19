"""BayesBind shortcut diagnostics — within-benchmark (val + test only).

Each (target, split) is scored against the OTHER split for the same target as
the "reference" set. A diagnostic that is predictive of `label=1` on
target X's test split using only reference-set features indicates a shortcut.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_CSV = ROOT / "outputs" / "tables" / "final" / "bayesbind_shortcut_results.csv"


def _enrich_factor(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.01) -> float:
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return float("nan")
    k = max(1, int(n * frac))
    idx = np.argsort(-y_score)[:k]
    hits = y_true[idx].sum()
    return float((hits / k) / (y_true.sum() / n))


def safe_auroc(y, s):
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def safe_ap(y, s):
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def run() -> pl.DataFrame:
    """BayesBind shortcut diagnostics — within-benchmark.

    Targets are partitioned across val/test (no target appears in both), so
    this is an *active-vs-decoy* shortcut analysis, not train-test leakage:

      For each (target, split): can a feature that is independent of the
      benchmark label nonetheless rank actives above decoys?
    """
    bb = pl.read_parquet(PROCESSED / "bayesbind_examples.parquet")
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet").select(
        "standard_inchi_key").drop_nulls()
    chembl_set = set(chembl_lig["standard_inchi_key"].to_list())
    bdb_lig = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet").select(
        "ligand_inchikey").drop_nulls()
    bdb_set = set(bdb_lig["ligand_inchikey"].to_list())
    pdb_lig = pl.read_parquet(PROCESSED / "pdbbind_ligands.parquet").select(
        "inchikey").drop_nulls()
    pdb_set = set(pdb_lig["inchikey"].to_list())

    # Cross-BayesBind-target leakage: build per-ligand and per-scaffold
    # multi-set of other targets it appears in.
    cross_lig_counts = (bb.filter(pl.col("inchikey").is_not_null())
                        .group_by(["inchikey"]).agg(
                            pl.col("target").n_unique().alias("n_targets_lig")))
    cross_lig_map = dict(zip(cross_lig_counts["inchikey"].to_list(),
                             cross_lig_counts["n_targets_lig"].to_list()))
    cross_scf_counts = (bb.filter(pl.col("scaffold_smiles").is_not_null())
                        .group_by(["scaffold_smiles"]).agg(
                            pl.col("target").n_unique().alias("n_targets_scf")))
    cross_scf_map = dict(zip(cross_scf_counts["scaffold_smiles"].to_list(),
                             cross_scf_counts["n_targets_scf"].to_list()))

    rows = []
    pairs = bb.select("target", "split").unique().sort(["split", "target"])
    for tgt, sp in pairs.iter_rows():
        ev = bb.filter((pl.col("target") == tgt) & (pl.col("split") == sp))
        if ev.is_empty():
            continue
        y = ev["label"].to_numpy().astype(np.int8)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            continue
        inchs = ev["inchikey"].to_list()
        scfs = ev["scaffold_smiles"].to_list()

        s_ch = np.array([1 if ik in chembl_set else 0 for ik in inchs], dtype=np.int8)
        s_bb = np.array([1 if ik in bdb_set else 0 for ik in inchs], dtype=np.int8)
        s_pb = np.array([1 if ik in pdb_set else 0 for ik in inchs], dtype=np.int8)
        # Cross-target: number of OTHER BayesBind targets sharing this ligand /
        # scaffold (counts > 1 -> appears in another target).
        s_xlig = np.array([(cross_lig_map.get(ik, 0) - 1) for ik in inchs], dtype=np.int16)
        s_xscf = np.array([(cross_scf_map.get(sc, 0) - 1) for sc in scfs], dtype=np.int16)
        s_kg = ((s_ch | s_bb | s_pb | (s_xlig > 0).astype(np.int8) | (s_xscf > 0).astype(np.int8))).astype(np.int8)

        for diag, score in (("chembl_overlap", s_ch),
                            ("bindingdb_overlap", s_bb),
                            ("pdbbind_overlap", s_pb),
                            ("cross_target_ligand", s_xlig),
                            ("cross_target_scaffold", s_xscf),
                            ("kg_nn_any", s_kg)):
            rows.append({
                "target": tgt,
                "split_setting": f"within_{sp}_actives_vs_decoys",
                "diagnostic": diag,
                "auroc": safe_auroc(y, score),
                "ap": safe_ap(y, score),
                "ef1pct": _enrich_factor(y, score, 0.01),
                "n_eval": int(len(y)),
                "n_positives": n_pos,
                "n_hits_in_score_gt0": int((score > 0).sum()),
            })
    return pl.DataFrame(rows)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = run()
    df.write_csv(OUT_CSV)
    print(f"[bayesbind] wrote {OUT_CSV} rows={df.height}")
    # Aggregate per diagnostic
    agg = df.group_by("diagnostic").agg([
        pl.col("auroc").mean().alias("mean_auroc"),
        pl.col("auroc").median().alias("median_auroc"),
        pl.col("auroc").min().alias("min_auroc"),
        pl.col("auroc").max().alias("max_auroc"),
        pl.len().alias("n_target_splits"),
    ]).sort("mean_auroc", descending=True)
    for r in agg.iter_rows(named=True):
        print(f"   {r['diagnostic']:>18}  mean={r['mean_auroc']:.3f}  median={r['median_auroc']:.3f}  min={r['min_auroc']:.3f}  max={r['max_auroc']:.3f}  n={r['n_target_splits']}")


if __name__ == "__main__":
    main()
