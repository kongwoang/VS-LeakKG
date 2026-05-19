"""Source-only / provenance-only diagnostic baselines.

For each diagnostic, the score is a binary or scalar feature **independent of
the val ligand's structure beyond its InChIKey membership in an external
database**. We measure whether actives are systematically more (or less)
likely to carry the feature than decoys, target by target.

Diagnostics:
  1. dataset_source_only — constant per dataset; AUROC undefined within a
     single dataset. We report it as "not meaningful" within LIT-PCBA AVE.
  2. chembl_overlap_only — val inchikey is in ChEMBL.
  3. bindingdb_overlap_only — val inchikey is in BindingDB.
  4. pdbbind_overlap_only — val inchikey is in PDBBind ligands.
  5. assay_only_confirmed — number of distinct confirmed-target assay rows
     this val ligand belongs to.
  6. assay_only_candidate — number of distinct candidate-only assay rows.
  7. document_only_confirmed — distinct confirmed-target documents.
  8. document_only_candidate — distinct candidate-only documents.
  9. decoy_protocol_only — constant within LIT-PCBA AVE; reported as
     "constant within dataset, see between-dataset comparison".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from vsleakkg.metrics import all_metrics

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_CSV = ROOT / "outputs" / "tables" / "final" / "table9_source_provenance_only_diagnostics.csv"


def main():
    ex = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet").select(
        "target", "split", "label", "inchikey"
    )
    val = ex.filter(pl.col("split") == "validation")
    print(f"[so] LIT-PCBA AVE val rows: {val.height}")

    # Source sets (already exact-InChIKey-mapped)
    chembl_set = set(pl.read_parquet(PROCESSED / "chembl_ligands.parquet")["standard_inchi_key"]
                     .drop_nulls().to_list())
    bdb_set = set(pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")["ligand_inchikey"]
                  .drop_nulls().to_list())
    pdb_set = set(pl.read_parquet(PROCESSED / "pdbbind_ligands.parquet")["inchikey"]
                  .drop_nulls().to_list())

    # ChEMBL provenance counts per inchikey
    cand = (pl.read_parquet(PROCESSED / "benchmark_chembl_candidate_provenance.parquet")
            .filter(pl.col("benchmark_dataset") == "LIT-PCBA AVE")
            .group_by("inchikey")
            .agg([pl.col("assay_chembl_id").n_unique().alias("n_assay_cand"),
                  pl.col("document_chembl_id").n_unique().alias("n_doc_cand")]))
    conf = (pl.read_parquet(PROCESSED / "benchmark_chembl_confirmed_provenance.parquet")
            .filter(pl.col("benchmark_dataset") == "LIT-PCBA AVE")
            .group_by("inchikey")
            .agg([pl.col("assay_chembl_id").n_unique().alias("n_assay_conf"),
                  pl.col("document_chembl_id").n_unique().alias("n_doc_conf")]))

    val = (val.join(cand, on="inchikey", how="left")
              .join(conf, on="inchikey", how="left")
              .with_columns([pl.col("n_assay_cand").fill_null(0),
                             pl.col("n_doc_cand").fill_null(0),
                             pl.col("n_assay_conf").fill_null(0),
                             pl.col("n_doc_conf").fill_null(0)]))

    # Score columns
    inch = val["inchikey"].to_list()
    s_chembl = np.array([1 if k in chembl_set else 0 for k in inch], dtype=np.int8)
    s_bdb    = np.array([1 if k in bdb_set    else 0 for k in inch], dtype=np.int8)
    s_pdb    = np.array([1 if k in pdb_set    else 0 for k in inch], dtype=np.int8)
    val = val.with_columns([
        pl.Series("s_chembl", s_chembl),
        pl.Series("s_bdb", s_bdb),
        pl.Series("s_pdb", s_pdb),
    ])

    diagnostics = [
        ("chembl_overlap_only",     "s_chembl"),
        ("bindingdb_overlap_only",  "s_bdb"),
        ("pdbbind_overlap_only",    "s_pdb"),
        ("assay_only_confirmed",    "n_assay_conf"),
        ("assay_only_candidate",    "n_assay_cand"),
        ("document_only_confirmed", "n_doc_conf"),
        ("document_only_candidate", "n_doc_cand"),
    ]

    rows = []
    for diag_name, col in diagnostics:
        for tgt in sorted(val["target"].unique().to_list()):
            sub = val.filter(pl.col("target") == tgt)
            y = sub["label"].to_numpy()
            s = sub[col].to_numpy().astype(float)
            m = all_metrics(y, s)
            rows.append({"dataset": "LIT-PCBA AVE", "target": tgt,
                         "diagnostic": diag_name, **m})
        # ALL rollup
        m = all_metrics(val["label"].to_numpy(), val[col].to_numpy().astype(float))
        rows.append({"dataset": "LIT-PCBA AVE", "target": "ALL",
                     "diagnostic": diag_name, **m})

    # Dataset-source / decoy-protocol within a single dataset are constants
    # → record as not meaningful rows for documentation.
    for name in ("dataset_source_only", "decoy_protocol_only"):
        rows.append({"dataset": "LIT-PCBA AVE", "target": "ALL",
                     "diagnostic": name, "n_eval": val.height,
                     "n_positives": int((val["label"] == 1).sum()),
                     "auroc": float("nan"), "ap": float("nan"),
                     "bedroc": float("nan"),
                     "ef0.5pct": float("nan"), "ef1pct": float("nan"),
                     "ef5pct": float("nan")})

    # Optional: BayesBind same-style diagnostics if labels exist
    try:
        bb = pl.read_parquet(PROCESSED / "bayesbind_examples.parquet")
        # Test split only (no train) — within-split active-vs-decoy.
        for spl in ("test", "val"):
            sub_all = bb.filter(pl.col("split") == spl)
            if sub_all.is_empty():
                continue
            inch = sub_all["inchikey"].to_list()
            s_ch = np.array([1 if k in chembl_set else 0 for k in inch], dtype=np.int8)
            s_bd = np.array([1 if k in bdb_set    else 0 for k in inch], dtype=np.int8)
            s_pd = np.array([1 if k in pdb_set    else 0 for k in inch], dtype=np.int8)
            for (diag_name, vec) in (("chembl_overlap_only", s_ch),
                                     ("bindingdb_overlap_only", s_bd),
                                     ("pdbbind_overlap_only", s_pd)):
                # Per target
                for tgt in sorted(sub_all["target"].unique().to_list()):
                    mask = (sub_all["target"] == tgt).to_numpy()
                    y = sub_all["label"].to_numpy()[mask]
                    s = vec[mask].astype(float)
                    m = all_metrics(y, s)
                    rows.append({"dataset": "BayesBind V1.5", "target": tgt,
                                 "diagnostic": f"{diag_name}_{spl}", **m})
                # ALL within split
                m = all_metrics(sub_all["label"].to_numpy(), vec.astype(float))
                rows.append({"dataset": "BayesBind V1.5", "target": f"ALL_{spl}",
                             "diagnostic": f"{diag_name}_{spl}", **m})
    except Exception as e:
        print(f"[so] BayesBind diagnostics skipped: {e}")

    out = pl.DataFrame(rows)
    out.write_csv(OUT_CSV)
    print(f"[so] wrote {OUT_CSV} ({out.height} rows)")

    # Print aggregate per (dataset, diagnostic)
    agg = (out.filter(pl.col("target").str.starts_with("ALL").not_())
              .group_by(["dataset", "diagnostic"])
              .agg([
                  pl.col("auroc").mean().alias("mean_auroc"),
                  pl.col("auroc").median().alias("median_auroc"),
                  pl.col("ap").mean().alias("mean_ap"),
                  pl.col("bedroc").mean().alias("mean_bedroc"),
                  pl.col("ef1pct").mean().alias("mean_ef1pct"),
                  pl.len().alias("n_targets"),
              ])
              .sort(["dataset", "mean_auroc"], descending=[False, True]))
    print(agg.to_pandas().to_string())


if __name__ == "__main__":
    main()
