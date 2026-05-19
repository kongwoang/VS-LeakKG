"""MVP contamination score per evaluation example.

For each test/query example, computes a heuristic score in [0, 1] combining:
  C_identity : 1 if same canonical SMILES (or InChIKey) appears among training
               actives, else 0.
  C_scaffold : 1 if same Bemis-Murcko scaffold appears among training actives,
               else 0.
  C_analog   : max ECFP4 Tanimoto to training actives.
  C_source   : 1 if dataset source is shared with the training pool, else 0.
  C_assay    : 1 if assay id is shared with training, else None (NA) if assay
               metadata is missing.

C_total = weighted mean over available (non-None) components. The current
weighting is uniform — this is deliberately the simplest defensible aggregator.
Better aggregators (paths in the KG, per-component calibration) are deferred.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import polars as pl

from . import chem as vc


def compute_scores(
    df: pl.DataFrame, *, train_mask, eval_mask,
    fp_col: str = "_fp", assay_col: Optional[str] = None,
) -> pl.DataFrame:
    train_actives = df.filter(train_mask & (pl.col("label") == 1))
    eval_ = df.filter(eval_mask)
    if eval_.is_empty():
        return pl.DataFrame()

    train_smiles = set(train_actives.filter(pl.col("smiles_canonical").is_not_null())
                       ["smiles_canonical"].to_list())
    train_scaffolds = set(train_actives.filter(pl.col("scaffold_smiles").is_not_null())
                          ["scaffold_smiles"].to_list())
    train_sources = set(train_actives["source"].drop_nulls().to_list())
    train_assays = None
    if assay_col and assay_col in train_actives.columns:
        train_assays = set(train_actives[assay_col].drop_nulls().to_list())

    fps_train = train_actives.filter(pl.col(fp_col).is_not_null())[fp_col].to_list()

    smi_eval = eval_["smiles_canonical"].to_list()
    scaf_eval = eval_["scaffold_smiles"].to_list()
    src_eval = eval_["source"].to_list()
    fps_eval = eval_[fp_col].to_list()
    assay_eval = eval_[assay_col].to_list() if (assay_col and assay_col in eval_.columns) else [None] * eval_.height

    c_identity = np.fromiter((1.0 if s in train_smiles else 0.0 for s in smi_eval), float, eval_.height)
    c_scaffold = np.fromiter((1.0 if s in train_scaffolds else 0.0 for s in scaf_eval), float, eval_.height)
    c_source = np.fromiter((1.0 if s in train_sources else 0.0 for s in src_eval), float, eval_.height)
    c_assay = np.full(eval_.height, np.nan, dtype=float)
    if train_assays is not None:
        for i, a in enumerate(assay_eval):
            c_assay[i] = 1.0 if a in train_assays else 0.0

    if fps_train:
        c_analog = vc.max_tanimoto_to_set(fps_eval, fps_train)
        c_analog = np.where(c_analog < 0, np.nan, c_analog)
    else:
        c_analog = np.full(eval_.height, np.nan, dtype=float)

    components = np.vstack([c_identity, c_scaffold, c_analog, c_source, c_assay])
    with np.errstate(invalid="ignore"):
        c_total = np.nanmean(components, axis=0)

    out = eval_.with_columns([
        pl.Series("c_identity", c_identity),
        pl.Series("c_scaffold", c_scaffold),
        pl.Series("c_analog", c_analog),
        pl.Series("c_source", c_source),
        pl.Series("c_assay", c_assay),
        pl.Series("c_total", c_total),
    ])
    return out
