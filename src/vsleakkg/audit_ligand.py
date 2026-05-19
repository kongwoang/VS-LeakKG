"""Ligand-level leakage audit.

All functions accept an examples frame that already has canonical SMILES,
InChIKey, scaffold_smiles, and optionally ECFP4 bit vectors. They emit small
summary DataFrames intended to be saved as CSV under outputs/tables/.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from . import chem as vc


def _bag(df: pl.DataFrame, group_col: str, value_col: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for grp, val in df.select([group_col, value_col]).drop_nulls().iter_rows():
        out.setdefault(grp, set()).add(val)
    return out


def identity_overlap_table(df: pl.DataFrame, group_col: str,
                           key_col: str = "smiles_canonical") -> pl.DataFrame:
    """Pairwise set overlap of identifying keys across distinct values of `group_col`.

    `key_col` is typically `smiles_canonical` or `inchikey`. Rows: (group_a,
    group_b, key, n_keys_a, n_keys_b, n_shared, jaccard)."""
    bags = _bag(df, group_col, key_col)
    rows = []
    keys = sorted(bags.keys())
    for a, b in combinations(keys, 2):
        sa, sb = bags[a], bags[b]
        inter = sa & sb
        union = sa | sb
        jacc = len(inter) / len(union) if union else 0.0
        rows.append((a, b, key_col, len(sa), len(sb), len(inter), float(jacc)))
    return pl.DataFrame(
        rows, schema=["group_a", "group_b", "key", "n_keys_a", "n_keys_b",
                      "n_shared", "jaccard"], orient="row"
    )


def intra_group_duplicates(df: pl.DataFrame, group_col: str,
                           key_col: str = "smiles_canonical") -> pl.DataFrame:
    """For each group, count duplicate identifying keys (n_rows - n_unique)."""
    rows = []
    for grp, sub in df.filter(pl.col(key_col).is_not_null()).group_by(group_col):
        # polars group_by returns (key, frame) tuples; key is wrapped in a tuple
        grp_val = grp[0] if isinstance(grp, tuple) else grp
        n_rows = sub.height
        n_unique = sub.select(pl.col(key_col)).unique().height
        rows.append((grp_val, key_col, n_rows, n_unique, n_rows - n_unique))
    return pl.DataFrame(
        rows, schema=["group", "key", "n_rows", "n_unique", "n_duplicate_rows"],
        orient="row",
    )


def scaffold_overlap_table(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    return identity_overlap_table(df, group_col=group_col, key_col="scaffold_smiles")


def analog_overlap_pairs(
    fps_a: Sequence, fps_b: Sequence, thresholds: Iterable[float],
) -> Dict[float, int]:
    """Count (a, b) Tanimoto pairs above each threshold. Skips None fingerprints."""
    return vc.count_pairs_above(
        [fp for fp in fps_a if fp is not None],
        [fp for fp in fps_b if fp is not None],
        thresholds,
    )


def analog_overlap_table(
    df: pl.DataFrame, group_col: str, fp_col: str = "_fp",
    thresholds: Iterable[float] = (0.6, 0.8, 0.9),
) -> pl.DataFrame:
    """Cross-group analog overlap. For each (group_a, group_b) and each
    threshold, count Tanimoto pairs >= threshold. Self-pairs (a == b) report
    intra-group pairs above threshold, excluding the identity diagonal."""
    groups = sorted(df.select(group_col).drop_nulls().unique().to_series().to_list())
    rows = []
    grouped = {g: df.filter(pl.col(group_col) == g) for g in groups}
    for i, a in enumerate(groups):
        for j in range(i, len(groups)):
            b = groups[j]
            fps_a = grouped[a][fp_col].to_list()
            fps_b = grouped[b][fp_col].to_list()
            if a == b:
                fps_b_no_self = fps_b[:]
                # For intra-group, count pairs (i, j) with i<j over the same list.
                counts = {t: 0 for t in thresholds}
                fps_clean = [fp for fp in fps_a if fp is not None]
                for ii in range(len(fps_clean)):
                    if ii + 1 >= len(fps_clean):
                        continue
                    refs = fps_clean[ii + 1:]
                    sims = np.asarray(vc.bulk_tanimoto(fps_clean[ii], refs))
                    for t in thresholds:
                        counts[t] += int((sims >= t).sum())
            else:
                counts = analog_overlap_pairs(fps_a, fps_b, thresholds)
            for t in thresholds:
                rows.append((a, b, float(t), counts[t]))
    return pl.DataFrame(
        rows, schema=["group_a", "group_b", "threshold", "n_pairs_above"],
        orient="row",
    )
