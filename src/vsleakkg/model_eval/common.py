"""Common helpers for smoke-subset preparation."""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "model_eval_inputs"
OUT.mkdir(parents=True, exist_ok=True)


def stratified_subset(df: pl.DataFrame, label_col: str, n: int = 100,
                      seed: int = 17) -> pl.DataFrame:
    """Return at most n rows, with ≥10 positives and ≥10 negatives when
    available."""
    if df.is_empty():
        return df
    pos = df.filter(pl.col(label_col) == 1)
    neg = df.filter(pl.col(label_col) == 0)
    npos = min(pos.height, max(10, n // 4))
    nneg = n - npos
    pos = pos.sample(n=npos, seed=seed) if pos.height > npos else pos
    neg = neg.sample(n=nneg, seed=seed) if neg.height > nneg else neg
    out = pl.concat([pos, neg], how="vertical_relaxed")
    return out.sample(n=min(n, out.height), seed=seed)


def write_smiles_csv(df: pl.DataFrame, out_path: Path,
                     smiles_col: str = "smiles_canonical",
                     extra_cols: tuple[str, ...] = ("target", "label", "inchikey")) -> None:
    keep = [c for c in [smiles_col] + list(extra_cols) if c in df.columns]
    df.select(keep).write_csv(out_path)
