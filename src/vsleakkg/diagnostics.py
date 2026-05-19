"""Shortcut-bias diagnostics: non-learning baselines that should *not* work if
the benchmark is leak-free."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from . import chem as vc


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> Optional[float]:
    """EF at top fraction. Returns None if either class is empty."""
    y_true = np.asarray(y_true).astype(int)
    n = y_true.size
    if n == 0:
        return None
    n_actives = int(y_true.sum())
    if n_actives == 0 or n_actives == n:
        return None
    k = max(1, int(round(n * top_frac)))
    order = np.argsort(-np.asarray(y_score))
    top_actives = int(y_true[order[:k]].sum())
    return float((top_actives / k) / (n_actives / n))


def ligand_identity_memorization(df: pl.DataFrame, *, train_mask, eval_mask,
                                 key_col: str = "smiles_canonical") -> Dict[str, Optional[float]]:
    """Predict label 1 if a query's key is also seen as label==1 in the train
    set, else 0. This is the strictest "did you memorize identity" check."""
    train = df.filter(train_mask & pl.col(key_col).is_not_null())
    eval_ = df.filter(eval_mask & pl.col(key_col).is_not_null())
    if train.is_empty() or eval_.is_empty():
        return {"auroc": None, "ap": None, "ef1pct": None, "n_eval": eval_.height, "n_train": train.height}
    train_actives = set(train.filter(pl.col("label") == 1)[key_col].to_list())
    y_true = eval_["label"].to_numpy().astype(int)
    y_score = np.fromiter((1.0 if k in train_actives else 0.0 for k in eval_[key_col].to_list()),
                          dtype=np.float32, count=eval_.height)
    return {
        "auroc": _safe_auroc(y_true, y_score),
        "ap": _safe_ap(y_true, y_score),
        "ef1pct": enrichment_factor(y_true, y_score, 0.01),
        "n_eval": int(eval_.height),
        "n_train": int(train.height),
    }


def scaffold_memorization(df: pl.DataFrame, *, train_mask, eval_mask) -> Dict[str, Optional[float]]:
    return ligand_identity_memorization(df, train_mask=train_mask, eval_mask=eval_mask,
                                        key_col="scaffold_smiles")


def ligand_knn_max_tanimoto(df: pl.DataFrame, *, train_mask, eval_mask,
                            fp_col: str = "_fp") -> Dict[str, Optional[float]]:
    """Score each eval ligand by max Tanimoto to any train **active** ligand.
    Larger = more similar to a training active. Reports AUROC / AP / EF1%."""
    train_actives = df.filter(train_mask & (pl.col("label") == 1) & pl.col(fp_col).is_not_null())
    eval_ = df.filter(eval_mask & pl.col(fp_col).is_not_null())
    if train_actives.is_empty() or eval_.is_empty():
        return {"auroc": None, "ap": None, "ef1pct": None, "n_eval": eval_.height,
                "n_train_actives": train_actives.height}
    refs = train_actives[fp_col].to_list()
    qfps = eval_[fp_col].to_list()
    y_true = eval_["label"].to_numpy().astype(int)
    y_score = vc.max_tanimoto_to_set(qfps, refs)
    return {
        "auroc": _safe_auroc(y_true, y_score),
        "ap": _safe_ap(y_true, y_score),
        "ef1pct": enrichment_factor(y_true, y_score, 0.01),
        "n_eval": int(eval_.height),
        "n_train_actives": int(train_actives.height),
    }
