"""Shared virtual-screening metrics: AUROC, AP, EF@k, BEDROC.

All functions are NaN-safe for one-class inputs and tolerate score ties.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _to_arr(y) -> np.ndarray:
    return np.asarray(y).ravel()


def safe_auroc(y, s) -> float:
    y = _to_arr(y); s = _to_arr(s)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def safe_ap(y, s) -> float:
    y = _to_arr(y); s = _to_arr(s)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def enrichment_factor(y, s, frac: float = 0.01) -> float:
    """EF@frac = (hits_in_top_frac / k) / (pos_rate)."""
    y = _to_arr(y).astype(np.int8); s = _to_arr(s).astype(np.float64)
    n = y.size
    pos = int(y.sum())
    if n == 0 or pos == 0 or pos == n:
        return float("nan")
    k = max(1, int(round(n * frac)))
    # Stable sort on (-score, random tie-break) gives deterministic result.
    order = np.argsort(-s, kind="stable")[:k]
    hits = int(y[order].sum())
    return float((hits / k) / (pos / n))


def bedroc(y, s, alpha: float = 20.0) -> float:
    """Truchon & Bayly (2007) BEDROC. Robust to ties via stable sort."""
    y = _to_arr(y).astype(np.int8); s = _to_arr(s).astype(np.float64)
    N = y.size
    n = int(y.sum())
    if N == 0 or n == 0 or n == N:
        return float("nan")
    # Ranks of positives, 1-indexed, ordered by decreasing score
    order = np.argsort(-s, kind="stable")
    ranks = np.where(y[order] == 1)[0] + 1  # 1..N
    Ra = n / N
    Ri = 1.0 - Ra
    sum_exp = np.exp(-alpha * ranks / N).sum()
    factor_a = sum_exp / (Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1))
    factor_b = (Ra * np.sinh(alpha / 2.0)) / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * Ra))
    return float(factor_a * factor_b + 1.0 / (1.0 - np.exp(alpha * Ri)))


def all_metrics(y, s, fracs: Iterable[float] = (0.005, 0.01, 0.05), bedroc_alpha: float = 20.0) -> dict:
    y = _to_arr(y); s = _to_arr(s)
    out = {
        "n_eval": int(y.size),
        "n_positives": int(np.sum(y == 1)),
        "auroc": safe_auroc(y, s),
        "ap": safe_ap(y, s),
        "bedroc": bedroc(y, s, alpha=bedroc_alpha),
    }
    for f in fracs:
        out[f"ef{f * 100:g}pct"] = enrichment_factor(y, s, f)
    return out


def aggregate(rows: list[dict], by: str = "diagnostic") -> dict[str, dict[str, float]]:
    """Aggregate per-target rows of `all_metrics` output by a grouping key."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[by]].append(r)
    out = {}
    for k, lst in buckets.items():
        out[k] = {}
        for m in ("auroc", "ap", "bedroc", "ef0.5pct", "ef1pct", "ef5pct"):
            vals = [x[m] for x in lst if not (x.get(m) is None or (isinstance(x.get(m), float) and np.isnan(x[m])))]
            if vals:
                arr = np.asarray(vals)
                out[k][f"mean_{m}"] = float(arr.mean())
                out[k][f"median_{m}"] = float(np.median(arr))
                out[k][f"min_{m}"] = float(arr.min())
                out[k][f"max_{m}"] = float(arr.max())
            else:
                out[k][f"mean_{m}"] = float("nan")
                out[k][f"median_{m}"] = float("nan")
                out[k][f"min_{m}"] = float("nan")
                out[k][f"max_{m}"] = float("nan")
        out[k]["n_groups"] = len(lst)
    return out
