"""MVP-1 audit orchestrator for VS-LeakKG.

Upgrades MVP-0 with:
  * Real LIT-PCBA AVE_unbiased train/validation splits (no more synthetic 80/20).
  * DEKOIS 2.0 as a second decoy protocol next to DUD-E.

Re-runs the per-dataset leakage audit, shortcut diagnostics, and contamination
score using the held-out validation split as the eval pool, plus produces a
DUD-E vs DEKOIS decoy-protocol comparison.

Usage:
    python -m vsleakkg.run_mvp1_audit
    python -m vsleakkg.run_mvp1_audit --inactive-cap 5000 --val-cap 2000 \
        --skip-dekois
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

# Make the package importable when run as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vsleakkg import chem as vc
from vsleakkg import build_graph as vb
from vsleakkg import load_litpcba_ave, load_dekois
from vsleakkg import audit_ligand as audit

PROJECT_ROOT = Path("D:/hoangpc/VS-LeakKG")
RAW          = PROJECT_ROOT / "data" / "raw"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
TABLES       = PROJECT_ROOT / "outputs" / "tables"
REPORTS      = PROJECT_ROOT / "outputs" / "reports"
LOGS         = PROJECT_ROOT / "outputs" / "logs"
DISK_LOG     = LOGS / "mvp1_audit_disk_usage.log"
RUN_LOG      = LOGS / "mvp1_audit.log"

for d in (PROCESSED, TABLES, REPORTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vsleakkg.mvp1")


# -------------------- disk + helpers --------------------

def _du_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def log_step(event: str, target: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"==== {ts} ====", f"event: {event}", f"target: {target}",
             f"cwd: {os.getcwd()}"]
    for part in ("C:/", "D:/"):
        try:
            u = shutil.disk_usage(part)
            lines.append(f"  drive {part}: used={u.used/1024**3:.2f}GB "
                         f"free={u.free/1024**3:.2f}GB")
        except OSError:
            pass
    proj_mb = _du_bytes(PROJECT_ROOT) / 1024**2
    lines.append(f"-- project size: {proj_mb:.2f} MB ({PROJECT_ROOT})")
    lines.append("")
    DISK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def check_disk(min_free_gb: float, label: str) -> bool:
    try:
        u = shutil.disk_usage(PROJECT_ROOT)
    except OSError:
        return True
    free_gb = u.free / 1024**3
    if free_gb < min_free_gb:
        log.warning("check_disk FAIL free=%.2fGB need>=%.2fGB label=%s",
                    free_gb, min_free_gb, label)
        return False
    return True


# -------------------- multiprocessing featurization --------------------

def _featurize_chunk(smiles_list: List[str]) -> List[tuple]:
    out = []
    for smi in smiles_list:
        f = vc.featurize(smi)
        out.append((f.smiles_canonical, f.inchikey, f.scaffold_smiles, f.parse_ok))
    return out


def _ecfp_bytes_chunk(smiles_list: List[Optional[str]]) -> List[Optional[bytes]]:
    return [vc.ecfp_bytes(s) if s else None for s in smiles_list]


def parallel_map(fn, items: List, workers: int, chunksize: int, label: str) -> List:
    if not items:
        return []
    chunks = [items[i:i+chunksize] for i in range(0, len(items), chunksize)]
    out: List = []
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for j, batch in enumerate(pool.imap(fn, chunks, chunksize=1), 1):
            out.extend(batch)
            if j % 50 == 0 or j == len(chunks):
                log.info("[%s] %d / %d chunks (%d / %d rows)",
                         label, j, len(chunks), len(out), len(items))
    return out


def parallel_featurize(smiles: List[str], workers: int, chunksize: int = 2000) -> pl.DataFrame:
    rows = parallel_map(_featurize_chunk, smiles, workers, chunksize, "featurize")
    return pl.DataFrame({
        "smiles_canonical": [r[0] for r in rows],
        "inchikey":         [r[1] for r in rows],
        "scaffold_smiles":  [r[2] for r in rows],
        "parse_ok":         [r[3] for r in rows],
    })


def parallel_ecfp_bytes(smiles: List[Optional[str]], workers: int, chunksize: int = 4000) -> List[Optional[bytes]]:
    return parallel_map(_ecfp_bytes_chunk, smiles, workers, chunksize, "ecfp")


def to_fp_objects(fp_bytes: Iterable[Optional[bytes]]):
    return [vc.bytes_to_fp(b) if b else None for b in fp_bytes]


# -------------------- common metrics --------------------

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


def _ef(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> Optional[float]:
    y = np.asarray(y_true).astype(int)
    n = y.size
    if n == 0:
        return None
    pos = int(y.sum())
    if pos == 0 or pos == n:
        return None
    k = max(1, int(round(n * top_frac)))
    order = np.argsort(-np.asarray(y_score))
    return float((y[order[:k]].sum() / k) / (pos / n))


# -------------------- run config --------------------

@dataclasses.dataclass
class RunConfig:
    workers: int = 12
    inactive_cap_per_target: int = 5000     # train inactives sampled per LIT-PCBA target
    val_cap_per_target: int = 2000          # val ligands sampled per LIT-PCBA target for analog audit
    similarity_threshold: float = 0.8       # for ligand_similar_to_ligand edges
    similarity_top_k_per_val: int = 5       # cap edges per val ligand
    analog_thresholds: Tuple[float, ...] = (0.6, 0.8, 0.9)
    skip_litpcba_ave: bool = False
    skip_dekois: bool = False
    rng_seed: int = 17


# -------------------- task 2: load + featurize LIT-PCBA AVE --------------------

def task_ave_load(cfg: RunConfig) -> pl.DataFrame:
    log_step("pre_step", "ave_load_and_featurize")
    cached = PROCESSED / "litpcba_ave_examples.parquet"
    if cached.exists():
        log.info("AVE: loading cached %s", cached)
        df = pl.read_parquet(cached)
        log_step("post_step", f"ave_load_and_featurize cached={df.height}")
        return df

    df = load_litpcba_ave.load_all(RAW / "LIT-PCBA" / "splits" / "AVE_unbiased")
    log.info("AVE: loaded %d examples across %d targets",
             df.height, df.select("target").n_unique())

    t0 = time.time()
    feats = parallel_featurize(df["smiles_input"].to_list(), workers=cfg.workers)
    df = df.with_columns([
        feats["smiles_canonical"], feats["inchikey"],
        feats["scaffold_smiles"], feats["parse_ok"],
    ])
    log.info("AVE: featurize done in %.1fs; parse_ok=%d/%d",
             time.time()-t0, int(df["parse_ok"].sum()), df.height)

    fp_bytes = parallel_ecfp_bytes(df["smiles_canonical"].to_list(), workers=cfg.workers)
    df = df.with_columns(pl.Series("fp_bytes", fp_bytes, dtype=pl.Binary))

    df.write_parquet(cached)
    log.info("AVE: wrote %s", cached)

    # Layout report.
    counts = (df.group_by(["target", "split", "label_type"])
                .agg(pl.len().alias("n"))
                .sort(["target", "split", "label_type"]))
    counts.write_csv(TABLES / "litpcba_ave_layout_counts.csv")
    (REPORTS / "litpcba_ave_layout_report.md").write_text(_render_ave_layout(df, counts), encoding="utf-8")

    log_step("post_step", f"ave_load_and_featurize rows={df.height}")
    return df


def _render_ave_layout(df: pl.DataFrame, counts: pl.DataFrame) -> str:
    by_split = df.group_by("split").agg(pl.len().alias("n")).sort("split")
    by_target = (df.group_by("target").agg(pl.len().alias("n")).sort("target"))
    return (
        "# LIT-PCBA AVE_unbiased — layout report\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "## Overall counts\n\n"
        f"- examples: **{df.height}**\n"
        f"- targets:  **{df.select('target').n_unique()}**\n"
        f"- splits:   **{df.select('split').n_unique()}** "
        f"({', '.join(df.select('split').drop_nulls().unique().to_series().to_list())})\n"
        f"- parse_ok: **{int(df['parse_ok'].sum())} / {df.height}**\n\n"
        "## By split\n\n"
        + "\n".join(f"- {r['split']}: {r['n']}" for r in by_split.iter_rows(named=True)) + "\n\n"
        "## Per target × split × label_type (full table)\n\n"
        + counts.to_pandas().to_string(index=False) + "\n\n"
        "## Source\n\n"
        "Files extracted from `data/raw/LIT-PCBA/splits/AVE_unbiased.tgz` (Tran-Nguyen et al.\n"
        "2020). The `_T` / `_V` suffix encodes the AVE-debiased train / validation split.\n"
    )


# -------------------- task 3: leakage audit on real split --------------------

def task_ave_leakage(df: pl.DataFrame, cfg: RunConfig) -> None:
    log_step("pre_step", "ave_leakage_audit")
    rng = np.random.default_rng(cfg.rng_seed)
    targets = sorted(df.select("target").drop_nulls().unique().to_series().to_list())

    id_rows = []
    scaf_rows = []
    analog_rows = []

    for target in targets:
        sub = df.filter(pl.col("target") == target)
        tr  = sub.filter(pl.col("split") == "train")
        val = sub.filter(pl.col("split") == "validation")
        if tr.is_empty() or val.is_empty():
            continue

        # Identity (canonical SMILES + InChIKey)
        for key in ("smiles_canonical", "inchikey"):
            tr_keys  = set(tr.filter(pl.col(key).is_not_null())[key].to_list())
            val_keys = set(val.filter(pl.col(key).is_not_null())[key].to_list())
            shared = tr_keys & val_keys
            union  = tr_keys | val_keys
            jacc = len(shared) / len(union) if union else 0.0
            id_rows.append((target, key, len(tr_keys), len(val_keys),
                            len(shared), float(jacc),
                            int((val[key].is_in(list(shared))).sum()) if shared else 0))

        # Scaffold
        tr_scaf  = set(tr.filter((pl.col("scaffold_smiles").is_not_null()) &
                                  (pl.col("scaffold_smiles") != ""))["scaffold_smiles"].to_list())
        val_scaf = set(val.filter((pl.col("scaffold_smiles").is_not_null()) &
                                   (pl.col("scaffold_smiles") != ""))["scaffold_smiles"].to_list())
        shared = tr_scaf & val_scaf
        union  = tr_scaf | val_scaf
        scaf_rows.append((target, len(tr_scaf), len(val_scaf), len(shared),
                          (len(shared) / len(union)) if union else 0.0,
                          int(val.filter(pl.col("scaffold_smiles").is_in(list(shared))).height)
                          if shared else 0))

        # Analog: cap train pool + val pool, compute max Tanimoto val->train.
        tr_actives = tr.filter(pl.col("label") == 1)
        tr_inactives = tr.filter(pl.col("label") == 0)
        if tr_inactives.height > cfg.inactive_cap_per_target:
            idx = rng.choice(tr_inactives.height, size=cfg.inactive_cap_per_target, replace=False)
            tr_inactives = tr_inactives[sorted(idx.tolist())]
        tr_pool = pl.concat([tr_actives, tr_inactives], how="vertical_relaxed")
        # Stratified val sampling: keep all val actives, sample only inactives.
        val_a = val.filter(pl.col("label") == 1)
        val_i = val.filter(pl.col("label") == 0)
        if val_a.height + val_i.height > cfg.val_cap_per_target:
            keep_i = max(0, cfg.val_cap_per_target - val_a.height)
            if val_i.height > keep_i:
                idx = rng.choice(val_i.height, size=keep_i, replace=False)
                val_i = val_i[sorted(idx.tolist())]
        val_pool = pl.concat([val_a, val_i], how="vertical_relaxed")
        tr_fps  = to_fp_objects(tr_pool.filter(pl.col("fp_bytes").is_not_null())["fp_bytes"].to_list())
        val_fps = to_fp_objects(val_pool.filter(pl.col("fp_bytes").is_not_null())["fp_bytes"].to_list())
        max_t_per_val = vc.max_tanimoto_to_set(val_fps, tr_fps) if (tr_fps and val_fps) else np.array([])

        for thr in cfg.analog_thresholds:
            n_above = int(np.sum(max_t_per_val >= thr)) if max_t_per_val.size else 0
            analog_rows.append((target, float(thr), n_above,
                                int(val_pool.height), int(tr_pool.height)))
        log.info("AVE leakage: %s  train_pool=%d  val_pool=%d  max_T_mean=%.3f",
                 target, tr_pool.height, val_pool.height,
                 float(np.mean(max_t_per_val)) if max_t_per_val.size else float("nan"))

    pl.DataFrame(id_rows, schema=[
        "target", "key", "n_keys_train", "n_keys_val",
        "n_shared", "jaccard", "n_val_rows_in_shared",
    ], orient="row").write_csv(TABLES / "litpcba_ave_identity_leakage.csv")

    pl.DataFrame(scaf_rows, schema=[
        "target", "n_scaffolds_train", "n_scaffolds_val",
        "n_scaffolds_shared", "jaccard", "n_val_rows_with_shared_scaffold",
    ], orient="row").write_csv(TABLES / "litpcba_ave_scaffold_overlap.csv")

    pl.DataFrame(analog_rows, schema=[
        "target", "threshold", "n_val_rows_above",
        "val_pool_size", "train_pool_size",
    ], orient="row").write_csv(TABLES / "litpcba_ave_analog_leakage.csv")

    (REPORTS / "litpcba_ave_leakage_report.md").write_text(_render_ave_leakage_report(cfg), encoding="utf-8")
    log_step("post_step", "ave_leakage_audit")


def _render_ave_leakage_report(cfg: RunConfig) -> str:
    return (
        "# LIT-PCBA AVE — train→validation leakage audit\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "## What changed vs MVP-0\n"
        "- Splits are the real AVE-debiased train (`*_T.smi`) and validation\n"
        "  (`*_V.smi`) per target. No more synthetic 80/20.\n"
        "- Identity overlap is measured both by canonical SMILES and InChIKey.\n"
        "- Analog overlap is reported as the number of validation ligands whose\n"
        "  maximum ECFP4 Tanimoto to any train ligand is ≥ {0.6, 0.8, 0.9}.\n\n"
        "## Tables\n"
        "- `outputs/tables/litpcba_ave_identity_leakage.csv` — per-target identity overlap\n"
        "- `outputs/tables/litpcba_ave_scaffold_overlap.csv` — per-target scaffold overlap\n"
        "- `outputs/tables/litpcba_ave_analog_leakage.csv` — per-target analog counts at 3 thresholds\n\n"
        "## Caps\n"
        f"- Train inactives capped per target at **{cfg.inactive_cap_per_target}** (sampled with seed {cfg.rng_seed}).\n"
        f"- Validation ligands capped per target at **{cfg.val_cap_per_target}** for the analog audit.\n"
        f"- Identity / scaffold overlaps are computed against the FULL (uncapped) train and validation sets.\n"
    )


# -------------------- task 4: shortcut diagnostics with real split --------------------

def task_ave_diagnostics(df: pl.DataFrame, cfg: RunConfig) -> None:
    log_step("pre_step", "ave_shortcut_diagnostics")
    rng = np.random.default_rng(cfg.rng_seed + 11)
    targets = sorted(df.select("target").drop_nulls().unique().to_series().to_list())
    rows = []

    for target in targets:
        sub = df.filter(pl.col("target") == target)
        tr  = sub.filter(pl.col("split") == "train")
        val = sub.filter(pl.col("split") == "validation")
        n_tr_a  = int((tr["label"] == 1).sum())
        n_tr_i  = int((tr["label"] == 0).sum())
        n_val_a = int((val["label"] == 1).sum())
        n_val_i = int((val["label"] == 0).sum())
        if n_tr_a < 1 or n_val_a < 1 or n_val_i < 1:
            rows.append((target, "skipped", "real_split", None, None, None,
                         n_tr_a, n_tr_i, n_val_a, n_val_i,
                         "missing class in train or validation"))
            continue

        # Cap val for runtime via stratified sampling: keep ALL val actives,
        # sample val inactives down to (val_cap - n_val_actives). This prevents
        # uniform sampling from accidentally dropping the (often tiny) positive
        # set on small targets.
        if val.height > cfg.val_cap_per_target:
            val_a = val.filter(pl.col("label") == 1)
            val_i = val.filter(pl.col("label") == 0)
            keep_i = max(0, cfg.val_cap_per_target - val_a.height)
            if val_i.height > keep_i:
                idx = rng.choice(val_i.height, size=keep_i, replace=False)
                val_i = val_i[sorted(idx.tolist())]
            val = pl.concat([val_a, val_i], how="vertical_relaxed")
        train_actives = tr.filter(pl.col("label") == 1)
        train_scafs   = set(train_actives.filter(pl.col("scaffold_smiles").is_not_null())["scaffold_smiles"].to_list())
        train_smiles  = set(train_actives.filter(pl.col("smiles_canonical").is_not_null())["smiles_canonical"].to_list())

        y_true = val["label"].to_numpy().astype(int)

        # Identity memorization
        s_id = np.fromiter(
            (1.0 if s in train_smiles else 0.0 for s in val["smiles_canonical"].to_list()),
            float, val.height,
        )
        # Scaffold memorization
        s_sc = np.fromiter(
            (1.0 if s in train_scafs else 0.0 for s in val["scaffold_smiles"].to_list()),
            float, val.height,
        )
        # Ligand KNN
        train_act_fps = to_fp_objects(train_actives.filter(pl.col("fp_bytes").is_not_null())["fp_bytes"].to_list())
        val_fps = to_fp_objects(val.filter(pl.col("fp_bytes").is_not_null())["fp_bytes"].to_list())
        s_knn = vc.max_tanimoto_to_set(val_fps, train_act_fps) if train_act_fps else np.full(val.height, -1.0, dtype=np.float32)
        # Replace -1 (missing fp) with 0 so AUROC still works on present rows.
        s_knn = np.where(s_knn < 0, 0.0, s_knn)

        for name, score in (("identity", s_id), ("scaffold", s_sc), ("ligand_knn", s_knn)):
            rows.append((target, name, "real_split",
                         _safe_auroc(y_true, score),
                         _safe_ap(y_true, score),
                         _ef(y_true, score, 0.01),
                         n_tr_a, n_tr_i, n_val_a, n_val_i, ""))
        log.info("AVE diagnostics: %s tr=(%d,%d) val=(%d,%d)",
                 target, n_tr_a, n_tr_i, n_val_a, n_val_i)

    pl.DataFrame(rows, schema=[
        "target", "diagnostic", "protocol",
        "auroc", "ap", "ef1pct",
        "n_train_actives", "n_train_inactives",
        "n_val_actives", "n_val_inactives", "note",
    ], orient="row").write_csv(TABLES / "litpcba_ave_shortcut_results.csv")

    (REPORTS / "litpcba_ave_shortcut_diagnostics.md").write_text(
        "# LIT-PCBA AVE — shortcut diagnostics with real split\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "## Protocol\n"
        "- `train` = `*_T.smi`, `validation` = `*_V.smi` (AVE-debiased).\n"
        "- `identity`  → score = 1 if val canonical SMILES seen as a train active.\n"
        "- `scaffold`  → score = 1 if val scaffold seen as a train active.\n"
        "- `ligand_knn` → score = max ECFP4 Tanimoto from val ligand to any train active.\n"
        f"- Validation capped at **{cfg.val_cap_per_target}** ligands per target (seed {cfg.rng_seed+11}).\n\n"
        "## Output\n"
        "`outputs/tables/litpcba_ave_shortcut_results.csv` — one row per (target, diagnostic).\n\n"
        "Skipped targets (rows with `diagnostic = skipped`) are explained in the\n"
        "`note` column — typically a missing class in train or validation.\n",
        encoding="utf-8")
    log_step("post_step", "ave_shortcut_diagnostics")


# -------------------- task 5: AVE graph (with train→val similarity edges) --------------------

def task_ave_graph(df: pl.DataFrame, cfg: RunConfig) -> None:
    log_step("pre_step", "ave_graph_build")
    df_g = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        df_g, include_decoy_protocol=False, include_protein_target=True,
    )

    # Add ligand_similar_to_ligand edges (train ligand -> val ligand, T>=threshold,
    # capped per val ligand).
    sim_edges = _ave_similarity_edges(df_g, cfg)
    if sim_edges.height > 0:
        edges = pl.concat([edges, sim_edges], how="vertical_relaxed").unique()

    nodes.write_parquet(PROCESSED / "litpcba_ave_nodes.parquet")
    edges.write_parquet(PROCESSED / "litpcba_ave_edges.parquet")
    (REPORTS / "litpcba_ave_graph_summary.md").write_text(
        _render_graph_summary("LIT-PCBA AVE_unbiased", nodes, edges,
                              extra=f"`ligand_similar_to_ligand` edges added for "
                              f"train→val pairs with Tanimoto ≥ {cfg.similarity_threshold} "
                              f"(top {cfg.similarity_top_k_per_val} per val ligand)."),
        encoding="utf-8")
    log_step("post_step", "ave_graph_build")


def _ave_similarity_edges(df_g: pl.DataFrame, cfg: RunConfig) -> pl.DataFrame:
    """Per target, for each (capped) validation ligand find its top-K train
    ligands at Tanimoto ≥ threshold, emit (val_lig, train_lig) edges."""
    rng = np.random.default_rng(cfg.rng_seed + 31)
    rows: List[tuple] = []
    targets = sorted(df_g.select("target").drop_nulls().unique().to_series().to_list())
    for target in targets:
        sub = df_g.filter((pl.col("target") == target) & pl.col("fp_bytes").is_not_null())
        tr_ligs  = sub.filter(pl.col("split") == "train").unique(subset=["ligand_node_id"])
        val_ligs = sub.filter(pl.col("split") == "validation").unique(subset=["ligand_node_id"])
        if tr_ligs.is_empty() or val_ligs.is_empty():
            continue
        # Cap train inactives + val rows (use ligand-level dedupe).
        tr_act  = tr_ligs.filter(pl.col("label") == 1)
        tr_ina  = tr_ligs.filter(pl.col("label") == 0)
        if tr_ina.height > cfg.inactive_cap_per_target:
            idx = rng.choice(tr_ina.height, size=cfg.inactive_cap_per_target, replace=False)
            tr_ina = tr_ina[sorted(idx.tolist())]
        tr_used = pl.concat([tr_act, tr_ina], how="vertical_relaxed")
        if val_ligs.height > cfg.val_cap_per_target:
            idx = rng.choice(val_ligs.height, size=cfg.val_cap_per_target, replace=False)
            val_ligs = val_ligs[sorted(idx.tolist())]
        tr_fps  = to_fp_objects(tr_used["fp_bytes"].to_list())
        tr_ids  = tr_used["ligand_node_id"].to_list()
        for v_fp, v_id in zip(to_fp_objects(val_ligs["fp_bytes"].to_list()),
                              val_ligs["ligand_node_id"].to_list()):
            if v_fp is None:
                continue
            sims = vc.bulk_tanimoto(v_fp, tr_fps)
            hits = np.where(sims >= cfg.similarity_threshold)[0]
            if hits.size == 0:
                continue
            top = hits[np.argsort(-sims[hits])][: cfg.similarity_top_k_per_val]
            for h in top:
                rows.append((v_id, tr_ids[h], "ligand_similar_to_ligand",
                             json.dumps({"tanimoto": round(float(sims[h]), 4),
                                         "target": target,
                                         "direction": "val_to_train"})))
    return pl.DataFrame(rows, schema=["src", "dst", "edge_type", "props"], orient="row")


def _render_graph_summary(label: str, nodes: pl.DataFrame, edges: pl.DataFrame,
                          extra: str = "") -> str:
    nbt = nodes.group_by("node_type").agg(pl.len().alias("n")).sort("node_type")
    eet = edges.group_by("edge_type").agg(pl.len().alias("n")).sort("edge_type")
    return (
        f"# {label} — graph summary\n\n"
        f"Nodes: **{nodes.height}** | Edges: **{edges.height}**\n\n"
        "## Nodes by type\n\n"
        + "\n".join(f"- {r['node_type']}: {r['n']}" for r in nbt.iter_rows(named=True))
        + "\n\n## Edges by type\n\n"
        + "\n".join(f"- {r['edge_type']}: {r['n']}" for r in eet.iter_rows(named=True))
        + (f"\n\n## Notes\n\n{extra}\n" if extra else "\n")
    )


# -------------------- task 6: contamination score with real split --------------------

def task_ave_contamination(df: pl.DataFrame, cfg: RunConfig) -> None:
    log_step("pre_step", "ave_contamination")
    rng = np.random.default_rng(cfg.rng_seed + 51)
    targets = sorted(df.select("target").drop_nulls().unique().to_series().to_list())
    parts = []
    for target in targets:
        sub = df.filter(pl.col("target") == target)
        tr  = sub.filter(pl.col("split") == "train")
        val = sub.filter(pl.col("split") == "validation")
        if tr.is_empty() or val.is_empty():
            continue

        # Identity / scaffold use the FULL train (cheap set membership).
        tr_smiles    = set(tr.filter(pl.col("smiles_canonical").is_not_null())["smiles_canonical"].to_list())
        tr_inchikeys = set(tr.filter(pl.col("inchikey").is_not_null())["inchikey"].to_list())
        tr_scafs     = set(tr.filter((pl.col("scaffold_smiles").is_not_null()) &
                                       (pl.col("scaffold_smiles") != ""))["scaffold_smiles"].to_list())

        # Analog (max-Tanimoto-to-train) is O(|train| * |val|). Cap train to
        # actives + sampled inactives so this finishes in seconds per target.
        tr_act = tr.filter(pl.col("label") == 1)
        tr_ina = tr.filter(pl.col("label") == 0)
        if tr_ina.height > cfg.inactive_cap_per_target:
            idx = rng.choice(tr_ina.height, size=cfg.inactive_cap_per_target, replace=False)
            tr_ina = tr_ina[sorted(idx.tolist())]
        tr_for_fp = pl.concat([tr_act, tr_ina], how="vertical_relaxed")
        tr_fps = to_fp_objects(tr_for_fp.filter(pl.col("fp_bytes").is_not_null())["fp_bytes"].to_list())
        log.info("AVE contam: %s  full_train=%d  capped_train=%d  val=%d",
                 target, tr.height, tr_for_fp.height, val.height)

        c_identity = np.fromiter(
            (1.0 if (s in tr_smiles) or (k in tr_inchikeys) else 0.0
             for s, k in zip(val["smiles_canonical"].to_list(), val["inchikey"].to_list())),
            float, val.height,
        )
        c_scaffold = np.fromiter(
            (1.0 if s in tr_scafs else 0.0 for s in val["scaffold_smiles"].to_list()),
            float, val.height,
        )
        val_fps = to_fp_objects(val["fp_bytes"].to_list())
        c_analog = vc.max_tanimoto_to_set(val_fps, tr_fps) if (tr_fps and val_fps) else np.full(val.height, -1.0)
        c_analog = np.where(c_analog < 0, np.nan, c_analog)
        c_source = np.ones(val.height, dtype=float)  # all from LIT-PCBA

        # c_total: weighted mean of identity, scaffold, analog (not c_source, per spec).
        components = np.vstack([c_identity, c_scaffold, c_analog])
        with np.errstate(invalid="ignore"):
            c_total = np.nanmean(components, axis=0)

        out = val.with_columns([
            pl.Series("c_identity", c_identity),
            pl.Series("c_scaffold", c_scaffold),
            pl.Series("c_analog", c_analog),
            pl.Series("c_source", c_source),
            pl.Series("c_total", c_total),
        ]).select([
            "target", "split", "label", "label_type",
            "smiles_canonical", "inchikey", "scaffold_smiles",
            "c_identity", "c_scaffold", "c_analog", "c_source", "c_total",
        ])
        parts.append(out)

    if not parts:
        log.warning("AVE contamination: nothing to score")
        return
    scored = pl.concat(parts, how="vertical_relaxed")
    scored.write_parquet(PROCESSED / "litpcba_ave_contamination_scores.parquet")

    summary = (scored.group_by("target")
               .agg([
                   pl.len().alias("n_val"),
                   pl.col("c_identity").mean().alias("c_identity_mean"),
                   pl.col("c_scaffold").mean().alias("c_scaffold_mean"),
                   pl.col("c_analog").drop_nans().mean().alias("c_analog_mean"),
                   pl.col("c_total").drop_nans().mean().alias("c_total_mean"),
               ])
               .sort("target"))
    summary.write_csv(TABLES / "litpcba_ave_contamination_score_summary.csv")
    (REPORTS / "litpcba_ave_contamination_score_summary.md").write_text(
        _render_contam_summary(summary), encoding="utf-8")
    log_step("post_step", "ave_contamination")


def _render_contam_summary(summary: pl.DataFrame) -> str:
    return (
        "# LIT-PCBA AVE — MVP-1 contamination score\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "Per-validation-example heuristic score in [0, 1], aggregating identity,\n"
        "scaffold, analog (max ECFP4 Tanimoto), and source-share. `c_source` is\n"
        "tracked separately and excluded from `c_total` so that 'all LIT-PCBA'\n"
        "does not trivially saturate the score.\n\n"
        "## Per-target means (`c_total` excludes `c_source`)\n\n"
        + _polars_to_md_table(summary) + "\n\n"
        "Full per-example: `data/processed/litpcba_ave_contamination_scores.parquet`.\n"
    )


def _polars_to_md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(
        ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))) for v in row
    ) + " |" for row in df.iter_rows())
    return "\n".join([header, sep, body])


# -------------------- task 7-8: DEKOIS extract + load --------------------

def task_dekois_extract() -> Optional[Path]:
    log_step("pre_step", "dekois_extract")
    zip_path = RAW / "DEKOIS" / "DEKOIS2.zip"
    extract_root = RAW / "DEKOIS" / "extracted"
    if not zip_path.exists():
        log.warning("DEKOIS2.zip not found at %s", zip_path)
        (REPORTS / "dekois_layout_report.md").write_text(
            f"# DEKOIS extract — failed\n\nNo zip at `{zip_path}`.\n",
            encoding="utf-8")
        return None
    extract_root.mkdir(parents=True, exist_ok=True)
    sentinel = extract_root / ".extracted_ok"
    if sentinel.exists():
        log.info("DEKOIS already extracted at %s", extract_root)
    else:
        if not check_disk(2.0, "DEKOIS extract"):
            log.warning("aborting DEKOIS extract — low disk")
            return None
        log.info("DEKOIS: extracting %s -> %s", zip_path, extract_root)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
        sentinel.write_text("")
    base = extract_root / "DEKOIS2"
    targets = sorted([p.name for p in base.iterdir() if p.is_dir()])
    n_smi = sum(1 for _ in base.rglob("active_decoys.smi"))
    (REPORTS / "dekois_layout_report.md").write_text(
        "# DEKOIS 2.0 — layout report\n\n"
        f"- extraction root: `{extract_root}`\n"
        f"- targets discovered: **{len(targets)}**\n"
        f"- `active_decoys.smi` files: **{n_smi}**\n"
        f"- targets: {', '.join(targets)}\n\n"
        "## Convention\n\n"
        "Each `active_decoys.smi` contains both actives and decoys, distinguished\n"
        "by the second-column identifier prefix:\n"
        "  - `BDB...`  → active (40 per target, sourced from BindingDB)\n"
        "  - `ZINC...` → decoy  (~1100 per target, sourced from ZINC)\n"
        "Any other prefix is reported as `unknown` in `dekois_dataset_summary.md`.\n",
        encoding="utf-8")
    log_step("post_step", "dekois_extract")
    return extract_root


def task_dekois_load(extract_root: Path, cfg: RunConfig) -> Optional[pl.DataFrame]:
    log_step("pre_step", "dekois_load_and_featurize")
    cached = PROCESSED / "dekois_examples.parquet"
    if cached.exists():
        df = pl.read_parquet(cached)
        log.info("DEKOIS: loaded cached %s (%d rows)", cached, df.height)
        log_step("post_step", "dekois_load_cached")
        return df
    df = load_dekois.load_all(extract_root)
    log.info("DEKOIS: loaded %d examples across %d targets",
             df.height, df.select("target").n_unique())
    n_unknown = int((df["label"] == -1).sum())
    if n_unknown > 0:
        log.warning("DEKOIS: %d rows with unknown id prefix", n_unknown)
    df = df.filter(pl.col("label") >= 0)  # drop unknowns from downstream

    feats = parallel_featurize(df["smiles_input"].to_list(), workers=cfg.workers)
    df = df.with_columns([
        feats["smiles_canonical"], feats["inchikey"],
        feats["scaffold_smiles"], feats["parse_ok"],
    ])
    fps = parallel_ecfp_bytes(df["smiles_canonical"].to_list(), workers=cfg.workers)
    df = df.with_columns(pl.Series("fp_bytes", fps, dtype=pl.Binary))
    df.write_parquet(cached)

    by = (df.group_by(["target", "label_type"]).agg(pl.len().alias("n"))
          .sort(["target", "label_type"]))
    counts = {
        "n_examples": df.height,
        "n_targets": df.select("target").n_unique(),
        "n_actives": int((df["label"] == 1).sum()),
        "n_decoys":  int((df["label"] == 0).sum()),
        "n_unknown_dropped": n_unknown,
        "parse_ok": int(df["parse_ok"].sum()),
        "parse_fail": int((~df["parse_ok"]).sum()),
    }
    (REPORTS / "dekois_dataset_summary.md").write_text(
        "# DEKOIS 2.0 — dataset summary\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in counts.items()) + "\n\n"
        "## Per target × label_type\n\n"
        + by.to_pandas().to_string(index=False) + "\n",
        encoding="utf-8")
    log_step("post_step", "dekois_load_and_featurize")
    return df


def task_dekois_graph(df: pl.DataFrame) -> None:
    log_step("pre_step", "dekois_graph_build")
    df_g = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        df_g, include_decoy_protocol=True, include_protein_target=True,
    )
    nodes.write_parquet(PROCESSED / "dekois_nodes.parquet")
    edges.write_parquet(PROCESSED / "dekois_edges.parquet")
    (REPORTS / "dekois_graph_summary.md").write_text(
        _render_graph_summary("DEKOIS 2.0", nodes, edges), encoding="utf-8")
    log_step("post_step", "dekois_graph_build")


# -------------------- task 9: DEKOIS diagnostics + cross-protocol --------------------

def task_dekois_diagnostics(df: pl.DataFrame, cfg: RunConfig) -> None:
    log_step("pre_step", "dekois_diagnostics")
    rng = np.random.default_rng(cfg.rng_seed + 91)
    rows = []
    targets = sorted(df.select("target").drop_nulls().unique().to_series().to_list())
    for target in targets:
        sub = df.filter(pl.col("target") == target)
        actives = sub.filter(pl.col("label") == 1)
        decoys  = sub.filter(pl.col("label") == 0)
        if actives.height < 5 or decoys.height < 5:
            rows.append((target, "skipped", "split_80_20_actives", None, None, None,
                         actives.height, decoys.height))
            continue
        pool = pl.concat([actives, decoys], how="vertical_relaxed").with_row_count("_pool_idx")
        perm = rng.permutation(pool.filter(pl.col("label") == 1)["_pool_idx"].to_numpy())
        n_train = max(1, int(0.8 * len(perm)))
        train_idx = perm[:n_train].tolist()
        train_mask = pl.col("_pool_idx").is_in(train_idx) & (pl.col("label") == 1)
        eval_mask = ~pl.col("_pool_idx").is_in(train_idx)

        tr_act_smi  = set(pool.filter(train_mask & pl.col("smiles_canonical").is_not_null())["smiles_canonical"].to_list())
        tr_act_scaf = set(pool.filter(train_mask & pl.col("scaffold_smiles").is_not_null())["scaffold_smiles"].to_list())
        tr_act_fps  = to_fp_objects(pool.filter(train_mask & pl.col("fp_bytes").is_not_null())["fp_bytes"].to_list())

        eval_df = pool.filter(eval_mask)
        y_true = eval_df["label"].to_numpy().astype(int)
        s_id = np.fromiter((1.0 if s in tr_act_smi else 0.0 for s in eval_df["smiles_canonical"].to_list()),
                            float, eval_df.height)
        s_sc = np.fromiter((1.0 if s in tr_act_scaf else 0.0 for s in eval_df["scaffold_smiles"].to_list()),
                            float, eval_df.height)
        eval_fps = to_fp_objects(eval_df["fp_bytes"].to_list())
        s_knn = vc.max_tanimoto_to_set(eval_fps, tr_act_fps) if tr_act_fps else np.zeros(eval_df.height, dtype=np.float32)
        s_knn = np.where(s_knn < 0, 0.0, s_knn)

        for name, score in (("identity", s_id), ("scaffold", s_sc), ("ligand_knn", s_knn)):
            rows.append((target, name, "split_80_20_actives",
                         _safe_auroc(y_true, score),
                         _safe_ap(y_true, score),
                         _ef(y_true, score, 0.01),
                         actives.height, decoys.height))

    pl.DataFrame(rows, schema=[
        "target", "diagnostic", "protocol", "auroc", "ap", "ef1pct",
        "n_actives", "n_decoys",
    ], orient="row").write_csv(TABLES / "dekois_shortcut_results.csv")

    (REPORTS / "dekois_shortcut_diagnostics.md").write_text(
        "# DEKOIS 2.0 — shortcut diagnostics\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "Same protocol as the MVP-0 DUD-E run: deterministic 80/20 split over\n"
        "each target's actives (40 per target in DEKOIS 2.0); all decoys go into\n"
        "the eval pool. We report identity / scaffold memorization and ligand KNN.\n\n"
        "Cross-protocol comparison vs DUD-E is in\n"
        "`outputs/reports/decoy_protocol_comparison_dude_vs_dekois.md`.\n",
        encoding="utf-8")

    # Cross-protocol comparison with DUD-E if MVP-0 results exist.
    dude_csv = TABLES / "dude_shortcut_results.csv"
    if dude_csv.exists():
        _write_decoy_protocol_comparison()
    log_step("post_step", "dekois_diagnostics")


def _write_decoy_protocol_comparison() -> None:
    dude = pl.read_csv(TABLES / "dude_shortcut_results.csv")
    dek  = pl.read_csv(TABLES / "dekois_shortcut_results.csv")
    summ_dude = (dude.filter(pl.col("diagnostic") == "ligand_knn")
                 .select(["target", "auroc", "ap", "ef1pct"])
                 .rename({"auroc": "dude_auroc", "ap": "dude_ap", "ef1pct": "dude_ef1pct"}))
    summ_dek  = (dek.filter(pl.col("diagnostic") == "ligand_knn")
                 .select(["target", "auroc", "ap", "ef1pct"])
                 .rename({"auroc": "dekois_auroc", "ap": "dekois_ap", "ef1pct": "dekois_ef1pct"}))
    overall_dude = dude.filter(pl.col("diagnostic") == "ligand_knn")["auroc"].drop_nulls().mean()
    overall_dek  = dek.filter(pl.col("diagnostic") == "ligand_knn")["auroc"].drop_nulls().mean()

    def _agg(name: str, frame: pl.DataFrame, col: str = "auroc") -> str:
        d = frame.filter(pl.col("diagnostic") == name)[col].drop_nulls()
        if d.is_empty():
            return f"  - {name}: no data"
        return f"  - {name}: mean={d.mean():.4f} median={d.median():.4f} n={d.len()}"

    body = (
        "# Decoy-protocol comparison — DUD-E vs DEKOIS 2.0\n\n"
        f"Generated by `vsleakkg.run_mvp1_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "Ligand-KNN AUROC (max ECFP4 Tanimoto to training actives) is a stand-in\n"
        "for the ligand-side shortcut available in each decoy protocol. Higher\n"
        "AUROC means the protocol does NOT effectively suppress ligand similarity\n"
        "between actives and decoys.\n\n"
        f"- **DUD-E mean ligand-KNN AUROC**: {overall_dude:.4f}\n"
        f"- **DEKOIS mean ligand-KNN AUROC**: {overall_dek:.4f}\n\n"
        "## All diagnostics — mean / median AUROC per protocol\n\n"
        "### DUD-E\n"
        + "\n".join(_agg(n, dude) for n in ("ligand_knn", "identity", "scaffold")) + "\n\n"
        "### DEKOIS\n"
        + "\n".join(_agg(n, dek)  for n in ("ligand_knn", "identity", "scaffold")) + "\n\n"
        "## Per-target ligand-KNN comparison\n\n"
        + _polars_to_md_table(summ_dude.join(summ_dek, on="target", how="inner").sort("target"))
        + "\n\nTargets only present in one protocol are omitted from the inner join.\n"
    )
    (REPORTS / "decoy_protocol_comparison_dude_vs_dekois.md").write_text(body, encoding="utf-8")


# -------------------- task 10: combined MVP-1 graph --------------------

def task_combined_graph() -> None:
    log_step("pre_step", "combined_mvp1_graph")
    nodes_parts, edges_parts = [], []
    for label, n_path, e_path in (
        ("LIT-PCBA AVE", PROCESSED / "litpcba_ave_nodes.parquet", PROCESSED / "litpcba_ave_edges.parquet"),
        ("DUD-E",        PROCESSED / "dude_nodes.parquet",        PROCESSED / "dude_edges.parquet"),
        ("DEKOIS",       PROCESSED / "dekois_nodes.parquet",      PROCESSED / "dekois_edges.parquet"),
    ):
        if n_path.exists() and e_path.exists():
            nodes_parts.append(pl.read_parquet(n_path))
            edges_parts.append(pl.read_parquet(e_path))
            log.info("combined: added %s", label)
    if not nodes_parts:
        log.warning("combined: no per-dataset graphs to combine")
        return
    nodes = pl.concat(nodes_parts, how="vertical_relaxed").unique(subset=["node_id"])
    edges = pl.concat(edges_parts, how="vertical_relaxed").unique()
    nodes.write_parquet(PROCESSED / "mvp1_nodes.parquet")
    edges.write_parquet(PROCESSED / "mvp1_edges.parquet")
    (REPORTS / "mvp1_graph_summary.md").write_text(
        _render_graph_summary("MVP-1 combined (LIT-PCBA AVE + DUD-E + DEKOIS)", nodes, edges),
        encoding="utf-8")
    log_step("post_step", "combined_mvp1_graph")


# -------------------- task 11: final report --------------------

def task_final_report(cfg: RunConfig) -> None:
    def _read(p: Path) -> str:
        return p.read_text(encoding="utf-8") if p.exists() else "(missing)"

    body = []
    body.append(f"# VS-LeakKG MVP-1 audit report\n\nGenerated {datetime.now(timezone.utc).isoformat()}.\n")
    body.append("## What changed from MVP-0\n\n"
                "- **LIT-PCBA is now audited with the real AVE_unbiased train/validation\n"
                "  split** (`active_T.smi`, `active_V.smi`, `inactive_T.smi`,\n"
                "  `inactive_V.smi`). MVP-0's synthetic 80/20 stand-in is retired.\n"
                "- **DEKOIS 2.0** is loaded and used as a second decoy protocol\n"
                "  alongside DUD-E. A direct DUD-E vs DEKOIS shortcut comparison is\n"
                "  emitted.\n"
                "- Contamination score now uses the real held-out validation set\n"
                "  rather than the synthetic actives-as-train trick.\n"
                "- The combined MVP-1 graph adds `Split = {train, validation, unknown}`\n"
                "  and `DecoyProtocol = {DUD-E, DEKOIS}` to the namespace.\n")
    body.append("## LIT-PCBA AVE — layout\n\n" + _read(REPORTS / "litpcba_ave_layout_report.md") + "\n")
    body.append("## LIT-PCBA AVE — leakage findings\n\n" + _read(REPORTS / "litpcba_ave_leakage_report.md") + "\n")
    body.append("## LIT-PCBA AVE — shortcut diagnostics\n\n" + _read(REPORTS / "litpcba_ave_shortcut_diagnostics.md") + "\n")
    body.append("## LIT-PCBA AVE — contamination score\n\n" + _read(REPORTS / "litpcba_ave_contamination_score_summary.md") + "\n")
    body.append("## LIT-PCBA AVE — graph\n\n" + _read(REPORTS / "litpcba_ave_graph_summary.md") + "\n")
    body.append("## DUD-E — summary (unchanged from MVP-0)\n\n" + _read(REPORTS / "dude_dataset_summary.md") + "\n\n"
                + _read(REPORTS / "dude_shortcut_diagnostics.md") + "\n")
    body.append("## DEKOIS 2.0 — summary\n\n" + _read(REPORTS / "dekois_dataset_summary.md") + "\n\n"
                + _read(REPORTS / "dekois_shortcut_diagnostics.md") + "\n")
    body.append("## Decoy-protocol comparison: DUD-E vs DEKOIS\n\n"
                + _read(REPORTS / "decoy_protocol_comparison_dude_vs_dekois.md") + "\n")
    body.append("## Combined MVP-1 graph\n\n" + _read(REPORTS / "mvp1_graph_summary.md") + "\n")
    body.append("## Limitations\n\n"
                "- Analog overlap and contamination `c_analog` cap train inactives\n"
                f"  per target at **{cfg.inactive_cap_per_target}** (sampled, seed {cfg.rng_seed})\n"
                f"  and validation per target at **{cfg.val_cap_per_target}** for the\n"
                "  expensive max-Tanimoto pass. Identity / scaffold overlaps use the\n"
                "  FULL train + validation sets.\n"
                "- DEKOIS lacks a built-in train/validation split — we use the same\n"
                "  deterministic 80/20 over actives that the MVP-0 DUD-E run used,\n"
                "  for a like-for-like comparison.\n"
                "- Assay metadata not yet attached (no ChEMBL join, `c_assay = NaN`).\n"
                "- BindingDB cross-source linking is still pending.\n"
                "- BayesBind / BigBind / PLINDER not in this audit.\n")
    body.append("## Next steps\n\n"
                "1. Join ChEMBL assay + document metadata so `c_assay` becomes real\n"
                "   and `(Example)-[from_assay]->(Assay)` edges populate the KG.\n"
                "2. Map BindingDB by InChIKey to add `(Ligand)-[also_in]->(Ligand)`\n"
                "   cross-source leakage edges.\n"
                "3. Extract BayesBind V1.5 (~110 MB; safe) for the phase-2 VS KNN\n"
                "   benchmark; keep BigBind compressed until the loader is ready.\n"
                "4. Once available, compare against PLINDER / DataSAIL-style\n"
                "   similarity-only splits as a no-leakage-by-construction baseline.\n"
                "5. Replace the heuristic `c_total` with KG path-based contamination\n"
                "   features (shortest path leakage through shared scaffold / assay\n"
                "   nodes) once the graph carries assay metadata.\n")
    (REPORTS / "mvp1_audit_report.md").write_text("\n".join(body), encoding="utf-8")


# -------------------- main --------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--inactive-cap", type=int, default=5000)
    parser.add_argument("--val-cap", type=int, default=2000)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument("--skip-litpcba-ave", action="store_true")
    parser.add_argument("--skip-dekois", action="store_true")
    args = parser.parse_args(argv)

    cfg = RunConfig(
        workers=args.workers,
        inactive_cap_per_target=args.inactive_cap,
        val_cap_per_target=args.val_cap,
        similarity_threshold=args.similarity_threshold,
        similarity_top_k_per_val=args.similarity_top_k,
        skip_litpcba_ave=args.skip_litpcba_ave,
        skip_dekois=args.skip_dekois,
    )

    log_step("mvp1_start", "vs-leakkg")

    if not cfg.skip_litpcba_ave:
        df_ave = task_ave_load(cfg)
        task_ave_leakage(df_ave, cfg)
        task_ave_diagnostics(df_ave, cfg)
        task_ave_graph(df_ave, cfg)
        task_ave_contamination(df_ave, cfg)

    if not cfg.skip_dekois:
        extract_root = task_dekois_extract()
        if extract_root is not None:
            df_dek = task_dekois_load(extract_root, cfg)
            if df_dek is not None and df_dek.height > 0:
                task_dekois_graph(df_dek)
                task_dekois_diagnostics(df_dek, cfg)

    task_combined_graph()
    task_final_report(cfg)

    log_step("mvp1_end", "vs-leakkg")
    print()
    print("MVP-1 audit complete. See outputs/reports/mvp1_audit_report.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
