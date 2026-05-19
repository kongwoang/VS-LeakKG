"""MVP audit orchestrator for VS-LeakKG.

Implements Step 6 / Tasks 1–10 from the project spec. Designed to be safe to
re-run: each task short-circuits gracefully if its inputs are missing and
writes a missing-input report instead of crashing the pipeline.

Usage:
    python -m vsleakkg.run_mvp_audit                 # full run, default caps
    python -m vsleakkg.run_mvp_audit --inactive-cap 5000 --decoy-cap 2000
    python -m vsleakkg.run_mvp_audit --skip-dude     # LIT-PCBA only

Outputs land under:
    data/processed/         parquet tables
    outputs/tables/         per-target / per-pair leakage CSVs
    outputs/reports/        markdown summaries
    outputs/logs/           run + disk logs
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Make the package importable when run as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vsleakkg import chem as vc
from vsleakkg import build_graph as vb
from vsleakkg import load_dude, load_litpcba
from vsleakkg import audit_ligand as audit
from vsleakkg import diagnostics as diag
from vsleakkg import contamination_score as cscore

PROJECT_ROOT = Path("D:/hoangpc/VS-LeakKG")
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"
REPORTS = PROJECT_ROOT / "outputs" / "reports"
LOGS = PROJECT_ROOT / "outputs" / "logs"
DISK_LOG = LOGS / "mvp_audit_disk_usage.log"
RUN_LOG = LOGS / "mvp_audit.log"

for d in (PROCESSED, TABLES, REPORTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vsleakkg.mvp")


# -------------------- disk logging --------------------

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


def log_disk(event: str, target: str) -> None:
    """Mirror scripts/log_disk.ps1 — write a structured block to the audit
    disk-usage log."""
    import shutil
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"==== {ts} ====",
        f"event: {event}",
        f"target: {target}",
        f"cwd: {os.getcwd()}",
    ]
    for part in ("C:/", "D:/"):
        try:
            usage = shutil.disk_usage(part)
            lines.append(f"  drive {part}: used={usage.used/1024**3:.2f}GB "
                         f"free={usage.free/1024**3:.2f}GB")
        except OSError:
            pass
    proj_mb = _du_bytes(PROJECT_ROOT) / 1024**2
    lines.append(f"-- project size: {proj_mb:.2f} MB ({PROJECT_ROOT})")
    lines.append("-- lsblk: unavailable on Windows")
    lines.append("-- free -h: unavailable on Windows")
    lines.append("")
    DISK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def check_disk(min_free_gb: float, label: str) -> bool:
    import shutil
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
    except OSError:
        return True
    free_gb = usage.free / 1024**3
    if free_gb < min_free_gb:
        log.warning("check_disk FAIL free=%.2fGB need>=%.2fGB label=%s",
                    free_gb, min_free_gb, label)
        return False
    return True


# -------------------- multiprocessing featurization --------------------
# Helpers must be top-level to be picklable on Windows.

def _featurize_chunk(smiles_list: List[str]) -> List[tuple]:
    out = []
    for smi in smiles_list:
        f = vc.featurize(smi)
        out.append((f.smiles_canonical, f.inchikey, f.scaffold_smiles, f.parse_ok))
    return out


def _ecfp_chunk(smiles_list: List[str]) -> List[Optional[bytes]]:
    return [vc.ecfp_bytes(s) if s else None for s in smiles_list]


def parallel_featurize(smiles: List[str], workers: int, chunksize: int = 2000) -> pl.DataFrame:
    """Run canonicalize/scaffold/InChIKey for each SMILES across processes."""
    n = len(smiles)
    if n == 0:
        return pl.DataFrame(schema={
            "smiles_canonical": pl.Utf8, "inchikey": pl.Utf8,
            "scaffold_smiles": pl.Utf8, "parse_ok": pl.Boolean,
        })
    chunks = [smiles[i:i+chunksize] for i in range(0, n, chunksize)]
    canonical, inchikey, scaffold, ok = [], [], [], []
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for j, batch in enumerate(pool.imap(_featurize_chunk, chunks, chunksize=1), 1):
            for can, ik, sc, parse_ok in batch:
                canonical.append(can)
                inchikey.append(ik)
                scaffold.append(sc)
                ok.append(parse_ok)
            if j % 50 == 0 or j == len(chunks):
                log.info("featurize: %d / %d chunks (%d / %d rows)",
                         j, len(chunks), len(canonical), n)
    return pl.DataFrame({
        "smiles_canonical": canonical,
        "inchikey": inchikey,
        "scaffold_smiles": scaffold,
        "parse_ok": ok,
    })


def serial_ecfp(smiles: List[str]) -> List[Optional[object]]:
    """ExplicitBitVect objects can't be pickled cleanly across procs on Windows
    — we keep ECFP in-process. Single-thread is acceptable for the sampled
    sets we run analog audits on."""
    return [vc.ecfp(s) if s else None for s in smiles]


# -------------------- pipeline tasks --------------------

@dataclasses.dataclass
class RunConfig:
    workers: int = 12
    inactive_cap_per_target: int = 5000      # LIT-PCBA inactives sampled per target for ECFP/analog
    decoy_cap_per_target: int = 2000         # DUD-E decoys sampled per target for ECFP/analog
    analog_thresholds: Tuple[float, ...] = (0.6, 0.8, 0.9)
    skip_litpcba: bool = False
    skip_dude: bool = False
    skip_graph: bool = False
    rng_seed: int = 17


def task_1_read_setup_state() -> dict:
    state = {"setup_report_exists": (PROJECT_ROOT / "outputs/setup_report.md").exists(),
             "manifest_exists":     (PROJECT_ROOT / "data/MANIFEST.md").exists()}
    log.info("setup state: %s", state)
    return state


def task_2_check_datasets() -> dict:
    avail = {}
    for name in ("LIT-PCBA", "DUD-E", "ChEMBL", "BindingDB"):
        d = RAW / name
        files = list(d.rglob("*")) if d.exists() else []
        size = sum(f.stat().st_size for f in files if f.is_file())
        avail[name] = {"path": str(d), "size_bytes": size, "non_empty": size > 0}
    log.info("datasets: %s", json.dumps(avail, indent=None, default=str))
    if not avail["LIT-PCBA"]["non_empty"]:
        (REPORTS / "missing_litpcba.md").write_text(
            "# LIT-PCBA missing\n\nNo data under `data/raw/LIT-PCBA/`. "
            "See `data/raw/manual_downloads_needed/LIT-PCBA_TODO.md`.\n",
            encoding="utf-8")
    if not avail["DUD-E"]["non_empty"]:
        (REPORTS / "missing_dude.md").write_text(
            "# DUD-E missing\n\nNo data under `data/raw/DUD-E/`. "
            "See `data/raw/manual_downloads_needed/DUD-E_TODO.md` if present.\n",
            encoding="utf-8")
    return avail


def _featurize_examples(df: pl.DataFrame, cfg: RunConfig, label: str) -> pl.DataFrame:
    log.info("[%s] featurizing %d SMILES with %d workers", label, df.height, cfg.workers)
    t0 = time.time()
    feats = parallel_featurize(df["smiles_input"].to_list(), workers=cfg.workers)
    df = df.with_columns([
        feats["smiles_canonical"], feats["inchikey"],
        feats["scaffold_smiles"], feats["parse_ok"],
    ])
    log.info("[%s] featurize done in %.1fs; parse_ok=%d/%d",
             label, time.time()-t0, int(df["parse_ok"].sum()), df.height)
    return df


def _ecfp_for_subset(df: pl.DataFrame, *, mask, label: str) -> pl.DataFrame:
    sub = df.filter(mask)
    log.info("[%s] computing ECFP4 for %d rows", label, sub.height)
    fps = serial_ecfp(sub["smiles_canonical"].to_list())
    # Attach via row-aligned join.
    sub = sub.with_columns(pl.Series("_fp", fps, dtype=pl.Object))
    return sub


def task_3_litpcba_sanity(cfg: RunConfig) -> Optional[pl.DataFrame]:
    log_disk("pre_step", "task_3_litpcba_sanity")
    cached = PROCESSED / "litpcba_examples.parquet"
    if cached.exists():
        log.info("LIT-PCBA: loading cached featurized parquet %s", cached)
        df = pl.read_parquet(cached)
        log.info("LIT-PCBA cached examples: %d", df.height)
    else:
        try:
            df = load_litpcba.load_all(RAW / "LIT-PCBA")
        except Exception as exc:
            log.warning("LIT-PCBA load failed: %s", exc)
            (REPORTS / "missing_litpcba.md").write_text(
                f"# LIT-PCBA load failed\n\n{exc}\n", encoding="utf-8")
            return None
        log.info("LIT-PCBA examples loaded: %d", df.height)
        df = _featurize_examples(df, cfg, "LIT-PCBA")
        df.write_parquet(cached)
        log.info("wrote %s", cached)

    # Counts.
    counts = {
        "n_examples": df.height,
        "n_targets": df.select("target").n_unique(),
        "n_unique_canonical": df.filter(pl.col("smiles_canonical").is_not_null())
                                .select("smiles_canonical").n_unique(),
        "n_unique_inchikey": df.filter(pl.col("inchikey").is_not_null())
                               .select("inchikey").n_unique(),
        "n_unique_scaffold": df.filter(pl.col("scaffold_smiles").is_not_null())
                               .select("scaffold_smiles").n_unique(),
        "n_splits": df.select("split").n_unique(),
        "n_actives": int((df["label"] == 1).sum()),
        "n_inactives": int((df["label"] == 0).sum()),
        "parse_ok": int(df["parse_ok"].sum()),
        "parse_fail": int((~df["parse_ok"]).sum()),
    }
    log.info("LIT-PCBA counts: %s", counts)

    # 1. Identity duplicates across targets and across labels.
    id_by_target = audit.identity_overlap_table(df, "target", "smiles_canonical")
    id_by_target.write_csv(TABLES / "litpcba_identity_leakage_by_target.csv")
    id_by_label = audit.identity_overlap_table(df, "label_type", "smiles_canonical")
    id_by_label.write_csv(TABLES / "litpcba_identity_leakage_by_label.csv")

    # 2. Intra-set duplicates (per target × label_type).
    df_grp = df.with_columns(pl.format("{}__{}", "target", "label_type").alias("_grp"))
    intra = audit.intra_group_duplicates(df_grp, "_grp", "smiles_canonical")
    intra.write_csv(TABLES / "litpcba_identity_intra_duplicates.csv")

    # 3. Scaffold overlap across targets (excluding empty scaffold for sanity).
    scaf_df = df.filter((pl.col("scaffold_smiles").is_not_null()) & (pl.col("scaffold_smiles") != ""))
    scaf_overlap = audit.scaffold_overlap_table(scaf_df, "target")
    scaf_overlap.write_csv(TABLES / "litpcba_scaffold_overlap.csv")

    # 4. Analog overlap at thresholds — between actives and a sample of inactives, per target.
    log.info("LIT-PCBA: analog overlap, capping inactives to %d per target",
             cfg.inactive_cap_per_target)
    rng = np.random.default_rng(cfg.rng_seed)
    rows = []
    for target in sorted(df.select("target").drop_nulls().unique().to_series().to_list()):
        actives = df.filter((pl.col("target") == target) & (pl.col("label") == 1) &
                            (pl.col("smiles_canonical").is_not_null()))
        inactives = df.filter((pl.col("target") == target) & (pl.col("label") == 0) &
                              (pl.col("smiles_canonical").is_not_null()))
        if inactives.height > cfg.inactive_cap_per_target:
            idx = rng.choice(inactives.height, size=cfg.inactive_cap_per_target, replace=False)
            inactives = inactives[sorted(idx.tolist())]
        fps_a = serial_ecfp(actives["smiles_canonical"].to_list())
        fps_b = serial_ecfp(inactives["smiles_canonical"].to_list())
        counts_a_vs_b = audit.analog_overlap_pairs(fps_a, fps_b, cfg.analog_thresholds)
        for t, c in counts_a_vs_b.items():
            rows.append((target, "actives_vs_inactives", t, c,
                         len([f for f in fps_a if f is not None]),
                         len([f for f in fps_b if f is not None])))
        # actives vs actives (intra)
        counts_a_vs_a = audit.analog_overlap_pairs(fps_a, fps_a, cfg.analog_thresholds)
        # Subtract diagonal (each active matches itself at Tanimoto=1).
        n_a = len([f for f in fps_a if f is not None])
        for t, c in counts_a_vs_a.items():
            adj = max(0, c - n_a)  # remove diagonal
            rows.append((target, "actives_vs_actives", t, adj, n_a, n_a))
    analog = pl.DataFrame(rows,
        schema=["target", "pair", "threshold", "n_pairs_above", "n_left", "n_right"],
        orient="row")
    analog.write_csv(TABLES / "litpcba_analog_overlap.csv")

    # 5. Summary report.
    (REPORTS / "litpcba_dataset_summary.md").write_text(_render_litpcba_summary(counts), encoding="utf-8")
    log_disk("post_step", "task_3_litpcba_sanity")
    return df


def _render_litpcba_summary(counts: dict) -> str:
    bullets = "\n".join(f"- **{k}**: {v}" for k, v in counts.items())
    return (
        "# LIT-PCBA — dataset summary\n\n"
        f"Generated by `vsleakkg.run_mvp_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        f"## Counts\n\n{bullets}\n\n"
        "## Splits\n\n"
        "The LIT-PCBA `full_data.tgz` archive does NOT contain train/val/test split\n"
        "files. All examples were assigned `split=\"unknown\"`. Downstream\n"
        "memorization/KNN diagnostics that need a held-out split are therefore\n"
        "computed by using *inactives as the eval set against actives as the\n"
        "train set* (per-target). This is documented as a stand-in, not a\n"
        "ground-truth train/test partition.\n"
    )


def task_4_litpcba_graph(df: pl.DataFrame) -> Optional[Tuple[pl.DataFrame, pl.DataFrame]]:
    log_disk("pre_step", "task_4_litpcba_graph")
    df = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        df, include_decoy_protocol=False, include_protein_target=True,
    )
    nodes.write_parquet(PROCESSED / "litpcba_nodes.parquet")
    edges.write_parquet(PROCESSED / "litpcba_edges.parquet")
    (REPORTS / "litpcba_graph_summary.md").write_text(
        _render_graph_summary("LIT-PCBA", nodes, edges), encoding="utf-8")
    log_disk("post_step", "task_4_litpcba_graph")
    return nodes, edges


def _polars_to_md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(
        ("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))) for v in row
    ) + " |" for row in df.iter_rows())
    return "\n".join([header, sep, body])


def _render_graph_summary(label: str, nodes: pl.DataFrame, edges: pl.DataFrame) -> str:
    nbt = nodes.group_by("node_type").agg(pl.len().alias("n")).sort("node_type")
    eet = edges.group_by("edge_type").agg(pl.len().alias("n")).sort("edge_type")
    return (
        f"# {label} — MVP graph summary\n\n"
        f"Nodes: **{nodes.height}** | Edges: **{edges.height}**\n\n"
        "## Nodes by type\n\n"
        + "\n".join(f"- {r['node_type']}: {r['n']}" for r in nbt.iter_rows(named=True))
        + "\n\n## Edges by type\n\n"
        + "\n".join(f"- {r['edge_type']}: {r['n']}" for r in eet.iter_rows(named=True))
        + "\n"
    )


def task_5_litpcba_diagnostics(df: pl.DataFrame, cfg: RunConfig) -> None:
    """Per-target diagnostics: random 80/20 split over actives, full inactive
    pool (sampled) acts as the held-out negative pool. Reports ligand KNN,
    identity memorization, and scaffold memorization."""
    log_disk("pre_step", "task_5_litpcba_diagnostics")
    rows = []
    rng = np.random.default_rng(cfg.rng_seed)
    for target in sorted(df.select("target").drop_nulls().unique().to_series().to_list()):
        sub = df.filter(pl.col("target") == target)
        actives = sub.filter(pl.col("label") == 1)
        inactives = sub.filter(pl.col("label") == 0)
        if actives.height < 5 or inactives.height < 5:
            rows.append((target, "skipped", "split_80_20_actives", None, None, None,
                         actives.height, inactives.height))
            continue
        if inactives.height > cfg.inactive_cap_per_target:
            idx = rng.choice(inactives.height, size=cfg.inactive_cap_per_target, replace=False)
            inactives = inactives[sorted(idx.tolist())]
        pool = pl.concat([actives, inactives], how="vertical_relaxed").with_row_count("_pool_idx")
        pool = pool.with_columns(pl.Series("_fp", serial_ecfp(pool["smiles_canonical"].to_list()),
                                           dtype=pl.Object))
        actives_idx_in_pool = pool.filter(pl.col("label") == 1)["_pool_idx"].to_numpy()
        perm = rng.permutation(actives_idx_in_pool)
        n_train = max(1, int(0.8 * len(perm)))
        train_actives_pool_idx = perm[:n_train].tolist()
        train_mask = pl.col("_pool_idx").is_in(train_actives_pool_idx) & (pl.col("label") == 1)
        eval_mask = ~pl.col("_pool_idx").is_in(train_actives_pool_idx)

        knn = diag.ligand_knn_max_tanimoto(pool, train_mask=train_mask, eval_mask=eval_mask)
        rows.append((target, "ligand_knn", "split_80_20_actives", knn["auroc"], knn["ap"], knn["ef1pct"],
                     actives.height, inactives.height))
        idn = diag.ligand_identity_memorization(pool, train_mask=train_mask, eval_mask=eval_mask)
        rows.append((target, "identity", "split_80_20_actives", idn["auroc"], idn["ap"], idn["ef1pct"],
                     actives.height, inactives.height))
        scf = diag.scaffold_memorization(pool, train_mask=train_mask, eval_mask=eval_mask)
        rows.append((target, "scaffold", "split_80_20_actives", scf["auroc"], scf["ap"], scf["ef1pct"],
                     actives.height, inactives.height))

    out = pl.DataFrame(rows, schema=[
        "target", "diagnostic", "protocol", "auroc", "ap", "ef1pct",
        "n_actives", "n_inactives_in_pool"], orient="row")
    out.write_csv(TABLES / "litpcba_shortcut_results.csv")

    (REPORTS / "litpcba_shortcut_diagnostics.md").write_text(
        "# LIT-PCBA — shortcut diagnostics (MVP)\n\n"
        "Train/val/test splits are NOT in the `full_data.tgz` archive. As a\n"
        "stand-in we use a deterministic random 80/20 split over each target's\n"
        "actives. All inactives (sampled to a cap) go into the eval pool. The\n"
        "eval pool is the 20% held-out actives + the (sampled) inactives.\n\n"
        "## Diagnostics\n"
        "- `ligand_knn` → score = max ECFP4 Tanimoto to a training-set active.\n"
        "- `identity`   → score = 1 if canonical SMILES seen as a train active.\n"
        "- `scaffold`   → score = 1 if Bemis-Murcko scaffold seen as a train active.\n\n"
        "Strong AUROC / EF1% here indicates that trivial memorization-style\n"
        "lookups already separate held-out actives from inactives — a\n"
        "memorization-prone benchmark. This is the same baseline the published\n"
        "LIT-PCBA audit uses.\n\n"
        f"Inactives cap per target: **{cfg.inactive_cap_per_target}** "
        f"(sampled with seed {cfg.rng_seed}).\n\n"
        "Results saved to `outputs/tables/litpcba_shortcut_results.csv`.\n",
        encoding="utf-8")
    log_disk("post_step", "task_5_litpcba_diagnostics")


def task_6_dude_load_and_graph(cfg: RunConfig) -> Optional[pl.DataFrame]:
    log_disk("pre_step", "task_6_dude_load")
    cached = PROCESSED / "dude_examples.parquet"
    if cached.exists():
        log.info("DUD-E: loading cached featurized parquet %s", cached)
        df = pl.read_parquet(cached)
        log.info("DUD-E cached examples: %d", df.height)
    else:
        try:
            df = load_dude.load_all(RAW / "DUD-E")
        except Exception as exc:
            log.warning("DUD-E load failed: %s", exc)
            (REPORTS / "missing_dude.md").write_text(
                f"# DUD-E load failed\n\n{exc}\n", encoding="utf-8")
            return None
        log.info("DUD-E examples loaded: %d", df.height)
        df = _featurize_examples(df, cfg, "DUD-E")
        df.write_parquet(cached)

    counts = {
        "n_examples": df.height,
        "n_targets": df.select("target").n_unique(),
        "n_unique_canonical": df.filter(pl.col("smiles_canonical").is_not_null())
                                .select("smiles_canonical").n_unique(),
        "n_unique_inchikey": df.filter(pl.col("inchikey").is_not_null())
                               .select("inchikey").n_unique(),
        "n_unique_scaffold": df.filter(pl.col("scaffold_smiles").is_not_null())
                               .select("scaffold_smiles").n_unique(),
        "n_actives": int((df["label"] == 1).sum()),
        "n_decoys": int((df["label"] == 0).sum()),
        "parse_ok": int(df["parse_ok"].sum()),
        "parse_fail": int((~df["parse_ok"]).sum()),
    }
    (REPORTS / "dude_dataset_summary.md").write_text(
        f"# DUD-E — dataset summary\n\n"
        f"Generated by `vsleakkg.run_mvp_audit` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in counts.items())
        + "\n\nAll examples carry `source = 'DUD-E'`, `label_type ∈ {active, decoy}`,\n"
        "and a `target` derived from the per-target folder name. There is no\n"
        "train/val/test split inside DUD-E — `split` is `unknown`.\n",
        encoding="utf-8")

    df = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        df, include_decoy_protocol=True, include_protein_target=True,
    )
    nodes.write_parquet(PROCESSED / "dude_nodes.parquet")
    edges.write_parquet(PROCESSED / "dude_edges.parquet")
    (REPORTS / "dude_graph_summary.md").write_text(
        _render_graph_summary("DUD-E", nodes, edges), encoding="utf-8")
    log_disk("post_step", "task_6_dude_load")
    return df


def task_7_dude_diagnostics(df: pl.DataFrame, cfg: RunConfig) -> None:
    log_disk("pre_step", "task_7_dude_diagnostics")
    rng = np.random.default_rng(cfg.rng_seed + 1)
    rows = []
    for target in sorted(df.select("target").drop_nulls().unique().to_series().to_list()):
        sub = df.filter(pl.col("target") == target)
        actives = sub.filter(pl.col("label") == 1)
        decoys = sub.filter(pl.col("label") == 0)
        if actives.height < 5 or decoys.height < 5:
            rows.append((target, "skipped", "too_few_examples", None, None, None,
                         actives.height, decoys.height))
            continue
        if decoys.height > cfg.decoy_cap_per_target:
            idx = rng.choice(decoys.height, size=cfg.decoy_cap_per_target, replace=False)
            decoys = decoys[sorted(idx.tolist())]
        pool = pl.concat([actives, decoys], how="vertical_relaxed").with_row_count("_pool_idx")
        pool = pool.with_columns(pl.Series("_fp", serial_ecfp(pool["smiles_canonical"].to_list()),
                                           dtype=pl.Object))
        # Random 80/20 split on actives only (deterministic per seed); decoys all go into eval pool.
        actives_idx_in_pool = pool.filter(pl.col("label") == 1)["_pool_idx"].to_numpy()
        perm = rng.permutation(actives_idx_in_pool)
        n_train = max(1, int(0.8 * len(perm)))
        train_actives_pool_idx = perm[:n_train].tolist()
        train_mask = pl.col("_pool_idx").is_in(train_actives_pool_idx) & (pl.col("label") == 1)
        eval_mask = ~pl.col("_pool_idx").is_in(train_actives_pool_idx)

        knn = diag.ligand_knn_max_tanimoto(pool, train_mask=train_mask, eval_mask=eval_mask)
        rows.append((target, "ligand_knn", "split_80_20_actives", knn["auroc"], knn["ap"], knn["ef1pct"],
                     actives.height, decoys.height))
        idn = diag.ligand_identity_memorization(pool, train_mask=train_mask, eval_mask=eval_mask)
        rows.append((target, "identity", "split_80_20_actives", idn["auroc"], idn["ap"], idn["ef1pct"],
                     actives.height, decoys.height))
        scf = diag.scaffold_memorization(pool, train_mask=train_mask, eval_mask=eval_mask)
        rows.append((target, "scaffold", "split_80_20_actives", scf["auroc"], scf["ap"], scf["ef1pct"],
                     actives.height, decoys.height))

    out = pl.DataFrame(rows, schema=[
        "target", "diagnostic", "protocol", "auroc", "ap", "ef1pct",
        "n_actives", "n_decoys_in_pool"], orient="row")
    out.write_csv(TABLES / "dude_shortcut_results.csv")
    (REPORTS / "dude_shortcut_diagnostics.md").write_text(
        "# DUD-E — shortcut diagnostics (MVP)\n\n"
        "DUD-E has no built-in train/val/test split. We use a deterministic\n"
        "80/20 random split over each target's actives; decoys go entirely into\n"
        "the eval pool. We report ligand KNN (max ECFP4 Tanimoto to held-in\n"
        "actives), identity memorization, and scaffold memorization per target.\n\n"
        f"Decoys cap per target: **{cfg.decoy_cap_per_target}** "
        f"(sampled with seed {cfg.rng_seed+1}).\n\n"
        "Note: a decoy-protocol-only diagnostic is **not meaningful within DUD-E\n"
        "alone** because every decoy shares the same property-matched protocol.\n"
        "This becomes meaningful in phase 2 when DEKOIS / LIT-PCBA inactives /\n"
        "BayesBind decoys are added — see `outputs/reports/mvp_audit_report.md`.\n",
        encoding="utf-8")
    log_disk("post_step", "task_7_dude_diagnostics")


def task_8_combined_graph() -> None:
    log_disk("pre_step", "task_8_combined_graph")
    have_lit = (PROCESSED / "litpcba_nodes.parquet").exists()
    have_dud = (PROCESSED / "dude_nodes.parquet").exists()
    if not (have_lit or have_dud):
        log.warning("task 8: no per-dataset graphs to combine")
        return
    nodes_parts = []
    edges_parts = []
    if have_lit:
        nodes_parts.append(pl.read_parquet(PROCESSED / "litpcba_nodes.parquet"))
        edges_parts.append(pl.read_parquet(PROCESSED / "litpcba_edges.parquet"))
    if have_dud:
        nodes_parts.append(pl.read_parquet(PROCESSED / "dude_nodes.parquet"))
        edges_parts.append(pl.read_parquet(PROCESSED / "dude_edges.parquet"))
    nodes = pl.concat(nodes_parts, how="vertical_relaxed").unique(subset=["node_id"])
    edges = pl.concat(edges_parts, how="vertical_relaxed").unique()
    nodes.write_parquet(PROCESSED / "mvp_nodes.parquet")
    edges.write_parquet(PROCESSED / "mvp_edges.parquet")
    (REPORTS / "mvp_graph_summary.md").write_text(
        _render_graph_summary("MVP combined (LIT-PCBA + DUD-E)", nodes, edges),
        encoding="utf-8")
    log_disk("post_step", "task_8_combined_graph")


def task_9_contamination_score(df_lit: Optional[pl.DataFrame],
                               df_dud: Optional[pl.DataFrame], cfg: RunConfig) -> None:
    """For MVP: per-target, treat actives as the 'train' set and the rest of
    the pool (inactives/decoys, sampled) as eval examples; compute the simple
    component-wise score. This is *not* the same as train/test contamination
    in a real benchmark, but it documents how the score will be wired once
    splits are available."""
    log_disk("pre_step", "task_9_contamination_score")
    parts = []
    rng = np.random.default_rng(cfg.rng_seed + 2)
    for label, df, eval_cap in (
        ("LIT-PCBA", df_lit, cfg.inactive_cap_per_target),
        ("DUD-E",    df_dud, cfg.decoy_cap_per_target),
    ):
        if df is None or df.is_empty():
            continue
        targets = sorted(df.select("target").drop_nulls().unique().to_series().to_list())
        for target in targets:
            sub = df.filter(pl.col("target") == target)
            actives = sub.filter(pl.col("label") == 1)
            evalset = sub.filter(pl.col("label") == 0)
            if actives.is_empty() or evalset.is_empty():
                continue
            if evalset.height > eval_cap:
                idx = rng.choice(evalset.height, size=eval_cap, replace=False)
                evalset = evalset[sorted(idx.tolist())]
            pool = pl.concat([actives, evalset], how="vertical_relaxed").with_row_count("_pool_idx")
            pool = pool.with_columns(pl.Series("_fp", serial_ecfp(pool["smiles_canonical"].to_list()),
                                               dtype=pl.Object))
            train_mask = (pl.col("label") == 1)
            eval_mask = (pl.col("label") == 0)
            scored = cscore.compute_scores(pool, train_mask=train_mask, eval_mask=eval_mask)
            if scored.height == 0:
                continue
            scored = scored.with_columns([
                pl.lit(label).alias("_dataset"),
                pl.lit(target).alias("_target"),
            ])
            parts.append(scored.select([
                "_dataset", "_target", "example_id" if "example_id" in scored.columns else "smiles_canonical",
                "c_identity", "c_scaffold", "c_analog", "c_source", "c_assay", "c_total",
            ]) if "example_id" in scored.columns else scored.select([
                "_dataset", "_target", "smiles_canonical",
                "c_identity", "c_scaffold", "c_analog", "c_source", "c_assay", "c_total",
            ]))
    if not parts:
        log.warning("task 9: nothing to score")
        return
    scored_all = pl.concat(parts, how="diagonal_relaxed")
    scored_all.write_parquet(PROCESSED / "mvp_contamination_scores.parquet")

    summary = scored_all.group_by("_dataset").agg([
        pl.len().alias("n_eval"),
        pl.col("c_identity").mean().alias("c_identity_mean"),
        pl.col("c_scaffold").mean().alias("c_scaffold_mean"),
        pl.col("c_analog").drop_nans().mean().alias("c_analog_mean"),
        pl.col("c_total").drop_nans().mean().alias("c_total_mean"),
    ]).sort("_dataset")
    summary.write_csv(TABLES / "mvp_contamination_score_summary.csv")

    (REPORTS / "mvp_contamination_score_summary.md").write_text(
        "# MVP contamination score — summary\n\n"
        "Per-example heuristic score in [0, 1] aggregating identity, scaffold,\n"
        "analog (max ECFP4 Tanimoto), source-share, and assay-share components.\n"
        "Because no train/test split is available for either LIT-PCBA or DUD-E,\n"
        "we use **actives as the synthetic 'train' set and inactives/decoys as\n"
        "the synthetic 'eval' set** per target. The scores quantify how easily\n"
        "the eval set can be predicted from the train set by trivial lookup —\n"
        "they are *not* a substitute for a proper held-out evaluation.\n\n"
        "## Dataset-level means\n\n"
        + _polars_to_md_table(summary) + "\n\n"
        "Full per-example scores: `data/processed/mvp_contamination_scores.parquet`.\n"
        "Components present: c_identity, c_scaffold, c_analog, c_source. The\n"
        "c_assay column is currently NaN — assay metadata is not yet attached.\n",
        encoding="utf-8")
    log_disk("post_step", "task_9_contamination_score")


def task_10_final_report(state: dict, avail: dict, cfg: RunConfig) -> None:
    log_disk("pre_step", "task_10_final_report")
    have_lit = (PROCESSED / "litpcba_examples.parquet").exists()
    have_dud = (PROCESSED / "dude_examples.parquet").exists()

    def _read_md(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else "(missing)"

    body = []
    body.append(f"# VS-LeakKG MVP audit report\n\nGenerated {datetime.now(timezone.utc).isoformat()}.\n")
    body.append("## 1. Repos cloned\n\nSee `outputs/setup_report.md` §1 for commit hashes:\n"
                "LIT-PCBA-audit, chembl-downloader, plinder.\n")
    body.append("## 2. Datasets available\n\n```\n" + json.dumps(avail, indent=2, default=str) + "\n```\n")
    body.append("## 3. Manual downloads required\n\nSee `data/raw/manual_downloads_needed/`:\n"
                "- BindingDB_TODO.md\n- LIT-PCBA_TODO.md (full_data present; split files still needed)\n"
                "- BayesBind_BigBind_TODO.md\n- PLINDER_TODO.md\n")
    body.append("## 4. Disk usage\n\n"
                "Pre/post snapshots in `outputs/logs/mvp_audit_disk_usage.log`. Setup\n"
                "consumed ~4.9 GB on D: (mostly chembl_35_sqlite.tar.gz).\n")
    body.append("## 5. LIT-PCBA findings\n\n" + _read_md(REPORTS / "litpcba_dataset_summary.md") + "\n")
    body.append("### Identity & scaffold leakage tables\n"
                "- `outputs/tables/litpcba_identity_leakage_by_target.csv`\n"
                "- `outputs/tables/litpcba_identity_leakage_by_label.csv`\n"
                "- `outputs/tables/litpcba_identity_intra_duplicates.csv`\n"
                "- `outputs/tables/litpcba_scaffold_overlap.csv`\n"
                "- `outputs/tables/litpcba_analog_overlap.csv`\n")
    body.append("### Shortcut diagnostics\n\n" + _read_md(REPORTS / "litpcba_shortcut_diagnostics.md") + "\n")
    body.append("## 6. DUD-E findings\n\n" + _read_md(REPORTS / "dude_dataset_summary.md") + "\n")
    body.append("### Shortcut diagnostics\n\n" + _read_md(REPORTS / "dude_shortcut_diagnostics.md") + "\n")
    body.append("## 7. Graph summaries\n\n"
                + _read_md(REPORTS / "litpcba_graph_summary.md") + "\n\n"
                + _read_md(REPORTS / "dude_graph_summary.md") + "\n\n"
                + _read_md(REPORTS / "mvp_graph_summary.md") + "\n")
    body.append("## 8. Contamination score\n\n"
                + _read_md(REPORTS / "mvp_contamination_score_summary.md") + "\n")
    body.append("## 9. Limitations\n\n"
                "- Train/val/test splits not in LIT-PCBA `full_data.tgz` — diagnostics use\n"
                "  a deterministic 80/20 random split over each target's actives, with\n"
                "  inactives sampled into the eval pool. This is a stand-in, NOT the\n"
                "  benchmark's published split.\n"
                "- DUD-E has no internal splits — same 80/20 over actives only.\n"
                "- Analog audits sample inactives/decoys (caps documented).\n"
                "- Assay metadata not attached — c_assay is NaN; needs ChEMBL/PubChem join.\n"
                "- Decoy-protocol shortcut diagnostic is single-protocol in DUD-E and\n"
                "  becomes meaningful only across DUD-E vs DEKOIS vs LIT-PCBA.\n"
                "- BindingDB, BayesBind/BigBind, PLINDER, PDBBind not in this audit.\n")
    body.append("## 10. Next steps\n\n"
                "1. Download LIT-PCBA train/val/test split files from Unistra (see TODO),\n"
                "   re-run task 3+5 with real splits.\n"
                "2. Join ChEMBL assay + document metadata so c_assay becomes real.\n"
                "3. Pull BindingDB TSV (manual) and cross-source map by InChIKey.\n"
                "4. Add DEKOIS so decoy-protocol shortcuts become a 3-way comparison.\n"
                "5. Compare against PLINDER / DataSAIL-style similarity-only splits.\n"
                "6. Replace the heuristic contamination score with path-based KG\n"
                "   features (e.g., shortest-path leakage through shared scaffold or\n"
                "   shared assay nodes) once the graph is richer.\n")
    (REPORTS / "mvp_audit_report.md").write_text("\n".join(body), encoding="utf-8")
    log_disk("post_step", "task_10_final_report")


# -------------------- main --------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--inactive-cap", type=int, default=5000)
    parser.add_argument("--decoy-cap", type=int, default=2000)
    parser.add_argument("--skip-litpcba", action="store_true")
    parser.add_argument("--skip-dude", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(
        workers=args.workers,
        inactive_cap_per_target=args.inactive_cap,
        decoy_cap_per_target=args.decoy_cap,
        skip_litpcba=args.skip_litpcba,
        skip_dude=args.skip_dude,
    )

    log_disk("setup_start", "mvp_audit")
    state = task_1_read_setup_state()
    avail = task_2_check_datasets()

    df_lit = None
    if not cfg.skip_litpcba and avail["LIT-PCBA"]["non_empty"]:
        df_lit = task_3_litpcba_sanity(cfg)
        if df_lit is not None and not cfg.skip_graph:
            task_4_litpcba_graph(df_lit)
            task_5_litpcba_diagnostics(df_lit, cfg)
    else:
        log.warning("LIT-PCBA skipped (skip flag or empty)")

    df_dud = None
    if not cfg.skip_dude and avail["DUD-E"]["non_empty"]:
        df_dud = task_6_dude_load_and_graph(cfg)
        if df_dud is not None:
            task_7_dude_diagnostics(df_dud, cfg)
    else:
        log.warning("DUD-E skipped (skip flag or empty)")

    task_8_combined_graph()
    task_9_contamination_score(df_lit, df_dud, cfg)
    task_10_final_report(state, avail, cfg)
    log_disk("setup_end", "mvp_audit")

    print()
    if avail["LIT-PCBA"]["non_empty"] and avail["DUD-E"]["non_empty"]:
        print("MVP setup and audit complete. See:")
        print(" - outputs/setup_report.md")
        print(" - data/MANIFEST.md")
        print(" - outputs/reports/mvp_audit_report.md")
    else:
        print("MVP partially complete. Manual downloads are required. See:")
        print(" - data/raw/manual_downloads_needed/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
