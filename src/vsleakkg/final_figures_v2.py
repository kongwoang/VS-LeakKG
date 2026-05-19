"""Add new figures for the full-completion pass (figs 7–10).

Existing figures (1–6) are produced by `final_figures.py` from the prior
pass and are not regenerated here unless their input tables changed.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables" / "final"
OUTDIR = ROOT / "outputs" / "reports" / "figures" / "final"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(OUTDIR / f"{name}.png", dpi=160, bbox_inches="tight")
    try:
        fig.savefig(OUTDIR / f"{name}.pdf", bbox_inches="tight")
    except Exception:
        pass


def fig7(plt):
    df = pl.read_csv(TABLES / "table7_weighted_contamination_by_target.csv").sort("mean_strict")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(df.height))
    width = 0.4
    ax.bar([i - width/2 for i in x], df["mean_strict"].to_list(), width,
           label="C_strict (weighted)", color="C0", alpha=0.8)
    ax.bar([i + width/2 for i in x], df["mean_candidate"].to_list(), width,
           label="C_candidate (weighted)", color="C1", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["target"].to_list(), rotation=45, ha="right")
    ax.set_ylabel("mean weighted contamination score")
    ax.set_title("Weighted path-product contamination — LIT-PCBA AVE val")
    ax.legend()
    _save(fig, "fig7_weighted_contamination_by_target"); plt.close(fig)


def fig8(plt):
    df = (pl.read_csv(TABLES / "table8_contamination_nn_metrics.csv")
            .filter(pl.col("target") != "ALL").sort("auroc"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(df.height))
    ax.bar(x, df["auroc"].to_list(), color="C2", alpha=0.8)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(df["target"].to_list(), rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Contamination-NN baseline AUROC by target (LIT-PCBA AVE)")
    ax.set_ylim(0.45, 0.9)
    _save(fig, "fig8_contamination_nn_by_target"); plt.close(fig)


def fig9(plt):
    df = pl.read_csv(TABLES / "table14_original_vs_generated_split_diagnostics.csv")
    df = df.filter(pl.col("scaffold_knn_auroc").is_not_nan())
    # Group by dataset, show original vs generated scaffold_knn AUROC
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = (df["dataset"] + "\n" + df["split"]).to_list()
    x = list(range(df.height))
    y = df["scaffold_knn_auroc"].to_list()
    colors = ["C3" if "original" in s or s == "original_AVE" else "C2" for s in df["split"]]
    ax.bar(x, y, color=colors, alpha=0.8)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("scaffold_knn AUROC")
    ax.set_ylim(0.45, 0.7)
    ax.set_title("Scaffold-cold splits collapse the scaffold-memorisation shortcut")
    _save(fig, "fig9_original_vs_generated_splits"); plt.close(fig)


def fig10(plt):
    # Proposal coverage bar — counts of items built per section
    sections = [
        ("§3.2 nodes/edges", 12, 12),
        ("§3.3 path scoring", 4, 4),
        ("§3.4 cold splits", 5, 10),
        ("§3.5 diagnostics", 3, 4),
        ("§3.6 metrics", 6, 9),
        ("§3.7 training (optional)", 0, 1),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(sections)))
    built = [b for _, b, _ in sections]
    total = [t for _, _, t in sections]
    ax.bar(x, total, color="lightgrey", label="proposal total")
    ax.bar(x, built, color="C2", label="built this pass")
    ax.set_xticks(x); ax.set_xticklabels([s for s, *_ in sections], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("items")
    ax.set_title("Proposal coverage after full-completion pass (30/40 built)")
    ax.legend()
    for i, (b, t) in enumerate(zip(built, total)):
        ax.text(i, t + 0.2, f"{b}/{t}", ha="center", fontsize=8)
    _save(fig, "fig10_proposal_coverage"); plt.close(fig)


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] matplotlib unavailable: {e}")
        return
    for fn in (fig7, fig8, fig9, fig10):
        try:
            fn(plt)
            print(f"[fig] ok: {fn.__name__}")
        except Exception as e:
            print(f"[fig] FAILED {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
