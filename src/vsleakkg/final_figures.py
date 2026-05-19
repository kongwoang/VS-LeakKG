"""Final figures for VS-LeakKG experiment summary.

Each figure tolerates missing inputs and writes both PNG and PDF when
matplotlib is available. Failures are logged but do not abort the run.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables" / "final"
OUTDIR = ROOT / "outputs" / "reports" / "figures" / "final"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(OUTDIR / f"{name}.png", dpi=160, bbox_inches="tight")
    try:
        fig.savefig(OUTDIR / f"{name}.pdf", bbox_inches="tight")
    except Exception:
        pass


def fig1_graph_growth(plt):
    fig, ax = plt.subplots(figsize=(6, 4))
    stages = ["MVP-0", "MVP-1", "MVP-1+PDBBind", "MVP-2", "MVP-2+ProtClusters"]
    nodes = [55, 4_181_664 + 1_669_148, 4_181_664 + 1_669_148 + 19037 + 11862,
             11_325_757, 11_325_757 + 35_571]
    edges = [50, 4_181_664 * 4, 4_181_664 * 4 + 76_148,
             41_865_600, 41_865_600 + 35_586]
    ax2 = ax.twinx()
    x = list(range(len(stages)))
    l1 = ax.plot(x, nodes, "o-", color="C0", label="nodes")
    l2 = ax2.plot(x, edges, "s--", color="C3", label="edges")
    ax.set_xticks(x); ax.set_xticklabels(stages, rotation=20)
    ax.set_yscale("log"); ax2.set_yscale("log")
    ax.set_ylabel("nodes (log)", color="C0"); ax2.set_ylabel("edges (log)", color="C3")
    ax.set_title("Graph growth: MVP-0 → MVP-2 + protein clusters")
    fig.tight_layout()
    _save(fig, "fig1_graph_growth")
    plt.close(fig)


def fig2_ligand_knn_by_dataset(plt):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = {}
    # LIT-PCBA: kg_nn_best from table5
    try:
        t5 = pl.read_csv(TABLES / "table5_similarity_vs_provenance.csv")
        data["LIT-PCBA AVE\n(scaffold_kg)"] = t5["kg_nn_auroc_best"].to_list()
    except Exception:
        pass
    # DUD-E ligand_knn
    try:
        d = pl.read_csv(ROOT / "outputs" / "tables" / "dude_shortcut_results.csv")
        v = d.filter(pl.col("diagnostic") == "ligand_knn")["auroc"].to_list()
        if v: data["DUD-E\nligand_knn"] = v
    except Exception:
        pass
    try:
        d = pl.read_csv(ROOT / "outputs" / "tables" / "dekois_shortcut_results.csv")
        v = d.filter(pl.col("diagnostic") == "ligand_knn")["auroc"].to_list()
        if v: data["DEKOIS\nligand_knn"] = v
    except Exception:
        pass
    try:
        bb = pl.read_csv(TABLES / "bayesbind_shortcut_results.csv")
        v = bb.filter(pl.col("diagnostic") == "pdbbind_overlap")["auroc"].to_list()
        if v: data["BayesBind\npdbbind_overlap"] = v
    except Exception:
        pass
    if not data:
        plt.close(fig)
        return
    ax.boxplot(list(data.values()), labels=list(data.keys()), showmeans=True)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.set_ylabel("AUROC per target")
    ax.set_title("Ligand-shortcut AUROC by benchmark")
    fig.tight_layout()
    _save(fig, "fig2_ligand_knn_by_dataset")
    plt.close(fig)


def fig3_litpcba_path_vs_shortcut(plt):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        t5 = pl.read_csv(TABLES / "table5_similarity_vs_provenance.csv").sort("similarity_only_score")
    except Exception:
        plt.close(fig); return
    x = t5["target"].to_list()
    sim = t5["similarity_only_score"].to_list()
    prov = t5["provenance_score_candidate"].to_list()
    auroc = t5["kg_nn_auroc_best"].to_list()
    idx = list(range(len(x)))
    width = 0.35
    ax.bar([i - width/2 for i in idx], sim, width, label="similarity-only reachability", color="C0", alpha=0.8)
    ax.bar([i + width/2 for i in idx], prov, width, label="provenance reachability", color="C1", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(idx, auroc, "ks-", label="best KG-NN AUROC")
    ax2.axhline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax2.set_ylim(0.4, 0.7); ax2.set_ylabel("KG-NN AUROC")
    ax.set_ylim(0, 1.05); ax.set_ylabel("reachability rate (val→train)")
    ax.set_xticks(idx); ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_title("LIT-PCBA AVE: high graph reachability, near-chance predictive shortcut")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig3_litpcba_path_vs_shortcut")
    plt.close(fig)


def fig4_mapping_rates(plt):
    try:
        t3 = pl.read_csv(TABLES / "table3_mapping_provenance_coverage.csv")
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ds = t3["dataset"].to_list()
    ch = t3["chembl_rate"].to_list()
    bb = t3["bindingdb_rate"].to_list()
    idx = list(range(len(ds)))
    w = 0.35
    ax.bar([i - w/2 for i in idx], ch, w, label="ChEMBL mapped", color="C2")
    ax.bar([i + w/2 for i in idx], bb, w, label="BindingDB mapped", color="C4")
    ax.set_xticks(idx); ax.set_xticklabels(ds, rotation=15)
    ax.set_ylabel("fraction of unique ligands mapped"); ax.set_ylim(0, 1.05)
    ax.set_title("ChEMBL / BindingDB exact-InChIKey mapping rates")
    ax.legend()
    fig.tight_layout(); _save(fig, "fig4_mapping_rates"); plt.close(fig)


def fig5_similarity_only_vs_provenance(plt):
    try:
        t5 = pl.read_csv(TABLES / "table5_similarity_vs_provenance.csv")
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t5["similarity_only_score"], t5["provenance_score_candidate"], s=60, c="C0")
    for r in t5.iter_rows(named=True):
        ax.annotate(r["target"], (r["similarity_only_score"], r["provenance_score_candidate"]),
                    fontsize=8, alpha=0.7)
    lo = min(t5["similarity_only_score"].min(), t5["provenance_score_candidate"].min()) - 0.02
    hi = max(t5["similarity_only_score"].max(), t5["provenance_score_candidate"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("similarity-only score (identity/scaffold/analog)")
    ax.set_ylabel("provenance-enriched score (+ ChEMBL/BindingDB/PDBBind/assay/doc)")
    ax.set_title("Provenance paths beyond similarity-only (LIT-PCBA AVE)")
    fig.tight_layout(); _save(fig, "fig5_similarity_only_vs_provenance"); plt.close(fig)


def fig6_pdbbind_protein_clusters(plt):
    try:
        t6 = pl.read_csv(TABLES / "table6_pdbbind_protein_clustering.csv")
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    th = t6["threshold"].to_list()
    nc = t6["n_clusters"].to_list()
    ns = t6["non_singleton_complexes"].to_list()
    idx = list(range(len(th)))
    w = 0.35
    ax2 = ax.twinx()
    ax.bar([i - w/2 for i in idx], nc, w, color="C0", label="n_clusters")
    ax2.bar([i + w/2 for i in idx], ns, w, color="C3", label="non-singleton complexes")
    ax.set_xticks(idx); ax.set_xticklabels(th)
    ax.set_ylabel("n_clusters", color="C0"); ax2.set_ylabel("non-singleton complexes", color="C3")
    ax.set_title("PDBBind protein clusters (exact-sequence fallback; mmseqs deferred)")
    fig.tight_layout(); _save(fig, "fig6_pdbbind_protein_clusters"); plt.close(fig)


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figures] matplotlib unavailable: {e}")
        return
    for fn in (fig1_graph_growth, fig2_ligand_knn_by_dataset, fig3_litpcba_path_vs_shortcut,
               fig4_mapping_rates, fig5_similarity_only_vs_provenance,
               fig6_pdbbind_protein_clusters):
        try:
            fn(plt)
            print(f"[figures] ok: {fn.__name__}")
        except Exception as e:
            print(f"[figures] FAILED {fn.__name__}: {e}")
    print(f"[figures] done. wrote to {OUTDIR}")


if __name__ == "__main__":
    main()
