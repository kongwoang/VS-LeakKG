"""PocketCluster MVP — composition-based pocket features + clustering.

Foldseek is not installed and we are not allowed to download external
structure databases. So this is a *composition* MVP, clearly labelled as
PocketCluster-MVP and not full structural pocket similarity.

Per pocket PDB file, we extract:
  - 20-D amino-acid composition vector (normalised counts)
  - n_residues
  - n_atoms
  - n_chains
  - net_charge (Lys + Arg - Asp - Glu, residue-level)
  - hydrophobic_fraction (Ala/Val/Leu/Ile/Phe/Met/Trp)
  - aromatic_fraction (Phe/Tyr/Trp)
  - polar_fraction (Ser/Thr/Asn/Gln/Cys/Tyr)

Then we cluster at 3 levels using KMeans on the 20-D composition vector:
  - coarse: 50 clusters
  - mid:    200 clusters
  - fine:   800 clusters

Each PDBBind complex maps to one pocket → three cluster ids.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW_PDB = ROOT / "data" / "raw" / "PBDBind" / "extracted" / "P-L"

AA_3 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA_INDEX = {a: i for i, a in enumerate(AA_3)}


def extract_features(pdb_path: Path) -> Optional[dict]:
    """Parse a pocket PDB file. Returns None on parse failure."""
    counts = np.zeros(20, dtype=np.int32)
    n_atoms = 0
    chains = set()
    residues = set()
    try:
        with pdb_path.open() as fh:
            for line in fh:
                if not (line.startswith("ATOM ") or line.startswith("ATOM\t")):
                    continue
                # Standard PDB columns: 13-16 atom name, 18-20 resname, 22 chain, 23-26 resnum
                resname = line[17:20].strip()
                chain = line[21:22]
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                rkey = (chain, resnum)
                if rkey not in residues:
                    residues.add(rkey)
                    if resname in AA_INDEX:
                        counts[AA_INDEX[resname]] += 1
                chains.add(chain)
                n_atoms += 1
    except Exception:
        return None
    n_res = int(counts.sum())
    if n_res == 0:
        return None
    comp = counts / n_res
    hydrophobic = comp[[AA_INDEX[x] for x in ("ALA","VAL","LEU","ILE","PHE","MET","TRP")]].sum()
    aromatic = comp[[AA_INDEX[x] for x in ("PHE","TYR","TRP")]].sum()
    polar = comp[[AA_INDEX[x] for x in ("SER","THR","ASN","GLN","CYS","TYR")]].sum()
    net_charge = (counts[AA_INDEX["LYS"]] + counts[AA_INDEX["ARG"]]
                  - counts[AA_INDEX["ASP"]] - counts[AA_INDEX["GLU"]])
    return {
        "pdb_id": pdb_path.stem.replace("_pocket", ""),
        **{f"aa_{a.lower()}": float(comp[i]) for i, a in enumerate(AA_3)},
        "n_residues": n_res,
        "n_atoms": int(n_atoms),
        "n_chains": int(len(chains)),
        "net_charge": int(net_charge),
        "hydrophobic_fraction": float(hydrophobic),
        "aromatic_fraction": float(aromatic),
        "polar_fraction": float(polar),
    }


def collect_features(parallel: bool = True) -> pl.DataFrame:
    paths = sorted(RAW_PDB.glob("*/*/*_pocket.pdb"))
    print(f"[pocket] found {len(paths)} pocket PDB files")
    rows: list[dict] = []
    if parallel:
        with ProcessPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(extract_features, p) for p in paths]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                if r is not None:
                    rows.append(r)
                if i % 2000 == 0:
                    print(f"[pocket]   parsed {i}/{len(paths)}")
    else:
        for i, p in enumerate(paths, 1):
            r = extract_features(p)
            if r is not None:
                rows.append(r)
            if i % 2000 == 0:
                print(f"[pocket]   parsed {i}/{len(paths)}")
    print(f"[pocket] features parsed: {len(rows)}")
    return pl.DataFrame(rows)


def cluster_features(df: pl.DataFrame) -> pl.DataFrame:
    from sklearn.cluster import KMeans

    aa_cols = [f"aa_{a.lower()}" for a in AA_3]
    X = df.select(aa_cols).to_numpy()
    print(f"[pocket] composition matrix: {X.shape}")

    out_cols = []
    for k, label in ((50, "coarse"), (200, "mid"), (800, "fine")):
        if X.shape[0] < k:
            k = max(2, X.shape[0] // 2)
        km = KMeans(n_clusters=k, random_state=17, n_init="auto")
        labels = km.fit_predict(X)
        col = f"cluster_{label}"
        df = df.with_columns(pl.Series(col, labels))
        out_cols.append(col)
        print(f"[pocket]   {label}: k={k}  inertia={km.inertia_:.2f}")
    return df


def build_graph_extension(df: pl.DataFrame) -> None:
    nodes = []
    edges = []
    for label, level in (("coarse", "PocketCluster_coarse"),
                        ("mid", "PocketCluster_mid"),
                        ("fine", "PocketCluster_fine")):
        col = f"cluster_{label}"
        uniq = df.select(pl.col(col)).unique()
        for v in uniq[col].to_list():
            nodes.append({
                "node_id": f"pktclu_{label}:{v}",
                "node_type": level,
                "cluster_level": label,
            })
        for row in df.iter_rows(named=True):
            edges.append({
                "src": f"pdbbind_pocket:{row['pdb_id']}",
                "dst": f"pktclu_{label}:{row[col]}",
                "edge_type": f"pocket_in_cluster_{label}",
            })
    pl.DataFrame(nodes).write_parquet(PROCESSED / "mvp2_plus_pocket_clusters_nodes.parquet")
    pl.DataFrame(edges).write_parquet(PROCESSED / "mvp2_plus_pocket_clusters_edges.parquet")
    print(f"[pocket] graph extension: nodes={len(nodes)} edges={len(edges)}")


def main():
    feat_out = PROCESSED / "pocket_features.parquet"
    if feat_out.exists():
        df = pl.read_parquet(feat_out)
        print(f"[pocket] reusing existing pocket_features ({df.height})")
    else:
        df = collect_features()
        df.write_parquet(feat_out)
        print(f"[pocket] wrote {feat_out}")
    df = cluster_features(df)
    df.write_parquet(PROCESSED / "pocket_clusters.parquet")
    print(f"[pocket] wrote pocket_clusters.parquet")
    build_graph_extension(df)


if __name__ == "__main__":
    main()
