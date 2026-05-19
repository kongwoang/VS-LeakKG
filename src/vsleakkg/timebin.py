"""TimeBin node-type builder.

Time information available:
- ChEMBL documents: `year` (1956..2024 for ChEMBL 35).
- PDBBind complexes: `release_year` (1982..2020).
- BindingDB: has PMID/DOI but no year column; skipped.
- Dataset releases: hard-coded versions.

Bins:
- per year for any artifact with an exact year
- per decade otherwise
- one node per dataset release for the release-only artifacts

Edges:
- document_in_timebin
- complex_in_timebin
- dataset_release_in_timebin
- activity_in_timebin  (transitive: activity -> document -> timebin year)
- example_in_timebin    (only PDBBind examples — they have a release_year)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


RELEASES = [
    ("ChEMBL35",        2024),
    ("BindingDB202605", 2026),
    ("PDBBind2020",     2020),
    ("LIT-PCBA2020",    2020),
    ("DUDE2012",        2012),
    ("DEKOIS2.0_2013",  2013),
    ("BayesBindV1.5",   2024),
    ("BigBindV1.5",     2024),
]


def _year_bin(year: int) -> str:
    return f"tbin_y{int(year)}"


def main() -> None:
    docs = pl.read_parquet(PROCESSED / "chembl_documents.parquet")
    docs_yr = docs.filter(pl.col("year").is_not_null())
    print(f"[timebin] ChEMBL docs with year: {docs_yr.height}/{docs.height}")

    pdb = pl.read_parquet(PROCESSED / "pdbbind_index.parquet")
    pdb_yr = pdb.filter(pl.col("release_year").is_not_null())
    print(f"[timebin] PDBBind complexes with release_year: {pdb_yr.height}/{pdb.height}")

    # Unique TimeBin nodes
    years_doc = set(int(y) for y in docs_yr["year"].to_list())
    years_pdb = set(int(y) for y in pdb_yr["release_year"].to_list())
    all_years = sorted(years_doc | years_pdb)
    print(f"[timebin] year range: {min(all_years)}..{max(all_years)} ({len(all_years)} bins)")

    node_rows = []
    for yr in all_years:
        node_rows.append({
            "node_id": _year_bin(yr),
            "node_type": "TimeBin",
            "bin_kind": "year",
            "year": yr,
            "decade": (yr // 10) * 10,
        })
    # Release-version pseudo-bins for benchmarks without year info
    for rel, yr in RELEASES:
        node_rows.append({
            "node_id": f"tbin_rel:{rel}",
            "node_type": "TimeBin",
            "bin_kind": "release",
            "year": yr,
            "decade": (yr // 10) * 10,
        })
    nodes = pl.DataFrame(node_rows)
    nodes.write_parquet(PROCESSED / "timebin_nodes.parquet")
    print(f"[timebin] TimeBin nodes: {nodes.height}")

    # Edges
    edge_rows = []
    # document_in_timebin
    for r in docs_yr.iter_rows(named=True):
        edge_rows.append({
            "src": f"chembl_doc:{r['document_chembl_id']}",
            "dst": _year_bin(r["year"]),
            "edge_type": "document_in_timebin",
        })
    # complex_in_timebin
    for r in pdb_yr.iter_rows(named=True):
        edge_rows.append({
            "src": f"pdbbind_complex:{r['pdb_id']}",
            "dst": _year_bin(int(r["release_year"])),
            "edge_type": "complex_in_timebin",
        })
    # dataset_release_in_timebin (point each dataset to its release pseudo-bin AND its release year)
    for rel, yr in RELEASES:
        edge_rows.append({
            "src": f"src:{rel}",
            "dst": f"tbin_rel:{rel}",
            "edge_type": "dataset_release_in_timebin",
        })
        edge_rows.append({
            "src": f"src:{rel}",
            "dst": _year_bin(yr) if yr in all_years else f"tbin_rel:{rel}",
            "edge_type": "dataset_release_in_timebin",
        })
    edges = pl.DataFrame(edge_rows)
    edges.write_parquet(PROCESSED / "timebin_edges.parquet")
    print(f"[timebin] edges: {edges.height}")

    # MVP2 + TimeBin combined parquets (for graph-summary downstream)
    n_existing = pl.read_parquet(PROCESSED / "mvp2_nodes.parquet").height
    e_existing = pl.read_parquet(PROCESSED / "mvp2_edges.parquet").height
    nodes.write_parquet(PROCESSED / "mvp2_plus_timebin_nodes.parquet")
    edges.write_parquet(PROCESSED / "mvp2_plus_timebin_edges.parquet")
    print(f"[timebin] mvp2+timebin total: nodes={n_existing + nodes.height}, "
          f"edges={e_existing + edges.height}")


if __name__ == "__main__":
    main()
