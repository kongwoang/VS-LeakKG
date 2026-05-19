"""Smoke test PDBBind parsers on a handful of complexes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/hoangpc/VS-LeakKG/src").resolve()))

from vsleakkg import load_pdbbind, run_pdbbind

idx_path = Path("D:/hoangpc/VS-LeakKG/data/raw/PBDBind/extracted/index/INDEX_general_PL.2020R1.lst")
df = load_pdbbind.parse_pl_index(idx_path)
print("index rows:", df.height)
print("affinity_parse_ok:", int(df["affinity_parse_ok"].sum()), "/", df.height)
print(df.head(5).to_pandas().to_string(index=False))

pl_root = Path("D:/hoangpc/VS-LeakKG/data/raw/PBDBind/extracted/P-L")
files = load_pdbbind.discover_complexes(pl_root)
print("discovered complexes:", len(files))

# Sample first 3 from earliest bucket and 2 from latest
sample = files[:3] + files[-2:]
for c in sample:
    print(f"--- {c.pdb_id} ({c.year_bucket}) ---")
    canon, ik, scaf, n_atoms, fmt, ok = load_pdbbind.parse_ligand(c.ligand_mol2, c.ligand_sdf)
    print(f"  ligand ok={ok} fmt={fmt} n_atoms={n_atoms}")
    print(f"  canon={canon}")
    print(f"  inchikey={ik}  scaffold={scaf}")
    prot = load_pdbbind.parse_protein_pdb(c.protein_pdb)
    print(f"  protein parse_ok={prot['parse_ok']} chains={prot['chains']} "
          f"n_residues={prot['n_residues']} n_atoms={prot['n_atoms']}")
    if prot['sequence_concat']:
        print(f"  seq[:60]={prot['sequence_concat'][:60]}")
print("ALL SMOKE OK")
