"""Smoke test: ensure all modules import and basic primitives work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/hoangpc/VS-LeakKG/src").resolve()))

print("importing modules...")
from vsleakkg import chem, io, load_litpcba, load_dude, graph_schema, build_graph
from vsleakkg import audit_ligand, diagnostics, contamination_score, run_mvp_audit
print("ok")

print("\n-- chem primitives --")
smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
f = chem.featurize(smi)
print("canonical:", f.smiles_canonical)
print("inchikey: ", f.inchikey)
print("scaffold: ", f.scaffold_smiles)
print("parse_ok: ", f.parse_ok)
fp = chem.ecfp(smi)
print("fp_bits:  ", fp.GetNumBits() if fp else None, " on_bits=", fp.GetNumOnBits() if fp else None)

print("\n-- DUD-E small load --")
import polars as pl
df = load_dude.load_target(Path("D:/hoangpc/VS-LeakKG/data/raw/DUD-E"), "aa2ar")
print("aa2ar rows:", df.height, " label counts:", df.group_by("label").len().sort("label").to_dict(as_series=False))

print("\n-- LIT-PCBA small load --")
extracted = load_litpcba.ensure_extracted(Path("D:/hoangpc/VS-LeakKG/data/raw/LIT-PCBA"))
print("extracted at:", extracted)
df_lit = load_litpcba.load_target(extracted, "ADRB2")
print("ADRB2 rows:", df_lit.height, " label counts:", df_lit.group_by("label").len().sort("label").to_dict(as_series=False))
print("\nALL GOOD")
