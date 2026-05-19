"""Smoke-input adapters for inference-only model evaluation.

These prepare ≤100-example subsets in a handful of common formats:
- a SMILES CSV (most generic)
- a ligand-only CSV with InChIKey + label
- a paired protein/pocket × ligand CSV (only where pocket info is available)
- a ConGLUDe-style `proteins.txt` + `smiles.txt` pair

Each adapter is conservative: it samples a stratified 100-row subset
with at least 10 actives and 10 decoys where possible.
"""
