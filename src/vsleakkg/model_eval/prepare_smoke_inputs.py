"""Build ≤100-row smoke subsets for LIT-PCBA AVE, DUD-E, DEKOIS, BayesBind."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from vsleakkg.model_eval.common import PROCESSED, OUT, stratified_subset, write_smiles_csv

ROOT = Path(__file__).resolve().parents[3]


def litpcba_smoke():
    df = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet").filter(
        pl.col("split") == "validation"
    ).select("target", "label", "inchikey", "smiles_canonical")
    sub = stratified_subset(df, "label", n=100)
    out_dir = OUT / "litpcba"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_smiles_csv(sub, out_dir / "smoke_smiles_label.csv")
    sub.write_parquet(out_dir / "smoke.parquet")
    # ConGLUDe-style: smiles.txt + proteins.txt (one target per row)
    sub.select("smiles_canonical").write_csv(out_dir / "smiles.txt",
                                              include_header=False)
    targets = sub["target"].unique().to_list()
    (out_dir / "proteins.txt").write_text("\n".join(t for t in targets if t))
    return sub.height


def dude_smoke():
    df = pl.read_parquet(PROCESSED / "dude_examples.parquet").select(
        "target", "label", "inchikey", "smiles_canonical")
    sub = stratified_subset(df, "label", n=100)
    out_dir = OUT / "dude"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_smiles_csv(sub, out_dir / "smoke_smiles_label.csv")
    sub.write_parquet(out_dir / "smoke.parquet")
    sub.select("smiles_canonical").write_csv(out_dir / "smiles.txt", include_header=False)
    (out_dir / "proteins.txt").write_text("\n".join(sorted(set(sub["target"].to_list()))))
    return sub.height


def dekois_smoke():
    df = pl.read_parquet(PROCESSED / "dekois_examples.parquet").select(
        "target", "label", "inchikey", "smiles_canonical")
    sub = stratified_subset(df, "label", n=100)
    out_dir = OUT / "dekois"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_smiles_csv(sub, out_dir / "smoke_smiles_label.csv")
    sub.write_parquet(out_dir / "smoke.parquet")
    sub.select("smiles_canonical").write_csv(out_dir / "smiles.txt", include_header=False)
    (out_dir / "proteins.txt").write_text("\n".join(sorted(set(sub["target"].to_list()))))
    return sub.height


def bayesbind_smoke():
    df = pl.read_parquet(PROCESSED / "bayesbind_examples.parquet").select(
        "target", "split", "label", "inchikey", "smiles_canonical", "uniprot")
    # Use val split (random + active mix)
    df = df.filter(pl.col("split") == "val")
    sub = stratified_subset(df, "label", n=100)
    out_dir = OUT / "bayesbind"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_smiles_csv(sub, out_dir / "smoke_smiles_label.csv",
                     extra_cols=("target", "split", "label", "inchikey", "uniprot"))
    sub.write_parquet(out_dir / "smoke.parquet")
    sub.select("smiles_canonical").write_csv(out_dir / "smiles.txt", include_header=False)
    (out_dir / "uniprots.txt").write_text(
        "\n".join(sorted(set(x for x in sub["uniprot"].to_list() if x))))
    return sub.height


def main():
    rows = {
        "LIT-PCBA AVE val": litpcba_smoke(),
        "DUD-E":             dude_smoke(),
        "DEKOIS":            dekois_smoke(),
        "BayesBind val":     bayesbind_smoke(),
    }
    print("[smoke] wrote subsets:")
    for ds, n in rows.items():
        print(f"   {ds:25s}  n={n}")


if __name__ == "__main__":
    main()
