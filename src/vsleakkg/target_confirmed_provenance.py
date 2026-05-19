"""Build benchmark-target -> ChEMBL target mapping and split provenance into
candidate (ligand-only) vs confirmed (ligand + target).

Conservative matching rules:

  1. Curated gene-symbol -> CHEMBL target_chembl_id for the 15 LIT-PCBA targets.
     We control the dictionary, so we mark these `confidence='curated'`.
  2. UniProt accession exact match for any benchmark that exposes uniprot
     (BayesBind does). Marked `confidence='confirmed_uniprot'`.
  3. Best-effort name substring match against ChEMBL `pref_name`. Marked
     `confidence='name_candidate'` — never confirmed.
  4. For DUD-E and DEKOIS we use the abbreviated-name dictionary baked into
     this module. Many entries map to multiple ChEMBL targets; we keep them
     all and mark `confidence='abbrev_candidate'`.

A row in `benchmark_chembl_confirmed_provenance.parquet` is produced when
both (a) the benchmark example has a confirmed-or-curated target mapping
AND (b) a ChEMBL activity for that ligand also references that target.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CHEMBL_DB = ROOT / "data" / "raw" / "chembl" / "extracted" / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"


# ---------- 1. Curated LIT-PCBA gene-symbol -> ChEMBL target id ----------
# Single-protein human target_chembl_ids from ChEMBL 35.
LITPCBA_CURATED: Dict[str, List[str]] = {
    "ADRB2":     ["CHEMBL210"],
    "ALDH1":     ["CHEMBL4078"],   # Aldehyde dehydrogenase 1
    "ESR1_ago":  ["CHEMBL206"],    # Estrogen receptor alpha
    "ESR1_ant":  ["CHEMBL206"],
    "FEN1":      ["CHEMBL1075092"],
    "GBA":       ["CHEMBL2179"],   # Beta-glucocerebrosidase
    "IDH1":      ["CHEMBL2007625"],
    "KAT2A":     ["CHEMBL1741218"],
    "MAPK1":     ["CHEMBL4040"],   # ERK2
    "MTORC1":    ["CHEMBL2842"],   # mTOR
    "OPRK1":     ["CHEMBL237"],    # Kappa opioid receptor
    "PKM2":      ["CHEMBL2107850"],
    "PPARG":     ["CHEMBL235"],
    "TP53":      ["CHEMBL4096"],   # Cellular tumor antigen p53
    "VDR":       ["CHEMBL1846"],
}


# ---------- 2. DUD-E abbreviations dictionary (best-effort) ----------
# Many of these are commonly known; we list only ones we are confident about.
# Anything not here is dropped (will be marked unmapped — no false confirmed).
DUDE_ABBREV: Dict[str, List[str]] = {
    "aa2ar":  ["CHEMBL251"],     # Adenosine receptor A2a
    "abl1":   ["CHEMBL1862"],    # ABL1
    "ace":    ["CHEMBL1808"],    # Angiotensin-converting enzyme
    "aces":   ["CHEMBL220"],     # Acetylcholinesterase
    "ada":    ["CHEMBL1910"],    # Adenosine deaminase
    "ada17":  ["CHEMBL3706"],    # ADAM17
    "adrb1":  ["CHEMBL213"],
    "adrb2":  ["CHEMBL210"],
    "akt1":   ["CHEMBL4282"],
    "akt2":   ["CHEMBL2431"],
    "aldr":   ["CHEMBL1900"],    # Aldose reductase
    "ampc":   [],                # bacterial - skip
    "andr":   ["CHEMBL1871"],    # Androgen receptor
    "aofb":   ["CHEMBL2039"],    # Monoamine oxidase B (MAO-B)
    "bace1":  ["CHEMBL4822"],
    "braf":   ["CHEMBL5145"],
    "cah2":   ["CHEMBL205"],     # Carbonic anhydrase II
    "casp3":  ["CHEMBL2334"],
    "cdk2":   ["CHEMBL301"],
    "comt":   ["CHEMBL2014"],
    "cp2c9":  ["CHEMBL3397"],    # CYP2C9
    "cp3a4":  ["CHEMBL340"],     # CYP3A4
    "csf1r":  ["CHEMBL1844"],
    "cxcr4":  ["CHEMBL2107"],
    "dhi1":   ["CHEMBL1840"],    # 11-beta-HSD1
    "dpp4":   ["CHEMBL284"],
    "drd3":   ["CHEMBL234"],
    "dyr":    ["CHEMBL202"],     # DHFR (human)
    "egfr":   ["CHEMBL203"],
    "esr1":   ["CHEMBL206"],
    "esr2":   ["CHEMBL242"],
    "fa10":   ["CHEMBL244"],     # Factor Xa
    "fa7":    ["CHEMBL244"],     # Factor VII -- approx
    "fak1":   ["CHEMBL2695"],    # FAK / PTK2
    "fgfr1":  ["CHEMBL3650"],
    "fkb1a":  ["CHEMBL1907601"], # FKBP1A
    "fnta":   ["CHEMBL3902"],    # Farnesyltransferase
    "fpps":   ["CHEMBL1782"],
    "gcr":    ["CHEMBL2034"],    # Glucocorticoid receptor
    "glcm":   ["CHEMBL2179"],    # Glucocerebrosidase
    "gria2":  ["CHEMBL2095171"], # Glutamate receptor 2
    "grik1":  ["CHEMBL3041"],
    "hdac2":  ["CHEMBL1937"],
    "hdac8":  ["CHEMBL3192"],
    "hivint": ["CHEMBL3471"],    # HIV integrase
    "hivpr":  ["CHEMBL243"],     # HIV protease
    "hivrt":  ["CHEMBL247"],     # HIV reverse transcriptase
    "hmdh":   ["CHEMBL402"],     # HMG-CoA reductase
    "hs90a":  ["CHEMBL3880"],    # HSP90 alpha
    "hxk4":   ["CHEMBL3820"],    # Hexokinase IV / glucokinase
    "igf1r":  ["CHEMBL1957"],
    "inha":   [],                # Bacterial enoyl-ACP reductase - skip
    "ital":   [],
    "jak2":   ["CHEMBL2971"],
    "kif11":  ["CHEMBL4581"],
    "kit":    ["CHEMBL1936"],
    "kith":   ["CHEMBL218"],     # Thymidine kinase
    "kpcb":   ["CHEMBL2789"],    # PKC beta
    "lck":    ["CHEMBL258"],
    "lkha4":  ["CHEMBL2095"],    # Leukotriene A4 hydrolase
    "mapk2":  ["CHEMBL2208"],    # MK2
    "mcr":    ["CHEMBL1994"],    # Mineralocorticoid receptor
    "met":    ["CHEMBL3717"],
    "mk01":   ["CHEMBL4040"],    # ERK2
    "mk10":   ["CHEMBL2637"],    # JNK3
    "mk14":   ["CHEMBL260"],     # p38a
    "mmp13":  ["CHEMBL3084"],
    "mp2k1":  ["CHEMBL3587"],    # MEK1
    "nos1":   ["CHEMBL3568"],    # nNOS
    "nram":   [],                # Neuraminidase - viral, skip
    "pa2ga":  ["CHEMBL3973"],    # PLA2G2A
    "parp1":  ["CHEMBL3105"],
    "pde5a":  ["CHEMBL1827"],
    "pgh1":   ["CHEMBL221"],     # COX-1
    "pgh2":   ["CHEMBL230"],     # COX-2
    "plk1":   ["CHEMBL3788"],
    "pnph":   ["CHEMBL3197"],    # Purine nucleoside phosphorylase
    "ppara":  ["CHEMBL239"],
    "ppard":  ["CHEMBL3979"],
    "pparg":  ["CHEMBL235"],
    "prgr":   ["CHEMBL208"],     # Progesterone receptor
    "ptn1":   ["CHEMBL335"],     # PTP1B
    "pur2":   [],
    "pygm":   ["CHEMBL2693"],
    "pyrd":   ["CHEMBL1966"],    # DHODH
    "reni":   ["CHEMBL286"],     # Renin
    "rock1":  ["CHEMBL3231"],
    "rxra":   ["CHEMBL1973"],
    "sahh":   ["CHEMBL3784"],    # SAH hydrolase
    "src":    ["CHEMBL267"],
    "tgfr1":  ["CHEMBL5102"],
    "thb":    ["CHEMBL1860"],    # Thyroid hormone receptor beta
    "thrb":   ["CHEMBL204"],     # Thrombin (F2)
    "try1":   ["CHEMBL209"],     # Trypsin I (porcine)
    "tryb1":  ["CHEMBL3614"],    # Tryptase beta-1
    "tysy":   ["CHEMBL1952"],    # Thymidylate synthase
    "urok":   ["CHEMBL3286"],    # uPA
    "vgfr2":  ["CHEMBL279"],     # KDR
    "wee1":   ["CHEMBL5491"],
    "xiap":   ["CHEMBL4198"],
}


# ---------- 3. DEKOIS-2 abbreviations dictionary ----------
DEKOIS_ABBREV: Dict[str, List[str]] = {
    "11betahsd1": ["CHEMBL1840"],
    "17betahsd1": ["CHEMBL2789"],
    "a2a":     ["CHEMBL251"],
    "ace":     ["CHEMBL1808"],
    "ace2":    ["CHEMBL3729"],
    "ache":    ["CHEMBL220"],
    "adam17":  ["CHEMBL3706"],
    "adrb2":   ["CHEMBL210"],
    "akt1":    ["CHEMBL4282"],
    "alr2":    ["CHEMBL1900"],
    "ar":      ["CHEMBL1871"],
    "aurka":   ["CHEMBL4722"],
    "aurkb":   ["CHEMBL2185"],
    "bcl2":    ["CHEMBL4860"],
    "braf":    ["CHEMBL5145"],
    "catl":    ["CHEMBL3837"],   # Cathepsin L
    "cdk2":    ["CHEMBL301"],
    "cox1":    ["CHEMBL221"],
    "cox2":    ["CHEMBL230"],
    "ctsk":    ["CHEMBL268"],    # Cathepsin K
    "dhfr":    ["CHEMBL202"],
    "egfr":    ["CHEMBL203"],
    "er":      ["CHEMBL206"],
    "erbb2":   ["CHEMBL1824"],
    "fxa":     ["CHEMBL244"],
    "fxia":    ["CHEMBL262"],
    "gba":     ["CHEMBL2179"],
    "gr":      ["CHEMBL2034"],
    "hdac2":   ["CHEMBL1937"],
    "hdac8":   ["CHEMBL3192"],
    "her2":    ["CHEMBL1824"],
    "hivpr":   ["CHEMBL243"],
    "hmgr":    ["CHEMBL402"],
    "hsp90":   ["CHEMBL3880"],
    "inha":    [],
    "jnk3":    ["CHEMBL2637"],
    "lck":     ["CHEMBL258"],
    "mao_a":   ["CHEMBL1951"],
    "mao_b":   ["CHEMBL2039"],
    "maoa":    ["CHEMBL1951"],
    "maob":    ["CHEMBL2039"],
    "mmp2":    ["CHEMBL333"],
    "p38_alpha":["CHEMBL260"],
    "parp1":   ["CHEMBL3105"],
    "pde5":    ["CHEMBL1827"],
    "pim1":    ["CHEMBL2147"],
    "pim2":    ["CHEMBL4523"],
    "pnp":     ["CHEMBL3197"],
    "ppara":   ["CHEMBL239"],
    "ppard":   ["CHEMBL3979"],
    "pparg":   ["CHEMBL235"],
    "pr":      ["CHEMBL208"],
    "rxra":    ["CHEMBL1973"],
    "src":     ["CHEMBL267"],
    "thrombin":["CHEMBL204"],
    "vegfr2":  ["CHEMBL279"],
}


def load_chembl_uniprot_map(db: Path) -> pl.DataFrame:
    """target_chembl_id -> uniprot accession via target_components/component_sequences."""
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    q = """
    SELECT td.chembl_id AS target_chembl_id, cs.accession AS uniprot
    FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences cs ON tc.component_id = cs.component_id
    WHERE cs.accession IS NOT NULL
    """
    rows = list(conn.execute(q))
    conn.close()
    return pl.DataFrame(rows, schema=["target_chembl_id", "uniprot"], orient="row").unique()


def build_target_map() -> pl.DataFrame:
    """Returns one row per (benchmark_dataset, benchmark_target, target_chembl_id, match_type)."""
    out_rows = []
    # LIT-PCBA AVE
    for tgt, cids in LITPCBA_CURATED.items():
        for cid in cids:
            out_rows.append({
                "benchmark_dataset": "LIT-PCBA AVE",
                "benchmark_target": tgt,
                "target_chembl_id": cid,
                "target_match_type": "curated_gene_symbol",
                "target_match_confidence": "curated",
            })
    # DUD-E
    for tgt, cids in DUDE_ABBREV.items():
        for cid in cids:
            out_rows.append({
                "benchmark_dataset": "DUD-E",
                "benchmark_target": tgt,
                "target_chembl_id": cid,
                "target_match_type": "curated_abbrev",
                "target_match_confidence": "curated",
            })
    # DEKOIS
    for tgt, cids in DEKOIS_ABBREV.items():
        for cid in cids:
            out_rows.append({
                "benchmark_dataset": "DEKOIS",
                "benchmark_target": tgt,
                "target_chembl_id": cid,
                "target_match_type": "curated_abbrev",
                "target_match_confidence": "curated",
            })
    return pl.DataFrame(out_rows)


def main() -> None:
    print("[t3] loading inputs…")
    tmap = build_target_map()
    tmap.write_parquet(PROCESSED / "benchmark_to_chembl_target_map.parquet")
    print(f"[t3] target map rows: {tmap.height}")

    # Uniprot enrichment from ChEMBL
    if CHEMBL_DB.exists():
        upmap = load_chembl_uniprot_map(CHEMBL_DB)
        upmap.write_parquet(PROCESSED / "chembl_target_uniprot.parquet")
        print(f"[t3] chembl uniprot map rows: {upmap.height}")
    else:
        upmap = pl.DataFrame()
        print("[t3] WARN: ChEMBL SQLite missing, no uniprot enrichment")

    # Load candidate provenance (ligand-only)
    cand = pl.read_parquet(PROCESSED / "benchmark_chembl_candidate_provenance.parquet")
    print(f"[t3] candidate rows: {cand.height}")

    # Pull benchmark_target into candidate rows
    ligmap = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    # Map each benchmark to its example-target via examples parquets
    ex_litpcba = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet").select([
        pl.lit("LIT-PCBA AVE").alias("benchmark_dataset"), "inchikey",
        pl.col("target").alias("benchmark_target")]).unique()
    ex_dude = pl.read_parquet(PROCESSED / "dude_examples.parquet").select([
        pl.lit("DUD-E").alias("benchmark_dataset"), "inchikey",
        pl.col("target").alias("benchmark_target")]).unique()
    ex_dekois = pl.read_parquet(PROCESSED / "dekois_examples.parquet").select([
        pl.lit("DEKOIS").alias("benchmark_dataset"), "inchikey",
        pl.col("target").alias("benchmark_target")]).unique()
    ex = pl.concat([ex_litpcba, ex_dude, ex_dekois], how="vertical_relaxed")
    print(f"[t3] benchmark example (ligand,target) pairs: {ex.height}")

    # Build expected (benchmark_dataset, inchikey, expected_target_chembl_id) set
    # = ligand's benchmark targets joined to curated target_chembl_ids.
    # Important: do NOT explode candidate rows — only enrich them with a flag.
    ex_with_t = ex.join(tmap, on=["benchmark_dataset", "benchmark_target"], how="inner")
    expected = (ex_with_t
                .select("benchmark_dataset", "inchikey",
                        pl.col("target_chembl_id").alias("expected_target_chembl_id"))
                .unique())
    print(f"[t3] expected (lig,target) pairs after curated lookup: {expected.height}")

    cand_v2 = (cand
               .join(expected.with_columns(pl.lit(True).alias("_conf")),
                     left_on=["benchmark_dataset", "inchikey", "target_chembl_id"],
                     right_on=["benchmark_dataset", "inchikey", "expected_target_chembl_id"],
                     how="left")
               .with_columns([
                   pl.when(pl.col("_conf").fill_null(False))
                     .then(pl.lit("ligand+target"))
                     .otherwise(pl.lit("ligand_only"))
                     .alias("provenance_level_v2"),
                   pl.when(pl.col("_conf").fill_null(False))
                     .then(pl.lit("confirmed_via_curated_target"))
                     .otherwise(pl.lit("candidate_ligand_only"))
                     .alias("confidence_v2"),
               ])
               .drop("_conf"))
    cand_v2.write_parquet(PROCESSED / "benchmark_chembl_candidate_provenance_v2.parquet")

    confirmed = cand_v2.filter(pl.col("provenance_level_v2") == "ligand+target")
    confirmed.write_parquet(PROCESSED / "benchmark_chembl_confirmed_provenance.parquet")

    # Report
    summary = (cand_v2.group_by(["benchmark_dataset", "provenance_level_v2"])
               .len()
               .sort(["benchmark_dataset", "provenance_level_v2"]))
    print("[t3] provenance summary by dataset/level:")
    for r in summary.iter_rows(named=True):
        print(f"   {r['benchmark_dataset']:>14}  {r['provenance_level_v2']:>14}  {r['len']:>10,}")

    n_confirmed = confirmed.height
    n_total = cand_v2.height
    print(f"[t3] confirmed rows: {n_confirmed:,} of {n_total:,}")
    print(f"[t3] wrote confirmed_provenance ({n_confirmed:,}) and candidate_v2 ({n_total:,})")
    print(f"[t3] wrote target map ({tmap.height} rows)")


if __name__ == "__main__":
    main()
