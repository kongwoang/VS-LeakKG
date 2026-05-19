# VS-LeakKG

**Contamination-aware heterogeneous provenance graph for virtual-screening benchmark integrity.**

VS-LeakKG audits four virtual-screening benchmarks (LIT-PCBA AVE, DUD-E, DEKOIS-2,
BayesBind V1.5) and three provenance databases (ChEMBL 35, BindingDB, PDBBind v2020) for
multi-axis leakage: ligand identity / scaffold / analog, protein sequence cluster,
pocket composition cluster, assay, document, dataset source, decoy protocol, and time.

It produces:

1. a typed heterogeneous graph with 22+ node types and 28+ edge types,
2. a weighted path-product contamination score per validation row,
3. shortcut diagnostic baselines (ligand-KNN, scaffold memorisation,
   contamination-NN, source-only),
4. multi-axis cold split generators (ligand-cold / scaffold-cold / protein-cold /
   strict-cold), and
5. comparison metrics (AUROC, AP, BEDROC, EF@0.5/1/5 %) on original vs generated splits.

The repository **does not contain any dataset**, generated report, figure, or
model checkpoint. Run the pipeline locally with the raw archives placed at the
paths in `data/MANIFEST.template.md`.

---

## Quick start (Linux)

```bash
# 1. Clone
git clone https://github.com/kongwoang/VS-LeakKG.git
cd VS-LeakKG

# 2. Create the conda environment
conda env create -f environment.yml
conda activate vsleakkg

# 3. Install the package itself (editable)
pip install -e .

# 4. Install MMseqs2 (used for protein and PDBBind→ChEMBL clustering)
conda install -y -c bioconda mmseqs2

# 5. (Optional) Install Foldseek for 3D pocket clustering
conda install -y -c bioconda foldseek

# 6. Place raw data under data/raw/ per data/MANIFEST.template.md
#    (the repo does not ship any data)
cp data/MANIFEST.template.md data/MANIFEST.md
# edit data/MANIFEST.md if you want to record local paths

# 7. Run the setup script (downloads what is auto-downloadable)
bash scripts/setup_data.sh

# 8. Run the full pipeline (overnight orchestrator)
bash scripts/overnight_run.sh

# 9. Run the audit + experiment passes
bash scripts/run_mvp_audit.sh
bash scripts/run_mvp1_audit.sh
```

All produced artifacts land under `data/processed/` and `outputs/` — both
git-ignored by default.

---

## Repository layout

```
VS-LeakKG/
├── environment.yml         conda env (rdkit, polars, scikit-learn, …)
├── requirements.txt        pip-only extras
├── README.md               this file
├── .gitignore              excludes data/, outputs/, *.parquet, *.pt, …
├── data/
│   ├── raw/                placeholder; raw archives go here
│   ├── processed/          placeholder; pipeline outputs here
│   └── MANIFEST.template.md  copy to MANIFEST.md and fill in
├── src/vsleakkg/           the Python package
│   ├── load_*.py             per-dataset readers
│   ├── build_graph.py        graph assembly
│   ├── metrics.py            AUROC / AP / BEDROC / EF
│   ├── weighted_contamination.py  proposal-style C(x_t) scoring
│   ├── contamination_nn.py   contamination-NN baseline
│   ├── source_only_diagnostics.py
│   ├── decile_worst_group.py
│   ├── split_generator.py    multi-axis cold splits
│   ├── split_comparison.py   original vs generated diagnostics
│   ├── pdbbind_cluster_proteins.py  mmseqs2 wrapper
│   ├── pdbbind_chembl_target_match.py
│   ├── pocket_cluster.py     pocket AA-composition MVP clustering
│   ├── timebin.py            TimeBin node type
│   ├── final_tables.py       table1..table6 producers
│   ├── final_figures.py / final_figures_v2.py
│   ├── run_mvp_audit.py / run_mvp1_audit.py / run_overnight.py
│   └── model_eval/           inference-only smoke-input adapters
├── scripts/                shell + PowerShell launchers
├── environments/           per-model env yml files (ConGLUDe / DrugCLIP / LigUnity / HypSeek)
├── notebooks/              empty placeholder
└── outputs/                placeholder; reports / tables / figures land here
```

---

## What this repo does **not** ship

- No datasets. The raw archives (ChEMBL SQLite, BindingDB TSV, PDBBind structures,
  LIT-PCBA / DUD-E / DEKOIS lists, BayesBind, BigBind metadata) must be obtained
  from their official sources — see `data/MANIFEST.template.md`.
- No `data/processed/` parquets.
- No `outputs/` reports, tables, or figures.
- No model checkpoints. The four model evaluation repos (ConGLUDe, DrugCLIP,
  LigUnity, HypSeek) are not vendored; clone them under `external/model_eval/`
  yourself when needed.

---

## License

Code is released under the terms in `LICENSE` (add one). The datasets retain
their own licenses; please consult each dataset's terms before redistribution.

---

## Citation

If you use VS-LeakKG in a publication, please cite the project (paper draft
in preparation). For the underlying benchmarks and resources, cite
DUD-E (Mysinger et al. 2012), LIT-PCBA (Tran-Nguyen et al. 2020),
DEKOIS-2 (Bauer et al. 2013), BayesBind, PDBBind, ChEMBL, and BindingDB.

---

## Project rules of engagement

This codebase was developed under the following operational rules:

- **No model training.** All shortcut scores come from rule-based features
  (InChIKey overlap, Tanimoto, scaffold equality, weighted path products,
  label-transfer from train neighbours). The pipeline never fits or fine-tunes
  a deep model.
- **No PLINDER full data.** Cross-protein structural similarity is captured
  via MMseqs2 sequence clustering; 3D pocket similarity uses the composition
  MVP unless Foldseek is installed.
- **No BigBind extracted archive.** Only the 12 top-level metadata CSVs are
  consumed.
- **All raw downloads are append-only.** The pipeline never deletes
  `data/raw/` contents.
