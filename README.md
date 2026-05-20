# VS-LeakKG

**Contamination-aware heterogeneous provenance graph for virtual-screening benchmark integrity.**

VS-LeakKG is a reproducible framework that audits the four most-used virtual-screening
(VS) benchmarks for *multi-axis leakage* between their validation sets and the public
data resources from which they were constructed. It assembles a typed heterogeneous
provenance graph linking every benchmark molecule, target, assay, document, source
dataset, decoy protocol, and timepoint, then computes a weighted **path-product
contamination score** `C(x_t)` per validation row and feeds it into shortcut-aware
baselines and cold-split generators.

The project's goal is to make benchmark-integrity claims falsifiable: instead of
"AUROC = 0.85, therefore the model generalises", you can ask "of the rows that
the model gets right, what fraction have an identical/analog ligand, identical
protein cluster, or identical assay in the training/provenance data?"

---

## Why this exists

Virtual-screening benchmarks routinely suffer from leakage that inflates reported
AUROC / AP / BEDROC / EF metrics:

- **Ligand-identity leakage** — the same SMILES (or its InChIKey-1) appears in
  both train and test.
- **Scaffold leakage** — different SMILES sharing a Bemis–Murcko scaffold.
- **Analog leakage** — ECFP4 Tanimoto ≥ 0.5 to a training ligand.
- **Protein-cluster leakage** — test target's UniProt sequence clusters with a
  training target's at 30 % identity (MMseqs2 easy-cluster).
- **Pocket-composition leakage** — pocket residues' amino-acid composition
  matches a training pocket within ε; an MVP proxy for Foldseek 3D similarity.
- **Assay / document leakage** — the same `assay_id` or PubMed document
  contributed activities to both sides.
- **Source-dataset leakage** — DUD-E vs LIT-PCBA share targets and ligands at
  the *resource* level even when their splits look different.
- **Decoy-protocol leakage** — the decoy-generation protocol itself biases
  shortcut features (DUD-E property-matched decoys are the canonical example).
- **Temporal leakage** — training documents are dated after the test's earliest
  document.

VS-LeakKG models all nine axes simultaneously as nodes + edges in a single graph,
weights each contamination path by a domain-informed coefficient, and exposes
the resulting score both as a diagnostic (per-row, per-axis) and as a baseline
(`contamination-NN`: label-transfer from the contamination-nearest neighbour;
mean AUROC ≈ 0.620 across LIT-PCBA AVE targets — strong evidence that "shortcuts
alone" already beat random by a wide margin).

---

## What this repository ships

A clean, reproducible **Python package** + **shell launchers**, with **no datasets,
no generated results, no figures, no model checkpoints, and no model weights**.

You bring the raw archives (per `data/MANIFEST.template.md`), and the pipeline
reproduces the full audit on Linux or Windows.

### Five concrete deliverables of the pipeline

1. **Typed heterogeneous provenance graph** — 22+ node types
   (`Ligand`, `Scaffold`, `Analog`, `Protein`, `ProteinCluster`, `Pocket`,
   `PocketCluster`, `Assay`, `Document`, `Target`, `Complex`, `SourceDataset`,
   `DecoyProtocol`, `TimeBin`, …) and 28+ edge types
   (`HAS_SCAFFOLD`, `IS_ANALOG_OF`, `MEASURED_IN`, `REPORTED_IN`,
   `CLUSTERS_WITH`, `BELONGS_TO_POCKETCLUSTER`, `CO_OCCURS_WITH`, …),
   emitted as parquet under `data/processed/graph/`.
2. **Weighted path-product contamination score** `C(x_t)` per validation row
   (`src/vsleakkg/weighted_contamination.py`), broken down by axis so you can
   answer "how much of this score comes from ligand-identity vs scaffold vs
   protein-cluster".
3. **Shortcut diagnostic baselines** — ligand-KNN (ECFP4), scaffold
   memorisation, **contamination-NN** (label-transfer from the
   contamination-nearest neighbour), and source-only diagnostics that gauge
   how predictive the *source dataset label alone* is.
4. **Multi-axis cold split generators** — `ligand-cold`, `scaffold-cold`,
   `protein-cold`, and `strict-cold` (intersection of all three) variants of
   each benchmark, plus the original split as control
   (`src/vsleakkg/split_generator.py`).
5. **Comparison metrics + figures** — AUROC, AP, BEDROC, EF@0.5/1/5 % computed
   per (benchmark × split-variant × baseline) and rendered as Tables 1–6 and
   Figures 1–4 (`src/vsleakkg/final_tables.py`, `final_figures.py`,
   `final_figures_v2.py`).

### Status against the original proposal

30 of 40 proposal items built (~75 %). The remaining items are either
deferred-on-purpose (no model training, no full PLINDER) or require artefacts
that cannot be redistributed (DrugCLIP / LigUnity / HypSeek weights).
The four `environments/model_eval_*.yml` files plus the
`src/vsleakkg/model_eval/` adapter sub-package wire up inference-only
smoke-input adapters so that the four reference models can be run by anyone
who obtains the weights independently.

---

## Datasets audited

| Benchmark | Role | Notes |
|---|---|---|
| **LIT-PCBA AVE-unbiased** | primary validation set (15 targets) | `actives_*.smi` / `inactives_*.smi` per target |
| **DUD-E** | secondary validation set (102 targets) | property-matched decoys; decoy-protocol leakage probe |
| **DEKOIS 2.0** | secondary validation set (81 targets) | curated decoys, distinct protocol |
| **BayesBind V1.5** | val + test (uncontaminated by construction) | independent reference for "clean" baseline |
| **ChEMBL 35** | provenance database | activities, documents, assays, targets, component_sequences |
| **BindingDB** | provenance database | cross-source activity reconciliation |
| **PDBBind v2020** | provenance database | bound-complex evidence, pocket source |
| **BigBind V1.5 metadata** | provenance metadata only | 12 top-level CSVs; full structural archive not consumed |

The exact archive URLs, expected paths, and sizes are listed in
`data/MANIFEST.template.md`.

---

## Getting the raw data

The repository ships **no datasets**. There are two ways to populate `data/raw/`:

### Path A — restore from the dataset archive (recommended)

A single `.zip` named **`VS-LeakKG_raw_datasets_YYYYMMDD.zip`** holds every
raw archive the pipeline needs (~27.85 GiB, ZIP_STORED so it opens fast and
adds no extra compression on top of the already-compressed sources).

The archive is **not in Git**. It lives in a **private Hugging Face dataset
repo**:

> https://huggingface.co/datasets/kongwoang/VS_LeakKG

To download, you need:

1. A Hugging Face account with read access to the repo (ask the project
   owner to grant access).
2. An HF token from https://huggingface.co/settings/tokens (Role: *Read* is
   enough).

The fastest path is the bundled fetcher — it reads the current version
filename from `scripts/dataset_version.sh` / `scripts/_dataset_version.ps1`,
so reproducing the **latest** release is always the same command:

```bash
export HF_TOKEN=<your hf token>
bash scripts/fetch_dataset.sh         # → _dataset_cache/<current version>.zip
```

```powershell
$env:HF_TOKEN = '<your hf token>'
.\scripts\fetch_dataset.ps1
```

Both are idempotent — re-running is a no-op once the zip is cached.

If you'd rather invoke `huggingface-cli` yourself (e.g. to land the zip
somewhere outside the repo):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                       # paste your token, or use HF_TOKEN
huggingface-cli download kongwoang/VS_LeakKG \
    VS-LeakKG_raw_datasets_20260519.zip \
    --repo-type dataset --local-dir .
```

If you do not have HF access, rebuild the archive yourself from any prior
pipeline run (see "Rebuilding the dataset archive" at the bottom of this
file).

What's inside the .zip:

```
raw/
├── ChEMBL/chembl_35_sqlite.tar.gz                 (~4.6 GB compressed)
├── BindingDB/BindingDB_All_202605_tsv.zip         (~0.5 GB compressed)
├── PBDBind/{P-L.tar.gz, index.tar.gz}             (~3.1 GB compressed)
├── BigBind/{BigBindV1.5.tar.gz, metadata/}        (~18.9 GB)
├── LIT-PCBA/{full_data.tgz, splits/}              (~0.34 GB)
├── BayesBind/BayesBindV1.5.tar.gz                 (~0.1 GB)
├── DEKOIS/DEKOIS2.zip                             (~0.08 GB)
├── DUD-E/<102 target dirs>                        (~0.1 GB)
└── manual_downloads_needed/, PLINDER/             (TODO notes only)
VS_LeakKG_proposal.pdf
data_MANIFEST_run_specific.md
```

After cloning the repo, restore `data/raw/` (extracts every inner archive
into the `<dataset>/extracted/` layout the pipeline expects). With no
argument, both scripts read the cached zip in `_dataset_cache/`:

```bash
# Linux / macOS  (uses _dataset_cache/<current version>.zip)
bash scripts/extract_datasets.sh

# or pass an explicit path
bash scripts/extract_datasets.sh /path/to/VS-LeakKG_raw_datasets_20260519.zip
```

```powershell
# Windows / PowerShell
.\scripts\extract_datasets.ps1
.\scripts\extract_datasets.ps1 -Zip "D:\path\to\VS-LeakKG_raw_datasets_20260519.zip"
```

Both scripts are **idempotent** — re-running skips any inner archive whose
target `extracted/` is already non-empty.

### Path B — fetch each source independently

If you don't have the archive, follow `data/MANIFEST.template.md` for the
official source URL of every dataset, place each archive under the expected
`data/raw/<DATASET>/` path, then run:

```bash
bash scripts/setup_data.sh         # downloads what is auto-downloadable
```

PDBBind requires registration and BindingDB rotates older releases, so a
fresh download will not give byte-identical inputs.

---

## Quick start (Linux)

After you've cloned the repo and set `HF_TOKEN`, the entire pipeline — download,
extract, graph build, audits — runs from a single command:

```bash
# 1. Clone + environment
git clone git@github.com:kongwoang/VS-LeakKG.git
cd VS-LeakKG
conda env create -f environment.yml
conda activate vsleakkg
pip install -e .
pip install -U "huggingface_hub[cli]"

# 2. Tools
conda install -y -c bioconda mmseqs2          # required
conda install -y -c bioconda foldseek         # optional (3D pocket clustering)

# 3. One-shot reproduce
export HF_TOKEN=<your hf token>
bash scripts/reproduce.sh
```

`reproduce.sh` auto-fetches the current dataset zip from Hugging Face into
`_dataset_cache/` if it isn't already there, then extracts, builds the graph,
and runs both audit passes.

To run the stages manually instead:

```bash
bash scripts/fetch_dataset.sh                # download (idempotent)
bash scripts/extract_datasets.sh             # restore data/raw/
bash scripts/overnight_run.sh                # graph build
bash scripts/run_mvp_audit.sh
bash scripts/run_mvp1_audit.sh
```

If you already have the zip on disk, point `reproduce.sh` at it:
`bash scripts/reproduce.sh /path/to/VS-LeakKG_raw_datasets_20260519.zip`.

All produced artifacts land under `data/processed/` and `outputs/` — both
git-ignored by default.

### Quick start (Windows / PowerShell)

The package is platform-agnostic; only the launchers differ:

```powershell
conda env create -f environment.yml
conda activate vsleakkg
pip install -e .
pip install -U "huggingface_hub[cli]"
# MMseqs2 Windows build at C:\Tools\mmseqs2\mmseqs\bin\mmseqs.exe is what
# was used during development; install via Cygwin or use the conda binary.

$env:HF_TOKEN = '<your hf token>'

# fetch + restore data/raw/
.\scripts\fetch_dataset.ps1
.\scripts\extract_datasets.ps1

# then run the audits
python -m vsleakkg.run_overnight
python -m vsleakkg.run_mvp_audit
python -m vsleakkg.run_mvp1_audit
```

---

## Pipeline stages in order

1. **Raw ingestion** (`scripts/setup_data.sh`, `scripts/download_full_cache.{sh,ps1}`)
   downloads what is publicly auto-downloadable (LIT-PCBA, DUD-E, DEKOIS,
   BindingDB, ChEMBL SQLite, BayesBind, BigBind metadata). PDBBind requires
   registration; place it manually per the manifest.
2. **Per-dataset loaders** (`src/vsleakkg/load_*.py`) parse each raw archive
   into a canonicalised parquet under `data/processed/`. Every loader emits
   a stable schema (`smiles`, `inchi_key`, `target_uniprot`,
   `source_dataset`, `assay_id`, `doc_id`, `time_year`, `label`, …).
3. **PDBBind enrichment** (`src/vsleakkg/run_pdbbind.py`,
   `pdbbind_cluster_proteins.py`, `pdbbind_chembl_target_match.py`) clusters
   PDBBind protein sequences and confirms 3,606 PDBBind↔ChEMBL target matches
   via MMseqs2 sequence search.
4. **Graph build** (`src/vsleakkg/build_graph.py`, `run_overnight.py`)
   joins all loaders and enrichments into a single typed graph
   (`graph_schema.py`), serialised as edge-typed parquet shards.
5. **Audits** (`src/vsleakkg/audit_ligand.py`, `diagnostics.py`,
   `bayesbind_diagnostics.py`) compute per-row contamination axes.
6. **Contamination score** (`src/vsleakkg/contamination_score.py`,
   `weighted_contamination.py`) applies the proposal's weighted-path-product
   formula.
7. **Baselines** (`src/vsleakkg/contamination_nn.py`,
   `source_only_diagnostics.py`) — shortcut-only models that should *not*
   beat a clean model on a clean split; if they do, the split is leaky.
8. **Split generation** (`src/vsleakkg/split_generator.py`) emits
   `ligand-cold`, `scaffold-cold`, `protein-cold`, `strict-cold` partitions.
9. **Comparison** (`src/vsleakkg/split_comparison.py`,
   `decile_worst_group.py`) compares original vs generated splits per
   benchmark and computes worst-decile group metrics.
10. **Reporting** (`src/vsleakkg/final_tables.py`, `final_figures.py`,
    `final_figures_v2.py`, `metrics.py`) emits Tables 1–6 and Figures 1–4
    under `outputs/`.

---

## Repository layout

```
VS-LeakKG/
├── environment.yml          conda env (rdkit, polars, scikit-learn, …)
├── requirements.txt         pip-only extras
├── README.md                this file
├── cleanup_report.md        audit of the GitHub-release cleanup
├── .gitignore               excludes data/, outputs/, *.parquet, *.pt, …
├── .gitattributes           line-ending normalisation
├── data/
│   ├── raw/                 placeholder; raw archives go here
│   ├── processed/           placeholder; pipeline outputs here
│   └── MANIFEST.template.md copy to MANIFEST.md and fill in
├── src/vsleakkg/            the Python package (37 modules)
│   ├── load_bayesbind.py
│   ├── load_bindingdb.py
│   ├── load_chembl.py / load_chembl_db.py
│   ├── load_dekois.py
│   ├── load_dude.py
│   ├── load_litpcba.py / load_litpcba_ave.py
│   ├── load_pdbbind.py
│   ├── build_graph.py / graph_schema.py
│   ├── run_overnight.py / run_pdbbind.py
│   ├── run_mvp_audit.py / run_mvp1_audit.py
│   ├── audit_ligand.py / diagnostics.py / bayesbind_diagnostics.py
│   ├── contamination_score.py / weighted_contamination.py
│   ├── contamination_nn.py / source_only_diagnostics.py
│   ├── split_generator.py / split_comparison.py
│   ├── pdbbind_cluster_proteins.py
│   ├── pdbbind_chembl_target_match.py
│   ├── pocket_cluster.py / timebin.py
│   ├── target_confirmed_provenance.py
│   ├── decile_worst_group.py
│   ├── final_tables.py / final_figures.py / final_figures_v2.py
│   ├── metrics.py / chem.py / io.py
│   └── model_eval/          inference-only smoke-input adapters
│       ├── __init__.py
│       ├── common.py
│       └── prepare_smoke_inputs.py
├── scripts/                 shell + PowerShell launchers
│   ├── dataset_version.sh / _dataset_version.ps1   <-- bump for new release
│   ├── fetch_dataset.{sh,ps1}                       HF download
│   ├── extract_datasets.{sh,ps1}                    restore data/raw/
│   ├── reproduce.sh                                 end-to-end orchestrator
│   ├── setup_data.sh
│   ├── overnight_run.sh
│   ├── run_mvp_audit.sh / run_mvp1_audit.sh
│   ├── download_full_cache.{sh,ps1}
│   ├── process_pdbbind.{sh,ps1}
│   ├── fetch_dude.ps1 / retry_dude.ps1
│   ├── log_disk.ps1
│   └── _probe_env.py / _smoke_mvp1.py / _smoke_pdbbind.py / _smoke_test.py
├── environments/            per-model env yml files
│   ├── model_eval_conglude.yml
│   ├── model_eval_drugclip.yml
│   ├── model_eval_hypseek.yml
│   └── model_eval_ligunity.yml
├── notebooks/               empty placeholder
└── outputs/                 placeholder; reports / tables / figures land here
```

---

## What this repo does **not** ship

- **No datasets.** The raw archives (ChEMBL SQLite, BindingDB TSV, PDBBind
  structures, LIT-PCBA / DUD-E / DEKOIS lists, BayesBind, BigBind metadata)
  must be obtained from their official sources — see
  `data/MANIFEST.template.md`.
- **No processed parquets.** `data/processed/` is empty in Git.
- **No outputs.** `outputs/` (reports, tables, figures) is empty in Git.
- **No model checkpoints.** The four model evaluation repos (ConGLUDe,
  DrugCLIP, LigUnity, HypSeek) are not vendored; clone them under
  `external/model_eval/` yourself when needed.
- **No PLINDER full data.** Cross-protein structural similarity is captured
  via MMseqs2 sequence clustering; 3D pocket similarity uses the
  amino-acid-composition MVP unless Foldseek is installed.

---

## Outputs you should expect after a full run

After `bash scripts/overnight_run.sh && bash scripts/run_mvp_audit.sh &&
bash scripts/run_mvp1_audit.sh`, `outputs/` will contain (approximate
sizes shown):

```
outputs/
├── reports/
│   ├── mvp_audit_report.md            ~50 KB
│   ├── mvp1_litpcba_ave_report.md     ~30 KB
│   ├── pdbbind_chembl_match_report.md ~10 KB
│   └── contamination_nn_results.md    ~20 KB
├── tables/
│   ├── table1_leakage_overview.csv
│   ├── table2_per_axis_contamination.csv
│   ├── table3_split_comparison.csv
│   ├── table4_baseline_metrics.csv
│   ├── table5_decile_worst_group.csv
│   └── table6_source_only_diagnostics.csv
└── figures/
    ├── fig1_graph_schema.png
    ├── fig2_contamination_distribution.png
    ├── fig3_split_comparison_auroc.png
    └── fig4_per_target_breakdown.png
```

Total `outputs/` footprint is on the order of **~10 MB**; `data/processed/`
is on the order of **~5 GB**. Plan for at least **100 GB free** during a
full reproduction so the raw archives, extracted trees, and processed
parquets all fit.

---

## Project rules of engagement

This codebase was developed under the following operational rules:

- **No model training.** All shortcut scores come from rule-based features
  (InChIKey-1 overlap, ECFP4 Tanimoto, Bemis–Murcko scaffold equality,
  weighted path products, label-transfer from train neighbours). The
  pipeline never fits or fine-tunes a deep model.
- **No PLINDER full data.** Cross-protein structural similarity is captured
  via MMseqs2 sequence clustering; 3D pocket similarity uses the
  composition MVP unless Foldseek is installed.
- **No BigBind extracted archive.** Only the 12 top-level metadata CSVs
  are consumed.
- **All raw downloads are append-only.** The pipeline never deletes
  `data/raw/` contents.
- **No checkpoints / weights vendored.** The four `environments/model_eval_*.yml`
  files and the `src/vsleakkg/model_eval/` adapters wire up inference for
  ConGLUDe / DrugCLIP / LigUnity / HypSeek so users who obtain the weights
  can run those models, but the weights themselves stay out of Git.

---

## Linux portability notes

The repo was developed on Windows 11 with PowerShell; the package is plain
Python + polars + RDKit and runs unchanged on Linux. Two integration points
need a one-line edit when porting:

- **MMseqs2 path.** Update the path constant in
  `src/vsleakkg/pdbbind_cluster_proteins.py` and
  `src/vsleakkg/pdbbind_chembl_target_match.py` from
  `C:\Tools\mmseqs2\mmseqs\bin\mmseqs.exe` to whatever
  `which mmseqs` returns after `conda install -c bioconda mmseqs2`.
- **Foldseek (optional).** Install via
  `conda install -c bioconda foldseek`. The default pipeline runs the
  AA-composition MVP and does not require Foldseek.

---

## License

Code is released under the terms in `LICENSE` (add one). The datasets retain
their own licenses; please consult each dataset's terms before
redistribution.

---

## Citation

If you use VS-LeakKG in a publication, please cite the project (paper draft
in preparation). For the underlying benchmarks and resources, cite
DUD-E (Mysinger et al. 2012), LIT-PCBA (Tran-Nguyen et al. 2020),
DEKOIS-2 (Bauer et al. 2013), BayesBind (Brocidiacono et al. 2024),
PDBBind (Liu et al. 2017), ChEMBL (Mendez et al. 2019), and
BindingDB (Liu et al. 2007).

---

## Project status — what is built and what is not

**Built (30 / 40 proposal items, ≈ 75 %):**

- Full provenance graph (22+ node types, 28+ edge types)
- Per-dataset loaders for all 7 listed sources + BigBind metadata
- PDBBind protein clustering (MMseqs2 easy-cluster, 30 % identity)
- PDBBind ↔ ChEMBL target matching (3,606 confirmed complexes)
- Weighted path-product contamination score, per axis
- Contamination-NN baseline (mean AUROC ≈ 0.620 across LIT-PCBA AVE targets)
- Ligand-KNN, scaffold-memorisation, source-only baselines
- Four cold-split generators (ligand / scaffold / protein / strict)
- Original vs generated split comparison with AUROC / AP / BEDROC / EF
- Decile-worst-group metrics
- Tables 1–6 and Figures 1–4 generators
- Inference adapters for ConGLUDe / DrugCLIP / LigUnity / HypSeek

**Deferred on purpose:**

- No model training pass (rules of engagement).
- No PLINDER full data ingestion (rules of engagement).
- No vendored model weights (license + size).

**Pending (would extend the audit):**

- Foldseek-based 3D pocket clustering as a replacement for the
  AA-composition MVP (the loader hooks are in place).
- Larger-scale chronological-leakage analysis using
  `src/vsleakkg/timebin.py` already wired in.
- A consolidated "leakage scorecard" PDF report combining the existing
  Markdown reports + the figure grid into a single artefact.

For a finer-grained accounting of cleanup actions taken before the
GitHub release, see `cleanup_report.md`.

---

## Rebuilding the dataset archive

If you've already run the pipeline once and want to produce your own
`VS-LeakKG_raw_datasets_YYYYMMDD.zip` (e.g. to share with a collaborator),
the recipe is:

1. Make sure `data/raw/` contains only the **source archives** — i.e. the
   `*.tar.gz`, `*.zip`, and per-target dirs listed in
   `data/MANIFEST.template.md`. Delete any `<dataset>/extracted/` trees
   first; they are reproducible from the archives next to them and would
   bloat the zip.
2. Create a STORE-mode `.zip` (no recompression — the inputs are already
   compressed). The shortest recipe is a one-line Python:

   ```bash
   python - <<'PY'
   import zipfile, os
   from pathlib import Path
   src = Path("data/raw")
   with zipfile.ZipFile("VS-LeakKG_raw_datasets.zip", "w",
                        compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
       for p in src.rglob("*"):
           if p.is_file():
               zf.write(p, arcname=p.relative_to(src.parent).as_posix())
   PY
   ```

   On a fast NTFS volume this takes ~30 s and produces a ~28 GiB zip with
   ZIP64 extensions (needed for the >4 GB BigBind archive).

3. Upload the new zip to the Hugging Face dataset repo, then bump the
   filename in **`scripts/dataset_version.sh`** (and its PowerShell mirror
   `scripts/_dataset_version.ps1`) so the fetcher pulls the new release.
   No other file needs to change.
