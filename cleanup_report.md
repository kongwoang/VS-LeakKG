# VS-LeakKG cleanup report

Date: 2026-05-19
Operator: autonomous cleanup pass

## 1. Backups

A single timestamped backup root holds **everything that was moved out of
the repository** (raw data, processed parquets, generated outputs, cloned
external repos, the project's old `data/MANIFEST.md`, and the proposal
PDF):

```
D:\hoangpc\VS-LeakKG_backups\20260519_090250\
├── raw\          75.9 GB   (data/raw → here)
├── processed\     4.47 GB  (data/processed → here)
├── outputs\        ~5 MB   (outputs/ → here; per-run reports + tables + figures)
├── external\      0.32 GB  (external/ → here; 4 cloned model repos incl. ConGLUDe weights)
├── data_MANIFEST_run_specific.md   (the previous run-tracking MANIFEST.md)
└── VS_LeakKG_proposal.pdf          (the original VS-LeakKG proposal PDF)
```

**No re-compression was performed.** Raw archives under
`data/raw/<source>/*.tar.gz` are already compressed; the SQLite extract
(`chembl_35.db`, 26 GB) is internally compressed and re-zipping yields
negligible savings while taking hours. Same-drive `Move-Item` on NTFS is
instant regardless of size.

Free disk before/after the backup move: 321.95 GB → 321.68 GB (the move
itself is metadata-only; no file copy).

## 2. What was removed from the repository tree

| Path | Before | Status |
|---|---|---|
| `data/raw/` | 77 GB raw data archives + extracts | moved to backup |
| `data/processed/` | 4.5 GB parquet outputs from all prior runs | moved to backup |
| `data/MANIFEST.md` | run-specific manifest with absolute Windows paths | moved to backup |
| `outputs/` | full report + table + figure tree (~5 MB) | moved to backup |
| `external/` | 4 cloned model repos + LIT-PCBA-audit + chembl-downloader + plinder | moved to backup |
| `VS_LeakKG.pdf` | the original proposal PDF | moved to backup |
| `src/**/__pycache__/` | Python bytecode caches | deleted |

## 3. What remains in the Git repository

70 tracked files, none larger than 1 MB. Full tree:

```
.gitattributes
.gitignore
README.md                              (new: Linux reproduction instructions)
cleanup_report.md                      (this file)
data/
├── .gitkeep (raw/, processed/)        (preserve dir layout)
└── MANIFEST.template.md               (new: dataset paths users must fill in)
environment.yml                        (conda env)
environments/
├── model_eval_conglude.yml
├── model_eval_drugclip.yml
├── model_eval_hypseek.yml
└── model_eval_ligunity.yml
notebooks/.gitkeep
outputs/.gitkeep
requirements.txt
scripts/                               (16 launchers; .sh + .ps1)
└── _probe_env.py, _smoke_mvp1.py, _smoke_pdbbind.py, _smoke_test.py,
   download_full_cache.{sh,ps1}, fetch_dude.ps1, log_disk.ps1,
   overnight_run.sh, process_pdbbind.{sh,ps1}, retry_dude.ps1,
   run_mvp1_audit.sh, run_mvp_audit.sh, setup_data.sh
src/vsleakkg/                          (37 .py modules + model_eval sub-package)
├── core loaders: load_bayesbind, load_bindingdb, load_chembl,
│                load_chembl_db, load_dekois, load_dude, load_litpcba,
│                load_litpcba_ave, load_pdbbind
├── graph: build_graph, graph_schema, run_overnight, run_pdbbind,
│         run_mvp_audit, run_mvp1_audit
├── audits: audit_ligand, diagnostics, bayesbind_diagnostics
├── contamination score: contamination_score, weighted_contamination
├── baselines: contamination_nn, source_only_diagnostics
├── splits: split_generator, split_comparison
├── reporting: decile_worst_group, final_figures, final_figures_v2,
│              final_tables, metrics
├── enrichments: pdbbind_cluster_proteins, pdbbind_chembl_target_match,
│                pocket_cluster, target_confirmed_provenance, timebin
├── shared: chem, io
└── model_eval/ (__init__, common, prepare_smoke_inputs)
```

## 4. `.gitignore` summary

- All datasets, processed parquets, generated reports, tables, figures,
  logs, and model checkpoints are excluded.
- Per-extension excludes for `.zip / .tar.gz / .parquet / .pt / .pth /
  .ckpt / .lmdb / .csv / .png / .pdf` etc.
- All Python / pytest / mypy / ipynb checkpoint caches excluded.
- `.gitkeep` exceptions preserve the `data/raw/`, `data/processed/`, and
  `outputs/` empty placeholder dirs after a fresh clone.

## 5. Linux reproduction quick recipe

```bash
git clone https://github.com/kongwoang/VS-LeakKG.git
cd VS-LeakKG
conda env create -f environment.yml
conda activate vsleakkg
pip install -e .
conda install -y -c bioconda mmseqs2
# place raw archives under data/raw/ per data/MANIFEST.template.md
cp data/MANIFEST.template.md data/MANIFEST.md   # then fill in
bash scripts/setup_data.sh
bash scripts/overnight_run.sh
bash scripts/run_mvp_audit.sh
```

Full instructions in `README.md`.

## 6. Confirmation: no data or results committed

- `git ls-files` lists **70 files**; none exceed 1 MB.
- No `.parquet`, `.tar.gz`, `.zip`, `.pt`, `.pth`, `.ckpt`, `.lmdb`,
  `.npy`, `.npz`, `.db`, `.sqlite`, `.fasta`, `.tsv`, `.csv`, `.png`,
  `.pdf` files are tracked.
- The only tracked CSV/PDF/PNG files allowed by `.gitignore` are under
  `docs/` and `configs/` (currently empty); no such files exist.
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, and `outputs/.gitkeep`
  are the only files under those dirs.

## 7. Commit + push

Single clean commit "Clean reproducible VS-LeakKG codebase" pushed to
`https://github.com/kongwoang/VS-LeakKG.git` (`main` branch). No force-push.
