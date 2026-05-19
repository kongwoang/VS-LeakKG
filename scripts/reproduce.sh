#!/usr/bin/env bash
# End-to-end reproduction of VS-LeakKG from the dataset archive.
#
# Usage:
#   bash scripts/reproduce.sh /path/to/VS-LeakKG_raw_datasets_YYYYMMDD.zip
#
# Stages:
#   0. Sanity-check the environment (python, mmseqs, tar, unzip).
#   1. Restore data/raw/ from the dataset .zip.
#   2. Build the heterogeneous provenance graph (overnight orchestrator).
#   3. Run the MVP + MVP1 audit passes.
#   4. Print where the produced artefacts live.
#
# Re-running is safe: extraction is idempotent and each audit step writes to
# a fresh sub-directory under outputs/.
set -euo pipefail

ZIP="${1:-}"
if [[ -z "${ZIP}" ]] || [[ ! -f "${ZIP}" ]]; then
  echo "usage: bash scripts/reproduce.sh /path/to/VS-LeakKG_raw_datasets_YYYYMMDD.zip" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "==[0/4] Environment check =="
for tool in python tar unzip; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "missing required tool: ${tool}" >&2
    exit 3
  fi
done
if ! command -v mmseqs >/dev/null 2>&1; then
  echo "warning: mmseqs2 not on PATH — PDBBind/ChEMBL clustering will be skipped."
fi
python -c "import vsleakkg" 2>/dev/null || {
  echo "vsleakkg package not importable; run: pip install -e ." >&2
  exit 4
}

echo
echo "==[1/4] Restoring data/raw/ from ${ZIP} =="
bash scripts/extract_datasets.sh "${ZIP}"

echo
echo "==[2/4] Building the provenance graph =="
bash scripts/overnight_run.sh

echo
echo "==[3/4] Running audit passes =="
bash scripts/run_mvp_audit.sh
bash scripts/run_mvp1_audit.sh

echo
echo "==[4/4] Done. Artefacts =="
echo "  data/processed/   parquet shards (graph + per-dataset loaders)"
echo "  outputs/reports/  Markdown reports"
echo "  outputs/tables/   Tables 1–6"
echo "  outputs/figures/  Figures 1–4"
