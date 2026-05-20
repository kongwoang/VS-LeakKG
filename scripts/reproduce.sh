#!/usr/bin/env bash
# End-to-end reproduction of VS-LeakKG, including the dataset download.
#
# Usage:
#   # Easiest — download from HF and run everything:
#   export HF_TOKEN=hf_...
#   bash scripts/reproduce.sh
#
#   # Or point at an existing zip on disk:
#   bash scripts/reproduce.sh /path/to/<dataset>.zip
#
# Stages:
#   0. Sanity-check the environment (python, tar, unzip, vsleakkg package).
#   1. Fetch the dataset archive from Hugging Face (skipped if already cached).
#   2. Restore data/raw/ from the archive.
#   3. Build the heterogeneous provenance graph (overnight orchestrator).
#   4. Run the MVP + MVP1 audit passes.
#   5. Print where the produced artefacts live.
#
# Re-running is safe: every stage is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=scripts/dataset_version.sh
. "${REPO_ROOT}/scripts/dataset_version.sh"

ZIP_ARG="${1:-}"

echo "==[0/5] Environment check =="
for tool in python tar unzip; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "missing required tool: ${tool}" >&2
    exit 3
  fi
done
if ! command -v mmseqs >/dev/null 2>&1; then
  echo "warning: mmseqs2 not on PATH - PDBBind/ChEMBL clustering will be skipped."
fi
python -c "import vsleakkg" 2>/dev/null || {
  echo "vsleakkg package not importable; run: pip install -e ." >&2
  exit 4
}

echo
echo "==[1/5] Dataset archive =="
if [[ -n "${ZIP_ARG}" ]]; then
  if [[ ! -f "${ZIP_ARG}" ]]; then
    echo "Provided zip not found: ${ZIP_ARG}" >&2
    exit 5
  fi
  ZIP="${ZIP_ARG}"
  echo "Using provided archive: ${ZIP}"
else
  ZIP="${DATASET_CACHE_DIR}/${DATASET_ZIP}"
  if [[ ! -f "${ZIP}" ]]; then
    echo "No cached archive at ${ZIP} - fetching from Hugging Face."
    bash "${REPO_ROOT}/scripts/fetch_dataset.sh"
  else
    echo "Using cached archive: ${ZIP}"
  fi
fi

echo
echo "==[2/5] Restoring data/raw/ =="
bash "${REPO_ROOT}/scripts/extract_datasets.sh" "${ZIP}"

echo
echo "==[3/5] Building the provenance graph =="
bash "${REPO_ROOT}/scripts/overnight_run.sh"

echo
echo "==[4/5] Running audit passes =="
bash "${REPO_ROOT}/scripts/run_mvp_audit.sh"
bash "${REPO_ROOT}/scripts/run_mvp1_audit.sh"

echo
echo "==[5/5] Done. Artefacts =="
echo "  data/processed/   parquet shards (graph + per-dataset loaders)"
echo "  outputs/reports/  Markdown reports"
echo "  outputs/tables/   Tables 1-6"
echo "  outputs/figures/  Figures 1-4"
