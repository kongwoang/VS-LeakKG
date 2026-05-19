#!/usr/bin/env bash
# Restore data/raw/ from the dataset archive VS-LeakKG_raw_datasets_<DATE>.zip.
#
# Usage:
#   bash scripts/extract_datasets.sh /path/to/VS-LeakKG_raw_datasets_YYYYMMDD.zip
#
# What it does:
#   1. Unzips the outer archive into a temp staging dir.
#   2. Moves raw/ into data/raw/ (merging with an existing data/raw/ if present).
#   3. Re-extracts every inner archive into the <dataset>/extracted/ layout
#      the pipeline expects (matches the original on-disk structure).
#
# Idempotent: skips any inner archive whose extracted/ target is non-empty.
set -euo pipefail

ZIP="${1:-}"
if [[ -z "${ZIP}" ]] || [[ ! -f "${ZIP}" ]]; then
  echo "usage: bash scripts/extract_datasets.sh /path/to/VS-LeakKG_raw_datasets_YYYYMMDD.zip" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

echo "[1/3] Unzipping outer archive → ${STAGE}"
unzip -q -o "${ZIP}" -d "${STAGE}"

echo "[2/3] Merging raw/ into ${REPO_ROOT}/data/raw/"
mkdir -p "${REPO_ROOT}/data/raw"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${STAGE}/raw/" "${REPO_ROOT}/data/raw/"
else
  cp -a "${STAGE}/raw/." "${REPO_ROOT}/data/raw/"
fi

# Also surface the proposal PDF and run-specific manifest at the repo root
# (these never go into Git — .gitignore excludes them).
[[ -f "${STAGE}/VS_LeakKG_proposal.pdf" ]] && cp -n "${STAGE}/VS_LeakKG_proposal.pdf" "${REPO_ROOT}/" || true
[[ -f "${STAGE}/data_MANIFEST_run_specific.md" ]] && cp -n "${STAGE}/data_MANIFEST_run_specific.md" "${REPO_ROOT}/data/MANIFEST.run_specific.md" || true

ROOT="${REPO_ROOT}/data/raw"

extract_tar() {
  local archive="$1" target="$2"
  if [[ -f "${archive}" ]]; then
    if [[ ! -d "${target}" ]] || [[ -z "$(ls -A "${target}" 2>/dev/null)" ]]; then
      echo "  tar → ${target}"
      mkdir -p "${target}"
      tar -xf "${archive}" -C "${target}"
    else
      echo "  skip (already extracted): ${target}"
    fi
  fi
}

extract_zip() {
  local archive="$1" target="$2"
  if [[ -f "${archive}" ]]; then
    if [[ ! -d "${target}" ]] || [[ -z "$(ls -A "${target}" 2>/dev/null)" ]]; then
      echo "  zip → ${target}"
      mkdir -p "${target}"
      unzip -q "${archive}" -d "${target}"
    else
      echo "  skip (already extracted): ${target}"
    fi
  fi
}

echo "[3/3] Extracting inner archives"
extract_tar "${ROOT}/ChEMBL/chembl_35_sqlite.tar.gz"          "${ROOT}/ChEMBL/extracted"
extract_zip "${ROOT}/BindingDB/BindingDB_All_202605_tsv.zip"  "${ROOT}/BindingDB/extracted"
extract_tar "${ROOT}/PBDBind/P-L.tar.gz"                      "${ROOT}/PBDBind/extracted"
extract_tar "${ROOT}/PBDBind/index.tar.gz"                    "${ROOT}/PBDBind/extracted"
extract_tar "${ROOT}/LIT-PCBA/full_data.tgz"                  "${ROOT}/LIT-PCBA/extracted"
extract_tar "${ROOT}/BayesBind/BayesBindV1.5.tar.gz"          "${ROOT}/BayesBind/extracted"
extract_zip "${ROOT}/DEKOIS/DEKOIS2.zip"                      "${ROOT}/DEKOIS/extracted"
extract_tar "${ROOT}/BigBind/BigBindV1.5.tar.gz"              "${ROOT}/BigBind/extracted"

echo "Done. data/raw/ is ready. Next:"
echo "  bash scripts/overnight_run.sh"
echo "  bash scripts/run_mvp_audit.sh"
echo "  bash scripts/run_mvp1_audit.sh"
