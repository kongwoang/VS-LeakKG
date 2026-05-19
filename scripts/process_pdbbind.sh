#!/usr/bin/env bash
set -euo pipefail
VSLEAKKG_ROOT="${VSLEAKKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
LOG="$VSLEAKKG_ROOT/outputs/logs/pdbbind_processing.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PDBBind processing starting at $VSLEAKKG_ROOT"
cd "$VSLEAKKG_ROOT/src"
"$PYTHON" -m vsleakkg.run_pdbbind "$@"
