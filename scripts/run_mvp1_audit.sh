#!/usr/bin/env bash
# Run the VS-LeakKG MVP-1 audit pipeline (real LIT-PCBA AVE splits + DEKOIS).
set -euo pipefail

VSLEAKKG_ROOT="${VSLEAKKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"

LOG_DIR="$VSLEAKKG_ROOT/outputs/logs"
RUN_LOG="$LOG_DIR/mvp1_audit.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(ts)] MVP-1 audit starting at $VSLEAKKG_ROOT"

cd "$VSLEAKKG_ROOT/src"
"$PYTHON" -m vsleakkg.run_mvp1_audit "$@"
status=$?

echo "[$(ts)] MVP-1 audit exit code: $status"
exit $status
