#!/usr/bin/env bash
# Run the VS-LeakKG MVP audit pipeline.
#
# Logs land in:
#   outputs/logs/mvp_audit.log           — Python log of every step
#   outputs/logs/mvp_audit_disk_usage.log — pre/post disk snapshot per step
#
# Re-runs are safe: per-task parquet outputs are overwritten in place, and
# missing-dataset tasks short-circuit to a "missing_*.md" report under
# outputs/reports/ instead of failing the pipeline.

set -euo pipefail

VSLEAKKG_ROOT="${VSLEAKKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"

LOG_DIR="$VSLEAKKG_ROOT/outputs/logs"
DISK_LOG="$LOG_DIR/mvp_audit_disk_usage.log"
RUN_LOG="$LOG_DIR/mvp_audit.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

check_disk() {
    local need="${1:-$MIN_FREE_GB}"
    local label="${2:-(unspecified)}"
    local free_kb
    free_kb=$(df -Pk "$VSLEAKKG_ROOT" | awk 'NR==2 {print $4}')
    local free_gb=$(( free_kb / 1024 / 1024 ))
    echo "[$(ts)] check_disk: free=${free_gb}GB need>=${need}GB label='$label'"
    if [ "$free_gb" -lt "$need" ]; then
        echo "[$(ts)] check_disk: NOT ENOUGH FREE SPACE — aborting before '$label'" >&2
        return 2
    fi
}

log_disk() {
    local event="${1:-event}"
    local target="${2:-(unspecified)}"
    {
        echo "==== $(ts) ===="
        echo "event: $event"
        echo "target: $target"
        echo "cwd: $(pwd)"
        echo "-- df -h --"
        df -h "$VSLEAKKG_ROOT" 2>/dev/null || true
        echo "-- du -sh project --"
        du -sh "$VSLEAKKG_ROOT" 2>/dev/null || true
        if command -v lsblk >/dev/null 2>&1; then echo "-- lsblk --"; lsblk 2>/dev/null || true; fi
        if command -v free  >/dev/null 2>&1; then echo "-- free -h --"; free  -h 2>/dev/null || true; fi
        echo ""
    } >> "$DISK_LOG"
}

echo "[$(ts)] MVP audit starting at $VSLEAKKG_ROOT"
log_disk "mvp_audit_start" "VS-LeakKG"
check_disk 10 "mvp_audit_start"

cd "$VSLEAKKG_ROOT/src"
"$PYTHON" -m vsleakkg.run_mvp_audit "$@"
status=$?

log_disk "mvp_audit_end" "VS-LeakKG"
echo "[$(ts)] MVP audit exit code: $status"
exit $status
