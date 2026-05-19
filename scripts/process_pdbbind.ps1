#requires -Version 5.1
<#
VS-LeakKG PDBBind processing wrapper.
Idempotent — re-runs hit the parquet caches.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ROOT  = 'D:\hoangpc\VS-LeakKG'
$LOG   = Join-Path $ROOT 'outputs\logs\pdbbind_processing.log'
$PY    = if ($env:VSLEAKKG_PYTHON) { $env:VSLEAKKG_PYTHON } else { 'python' }
New-Item -ItemType Directory -Path (Split-Path $LOG) -Force | Out-Null

Push-Location (Join-Path $ROOT 'src')
try {
    & $PY -m vsleakkg.run_pdbbind @args 2>&1 | Tee-Object -FilePath $LOG -Append
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
