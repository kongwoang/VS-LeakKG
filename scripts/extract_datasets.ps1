#requires -Version 5.1
<#
.SYNOPSIS
    Restore data/raw/ from the dataset archive VS-LeakKG_raw_datasets_<DATE>.zip.

.DESCRIPTION
    1. Unzips the outer archive into a temp staging dir.
    2. Moves raw/ into <repo>/data/raw/ (merging if data/raw/ already exists).
    3. Re-extracts every inner archive into <dataset>\extracted\ — the layout
       the pipeline expects.

    Idempotent: skips any inner archive whose extracted/ target is non-empty.

.EXAMPLE
    .\scripts\extract_datasets.ps1 -Zip "D:\hoangpc\VS-LeakKG_backups\VS-LeakKG_raw_datasets_20260519.zip"
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Zip
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Zip)) {
    throw "Zip not found: $Zip"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage    = Join-Path ([System.IO.Path]::GetTempPath()) ("vsleakkg_stage_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

try {
    Write-Host "[1/3] Unzipping outer archive -> $Stage"
    Expand-Archive -LiteralPath $Zip -DestinationPath $Stage -Force

    Write-Host "[2/3] Merging raw\ into $RepoRoot\data\raw\"
    $TargetRoot = Join-Path $RepoRoot "data\raw"
    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
    Get-ChildItem -Force -Path (Join-Path $Stage "raw") | ForEach-Object {
        $dest = Join-Path $TargetRoot $_.Name
        if (Test-Path $dest) {
            # merge: copy contents recursively
            Copy-Item -Path (Join-Path $_.FullName "*") -Destination $dest -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Move-Item -LiteralPath $_.FullName -Destination $dest -Force
        }
    }

    $proposal = Join-Path $Stage "VS_LeakKG_proposal.pdf"
    if (Test-Path -LiteralPath $proposal) {
        Copy-Item -LiteralPath $proposal -Destination (Join-Path $RepoRoot "VS_LeakKG_proposal.pdf") -Force
    }
    $runManifest = Join-Path $Stage "data_MANIFEST_run_specific.md"
    if (Test-Path -LiteralPath $runManifest) {
        Copy-Item -LiteralPath $runManifest -Destination (Join-Path $RepoRoot "data\MANIFEST.run_specific.md") -Force
    }

    function Extract-Tar($Archive, $Target) {
        if (-not (Test-Path -LiteralPath $Archive)) { return }
        if ((Test-Path -LiteralPath $Target) -and (@(Get-ChildItem -Force -LiteralPath $Target -ErrorAction SilentlyContinue).Count -gt 0)) {
            Write-Host "  skip (already extracted): $Target"
            return
        }
        Write-Host "  tar -> $Target"
        New-Item -ItemType Directory -Force -Path $Target | Out-Null
        & tar -xf $Archive -C $Target
        if ($LASTEXITCODE -ne 0) { throw "tar failed for $Archive" }
    }

    function Extract-Zip($Archive, $Target) {
        if (-not (Test-Path -LiteralPath $Archive)) { return }
        if ((Test-Path -LiteralPath $Target) -and (@(Get-ChildItem -Force -LiteralPath $Target -ErrorAction SilentlyContinue).Count -gt 0)) {
            Write-Host "  skip (already extracted): $Target"
            return
        }
        Write-Host "  zip -> $Target"
        New-Item -ItemType Directory -Force -Path $Target | Out-Null
        Expand-Archive -LiteralPath $Archive -DestinationPath $Target -Force
    }

    Write-Host "[3/3] Extracting inner archives"
    Extract-Tar (Join-Path $TargetRoot "ChEMBL\chembl_35_sqlite.tar.gz")             (Join-Path $TargetRoot "ChEMBL\extracted")
    Extract-Zip (Join-Path $TargetRoot "BindingDB\BindingDB_All_202605_tsv.zip")     (Join-Path $TargetRoot "BindingDB\extracted")
    Extract-Tar (Join-Path $TargetRoot "PBDBind\P-L.tar.gz")                         (Join-Path $TargetRoot "PBDBind\extracted")
    Extract-Tar (Join-Path $TargetRoot "PBDBind\index.tar.gz")                       (Join-Path $TargetRoot "PBDBind\extracted")
    Extract-Tar (Join-Path $TargetRoot "LIT-PCBA\full_data.tgz")                     (Join-Path $TargetRoot "LIT-PCBA\extracted")
    Extract-Tar (Join-Path $TargetRoot "BayesBind\BayesBindV1.5.tar.gz")             (Join-Path $TargetRoot "BayesBind\extracted")
    Extract-Zip (Join-Path $TargetRoot "DEKOIS\DEKOIS2.zip")                         (Join-Path $TargetRoot "DEKOIS\extracted")
    Extract-Tar (Join-Path $TargetRoot "BigBind\BigBindV1.5.tar.gz")                 (Join-Path $TargetRoot "BigBind\extracted")

    Write-Host ""
    Write-Host "Done. data\raw\ is ready. Next:"
    Write-Host "  python -m vsleakkg.run_overnight"
    Write-Host "  python -m vsleakkg.run_mvp_audit"
    Write-Host "  python -m vsleakkg.run_mvp1_audit"
}
finally {
    if (Test-Path -LiteralPath $Stage) {
        Remove-Item -Recurse -Force -LiteralPath $Stage -ErrorAction SilentlyContinue
    }
}
