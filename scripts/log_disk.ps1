param(
    [Parameter(Mandatory=$true)][string]$Event,
    [Parameter(Mandatory=$true)][string]$Target,
    [string]$LogFile = "D:\hoangpc\VS-LeakKG\outputs\logs\disk_usage.log",
    [string]$ProjectRoot = "D:\hoangpc\VS-LeakKG"
)

$ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
$cwd = (Get-Location).Path

$lines = @()
$lines += "==== $ts ===="
$lines += "event: $Event"
$lines += "target: $Target"
$lines += "cwd: $cwd"

# df -h equivalent: Get-PSDrive for filesystem providers
$lines += "-- drives (PowerShell Get-PSDrive, GB) --"
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    $used = if ($_.Used) { [math]::Round($_.Used/1GB,2) } else { 0 }
    $free = if ($_.Free) { [math]::Round($_.Free/1GB,2) } else { 0 }
    $lines += ("  {0}: used={1}GB free={2}GB root={3}" -f $_.Name,$used,$free,$_.Root)
}

# du -sh equivalent for project root
if (Test-Path $ProjectRoot) {
    try {
        $size = (Get-ChildItem $ProjectRoot -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        if (-not $size) { $size = 0 }
        $lines += ("-- project size: {0} MB ({1})" -f ([math]::Round($size/1MB,2)), $ProjectRoot)
    } catch {
        $lines += "-- project size: <error: $_>"
    }
} else {
    $lines += "-- project size: <root missing>"
}

# lsblk / free -h: unavailable on Windows. Record stubs.
$lines += "-- lsblk: unavailable on Windows"
$lines += "-- free -h: unavailable on Windows"
try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($os) {
        $totMB = [math]::Round($os.TotalVisibleMemorySize/1024,0)
        $freeMB = [math]::Round($os.FreePhysicalMemory/1024,0)
        $lines += ("-- memory (Win32_OperatingSystem): total={0}MB free={1}MB" -f $totMB,$freeMB)
    }
} catch { }

$lines += ""
$dir = Split-Path $LogFile -Parent
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
Add-Content -Path $LogFile -Value $lines -Encoding utf8
