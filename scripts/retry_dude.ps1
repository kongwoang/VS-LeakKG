param(
    [string]$Root = "D:\hoangpc\VS-LeakKG\data\raw\DUD-E"
)
$manPath = Join-Path $Root '_download_manifest.csv'
$rows = Import-Csv $manPath
$retries = @()
foreach ($r in $rows) {
    if ($r.status -eq 'ok' -or $r.status -eq 'exists') { $retries += $r; continue }
    $tdir = Join-Path $Root $r.target
    $out = Join-Path $tdir $r.file
    $url = "https://dude.docking.org/targets/$($r.target)/$($r.file)"
    # Add a User-Agent and a longer retry budget this pass.
    & curl.exe -L -sS -f --retry 5 --retry-delay 3 --connect-timeout 60 `
        --max-time 120 -A 'Mozilla/5.0 (compatible; vsleakkg)' `
        -o $out $url 2>$null
    $code = $LASTEXITCODE
    if ($code -eq 0 -and (Test-Path $out)) {
        $sz = (Get-Item $out).Length
        if ($sz -gt 0) {
            $retries += [pscustomobject]@{ target=$r.target; file=$r.file; status='ok_retry'; bytes=$sz }
            continue
        }
    }
    $retries += [pscustomobject]@{ target=$r.target; file=$r.file; status="curl_fail_$code"; bytes=0 }
}
$retries | Export-Csv -Path $manPath -NoTypeInformation -Encoding utf8
$retries | Group-Object status | Select-Object Count,Name | Format-Table -AutoSize | Out-String
"total_MB=" + [math]::Round(((Get-ChildItem $Root -Recurse -File -Filter '*.ism' | Measure-Object -Property Length -Sum).Sum/1MB),2)
