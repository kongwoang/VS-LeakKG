param(
    [string]$Root = "D:\hoangpc\VS-LeakKG\data\raw\DUD-E",
    [string[]]$Targets
)

if (-not $Targets -or $Targets.Count -eq 0) {
    $Targets = @(
        'aa2ar','abl1','ace','aces','ada','ada17','adrb1','adrb2','akt1','akt2',
        'aldr','ampc','andr','aofb','bace1','braf','cah2','casp3','cdk2','comt',
        'cp2c9','cp3a4','csf1r','cxcr4','def','dhi1','dpp4','drd3','dyr','egfr',
        'esr1','esr2','fa10','fa7','fabp4','fak1','fgfr1','fkb1a','fnta','fpps',
        'gcr','glcm','gria2','grik1','hdac2','hdac8','hivint','hivpr','hivrt','hmdh',
        'hs90a','hxk4','igf1r','inha','ital','jak2','kif11','kit','kith','kpcb',
        'lck','lkha4','mapk2','mcr','met','mk01','mk10','mk14','mmp13','mp2k1',
        'nos1','nram','pa2ga','parp1','pde5a','pgh1','pgh2','plk1','pnph','ppara',
        'ppard','pparg','prgr','ptn1','pur2','pygm','pyrd','reni','rock1','rxra',
        'sahh','src','tgfr1','thb','thrb','try1','tryb1','tysy','urok','vgfr2',
        'wee1','xiap'
    )
}

New-Item -ItemType Directory -Path $Root -Force | Out-Null
$results = @()
foreach ($t in $Targets) {
    $tdir = Join-Path $Root $t
    New-Item -ItemType Directory -Path $tdir -Force | Out-Null
    $files = @('actives_final.ism','decoys_final.ism')
    foreach ($f in $files) {
        $out = Join-Path $tdir $f
        if (Test-Path $out -PathType Leaf) {
            $existing = (Get-Item $out).Length
            if ($existing -gt 0) {
                $results += [pscustomobject]@{ target=$t; file=$f; status='exists'; bytes=$existing }
                continue
            }
        }
        $url = "https://dude.docking.org/targets/$t/$f"
        & curl.exe -L -sS -f --retry 2 --retry-delay 2 --connect-timeout 30 -o $out $url 2>$null
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            $results += [pscustomobject]@{ target=$t; file=$f; status="curl_fail_$code"; bytes=0 }
        } else {
            $sz = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
            $results += [pscustomobject]@{ target=$t; file=$f; status='ok'; bytes=$sz }
        }
    }
}

$results | Export-Csv -Path (Join-Path $Root '_download_manifest.csv') -NoTypeInformation -Encoding utf8
$ok = ($results | Where-Object { $_.status -eq 'ok' -or $_.status -eq 'exists' }).Count
$bad = ($results | Where-Object { $_.status -notin @('ok','exists') }).Count
$total = ($results | Measure-Object -Property bytes -Sum).Sum
"DUD-E: ok/exists=$ok, failed=$bad, total_bytes=$total"
