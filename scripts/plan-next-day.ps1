param(
  [Parameter(Mandatory = $false)]
  [string]$CsvPath,

  [Parameter(Mandatory = $false)]
  [string]$CsvText,

  [Parameter(Mandatory = $false)]
  [string]$DefaultSymbol = "BANKNIFTY",

  [Parameter(Mandatory = $false)]
  [int]$GapThresholdPoints = 30,

  [Parameter(Mandatory = $false)]
  [int]$FlatThresholdPoints = 15,

  [Parameter(Mandatory = $false)]
  [int]$StopBufferPoints = 15,

  [Parameter(Mandatory = $false)]
  [int]$TargetStepPoints = 40,

  [Parameter(Mandatory = $false)]
  [int]$SwingWindow = 2,

  [Parameter(Mandatory = $false)]
  [int]$ClusterBucketPoints = 5,

  [Parameter(Mandatory = $false)]
  [int]$TopClusters = 3
  ,
  [Parameter(Mandatory = $false)]
  [int]$VolumeBucketPoints = 10,

  [Parameter(Mandatory = $false)]
  [int]$TopVolumeClusters = 5
  ,
  [Parameter(Mandatory = $false)]
  [string]$LastHourStart = "14:30"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CsvPath) -and [string]::IsNullOrWhiteSpace($CsvText)) {
  throw "Provide either -CsvPath or -CsvText (paste raw CSV lines)."
}
if (-not [string]::IsNullOrWhiteSpace($CsvPath) -and -not [string]::IsNullOrWhiteSpace($CsvText)) {
  throw "Provide only one: -CsvPath or -CsvText (not both)."
}
if (-not [string]::IsNullOrWhiteSpace($CsvPath) -and -not (Test-Path -LiteralPath $CsvPath)) {
  throw "CSV not found: $CsvPath"
}

function Parse-CandleLine([string]$line) {
  # Accepted formats:
  # 1) SYMBOL,YYYY-MM-DD,HH:MM,open,high,low,close,volume
  # 2) YYYY-MM-DD,HH:MM,open,high,low,close,volume   (symbol is inferred)
  $parts = $line.Split(",")
  if ($parts.Length -lt 7) { return $null }

  $dateRegex = '^\d{4}-\d{2}-\d{2}$'
  $symbol = $null
  $date = $null
  $time = $null
  $o = $null
  $h = $null
  $l = $null
  $c = $null
  $vol = $null

  if ($parts[0].Trim() -match $dateRegex) {
    # YYYY-MM-DD,HH:MM,open,high,low,close,volume
    $symbol = $DefaultSymbol
    $date = $parts[0].Trim()
    $time = $parts[1].Trim()
    $o = $parts[2]
    $h = $parts[3]
    $l = $parts[4]
    $c = $parts[5]
    $vol = if ($parts.Length -ge 7) { $parts[6] } else { "0" }
  } else {
    # SYMBOL,YYYY-MM-DD,HH:MM,open,high,low,close,volume
    $symbol = $parts[0].Trim()
    $date = $parts[1].Trim()
    $time = $parts[2].Trim()
    $o = $parts[3]
    $h = $parts[4]
    $l = $parts[5]
    $c = $parts[6]
    $vol = if ($parts.Length -ge 8) { $parts[7] } else { "0" }
  }

  $dt = [datetime]::ParseExact("$date $time", "yyyy-MM-dd HH:mm", $null)

  [pscustomobject]@{
    Symbol = $symbol
    Dt = $dt
    Open = [double]$o
    High = [double]$h
    Low = [double]$l
    Close = [double]$c
    Volume = [double]$vol
  }
}

$rawLines =
  if (-not [string]::IsNullOrWhiteSpace($CsvText)) { $CsvText -split "(\r\n|\n|\r)" } else { Get-Content -LiteralPath $CsvPath }
$candles = New-Object System.Collections.Generic.List[object]
foreach ($line in $rawLines) {
  if ([string]::IsNullOrWhiteSpace($line)) { continue }
  if ($line.TrimStart().StartsWith("#")) { continue }
  if ($line -match "^(SYMBOL|NIFTY)\b" -and $line -match ",Date," ) { continue }
  if ($line -match "^(Date|DATE)\b" -and $line -match ",Time," ) { continue }

  $c = Parse-CandleLine $line
  if ($null -ne $c) { $candles.Add($c) }
}

if ($candles.Count -lt 30) {
  throw "Need at least ~30 candles for robust planning. Got $($candles.Count)."
}

$candles = $candles | Sort-Object Dt
$symbol = $candles[0].Symbol
$dateStr = $candles[0].Dt.ToString("yyyy-MM-dd")

$dayOpen = $candles[0].Open
$dayClose = $candles[-1].Close
$dayHigh = ($candles | Measure-Object -Property High -Maximum).Maximum
$dayLow = ($candles | Measure-Object -Property Low -Minimum).Minimum
$range = [math]::Max(0.01, ($dayHigh - $dayLow))

$endTime = $candles[-1].Dt
$start30 = $endTime.AddMinutes(-30)
$last30 = $candles | Where-Object { $_.Dt -ge $start30 }
if ($last30.Count -lt 5) { $last30 = $candles | Select-Object -Last 30 }

$last30StartClose = $last30[0].Close
$last30EndClose = $last30[-1].Close
$last30Delta = $last30EndClose - $last30StartClose

function Round-ToBucket([double]$price, [int]$bucket) {
  return [math]::Round($price / $bucket) * $bucket
}

function Get-RoundLevels([double]$anchor) {
  # Use 50-pt grid by default (common for NIFTY), keep it simple and explicit.
  $step = 50
  $base = [math]::Round($anchor / $step) * $step
  [pscustomobject]@{
    Step = $step
    Base = $base
    Up = $base + $step
    Down = $base - $step
  }
}

$round = Get-RoundLevels $dayClose

function Get-Swings($items, [int]$w) {
  $highs = New-Object System.Collections.Generic.List[double]
  $lows = New-Object System.Collections.Generic.List[double]
  for ($i = $w; $i -le ($items.Count - 1 - $w); $i++) {
    $h = $items[$i].High
    $l = $items[$i].Low

    $isSwingHigh = $true
    $isSwingLow = $true
    for ($j = 1; $j -le $w; $j++) {
      if ($h -le $items[$i - $j].High) { $isSwingHigh = $false }
      if ($h -lt $items[$i + $j].High) { $isSwingHigh = $false }

      if ($l -ge $items[$i - $j].Low) { $isSwingLow = $false }
      if ($l -gt $items[$i + $j].Low) { $isSwingLow = $false }
    }
    if ($isSwingHigh) { $highs.Add($h) }
    if ($isSwingLow) { $lows.Add($l) }
  }
  [pscustomobject]@{ Highs = $highs; Lows = $lows }
}

function Get-Clusters([System.Collections.Generic.List[double]]$prices, [int]$bucket, [int]$topN) {
  $counts = @{}
  foreach ($p in $prices) {
    $b = Round-ToBucket $p $bucket
    $k = $b.ToString("F0")
    if (-not $counts.ContainsKey($k)) { $counts[$k] = 0 }
    $counts[$k]++
  }
  $counts.GetEnumerator() |
    Sort-Object -Property Value -Descending |
    Select-Object -First $topN |
    ForEach-Object { [double]$_.Key }
}

$sw = Get-Swings $candles $SwingWindow
$swingHighClusters = Get-Clusters $sw.Highs $ClusterBucketPoints $TopClusters
$swingLowClusters = Get-Clusters $sw.Lows $ClusterBucketPoints $TopClusters

function Get-VolumeClusters($items, [int]$bucket, [int]$topN) {
  # Proxy volume-by-price using typical price (HLC3) binned into buckets.
  $vol = @{}
  foreach ($c in $items) {
    $tp = ($c.High + $c.Low + $c.Close) / 3.0
    $b = Round-ToBucket $tp $bucket
    $k = $b.ToString("F0")
    if (-not $vol.ContainsKey($k)) { $vol[$k] = 0.0 }
    $vol[$k] += [double]$c.Volume
  }
  $vol.GetEnumerator() |
    Sort-Object -Property Value -Descending |
    Select-Object -First $topN |
    ForEach-Object { [double]$_.Key }
}

$volumeClusters = Get-VolumeClusters $candles $VolumeBucketPoints $TopVolumeClusters

function F([double]$x) { return $x.ToString("F2") }

$pdh = $dayHigh
$pdl = $dayLow

$lastHourStartDt = [datetime]::ParseExact("$dateStr $LastHourStart", "yyyy-MM-dd HH:mm", $null)
$lastHour = $candles | Where-Object { $_.Dt -ge $lastHourStartDt }
$lastHour = @($lastHour)
if ($lastHour.Count -lt 1) {
  $lastHour = if ($candles.Count -ge 60) { @($candles | Select-Object -Last 60) } else { @($candles) }
}
$lastHourHigh = ($lastHour | Measure-Object -Property High -Maximum).Maximum
$lastHourLow = ($lastHour | Measure-Object -Property Low -Minimum).Minimum

Write-Host ""
Write-Host "Nexus Ultra next-day planner (educational/backtest use; not financial advice)"
Write-Host "Symbol: $symbol  Date: $dateStr"
Write-Host ""
Write-Host "Key Levels"
Write-Host ("- PDH:  {0}" -f (F $pdh))
Write-Host ("- PDL:  {0}" -f (F $pdl))
Write-Host ("- LastHourHigh({0}+): {1}" -f $LastHourStart, (F $lastHourHigh))
Write-Host ("- LastHourLow({0}+):  {1}" -f $LastHourStart, (F $lastHourLow))
Write-Host ("- Open: {0}" -f (F $dayOpen))
Write-Host ("- Close:{0}" -f (F $dayClose))
Write-Host ("- Range:{0}" -f (F $range))
Write-Host ("- Round({0}): base {1} | up {2} | down {3}" -f $round.Step, (F $round.Base), (F $round.Up), (F $round.Down))
Write-Host ("- Last30 delta (close): {0}" -f (F $last30Delta))
Write-Host ("- Swing-high clusters: {0}" -f (($swingHighClusters | ForEach-Object { F $_ }) -join ", "))
Write-Host ("- Swing-low clusters:  {0}" -f (($swingLowClusters | ForEach-Object { F $_ }) -join ", "))
Write-Host ("- High-volume clusters (bucket {0}, HLC3 proxy): {1}" -f $VolumeBucketPoints, (($volumeClusters | ForEach-Object { F $_ }) -join ", "))

$buyersSL = @($pdl, $lastHourLow) + @($swingLowClusters)
$sellersSL = @($pdh, $lastHourHigh) + @($swingHighClusters)
Write-Host ("- Buyer SL pools (below):  {0}" -f (($buyersSL | Sort-Object -Unique | ForEach-Object { F $_ }) -join ", "))
Write-Host ("- Seller SL pools (above): {0}" -f (($sellersSL | Sort-Object -Unique | ForEach-Object { F $_ }) -join ", "))

Write-Host ""
Write-Host "Scenario Plans (next session open vs prev close)"
Write-Host ("Thresholds: gap>= {0} pts | flat<= {1} pts | stop buffer {2} pts | target step {3} pts" -f $GapThresholdPoints, $FlatThresholdPoints, $StopBufferPoints, $TargetStepPoints)
Write-Host ""

# GAP UP PLAN (relative to prev close)
Write-Host "1) GAP UP"
Write-Host ("- Hypothesis: comfort risk above close; prefer SHORT only after rejection/failed acceptance near resistance (PDH/cluster/round).")
$gapUpShortTrigger = $pdh
$gapUpSL = $pdh + $StopBufferPoints
$gapUpT1 = $dayClose
$gapUpT2 = $pdl
$gapUpT3 = $pdl - $TargetStepPoints
Write-Host ("- Short trigger level: {0} (rejection/fail then back below)" -f (F $gapUpShortTrigger))
Write-Host ("- SL: {0}" -f (F $gapUpSL))
Write-Host ("- Targets: T1 {0} | T2 {1} | T3 {2}" -f (F $gapUpT1), (F $gapUpT2), (F $gapUpT3))
Write-Host ("- Invalidation: if accepts above {0} with continuation, skip contrarian and wait (DT-SL-073/103)." -f (F $gapUpSL))

# GAP DOWN PLAN
Write-Host ""
Write-Host "2) GAP DOWN"
Write-Host ("- Hypothesis: do NOT sell immediately; wait for pullback to create liquidity then sell rejection (DT-SL-024).")
$gapDownPullbackSell = [math]::Min($dayClose, $pdl + $FlatThresholdPoints)
$gapDownSL = $gapDownPullbackSell + $StopBufferPoints
$gapDownT1 = $pdl
$gapDownT2 = $pdl - $TargetStepPoints
$gapDownT3 = $pdl - (2 * $TargetStepPoints)
Write-Host ("- Pullback sell zone anchor: {0} (sell rejection/failed reclaim)" -f (F $gapDownPullbackSell))
Write-Host ("- SL: {0}" -f (F $gapDownSL))
Write-Host ("- Targets: T1 {0} | T2 {1} | T3 {2}" -f (F $gapDownT1), (F $gapDownT2), (F $gapDownT3))

# FLAT PLAN
Write-Host ""
Write-Host "3) FLAT OPEN (near prev close)"
Write-Host ("- Hypothesis: let range edges decide; trade only after confirmation (DT-SL-073) and microstructure validation (DT-SL-122/123).")
$flatLongTrigger = $pdh
$flatLongSL = $pdh - $StopBufferPoints
$flatLongT1 = $pdh + $TargetStepPoints
$flatShortTrigger = $pdl
$flatShortSL = $pdl + $StopBufferPoints
$flatShortT1 = $pdl - $TargetStepPoints
Write-Host ("- Long trigger: above {0} (acceptance) | SL {1} | T1 {2}" -f (F $flatLongTrigger), (F $flatLongSL), (F $flatLongT1))
Write-Host ("- Short trigger: below {0} (acceptance) | SL {1} | T1 {2}" -f (F $flatShortTrigger), (F $flatShortSL), (F $flatShortT1))

Write-Host ""
Write-Host "Notes"
Write-Host "- Use 5/15m for bias + 1m for execution (DT-SL-121)."
Write-Host "- Comfort engine: avoid chasing obvious alignment; wait for trap confirmation (DT-SL-103..110)."
Write-Host "- Operator exit signatures can override holding (DT-SL-086/093/111)."
Write-Host "- SL clusters indicate probability, not timing: allow delay + trap-first paths (DT-SL-035/036/038)."
Write-Host "- Execution discipline: max 1-2 attempts per idea; stop early on mismatch/whipsaw days (DT-SL-039/049)."
Write-Host "- Validity filter: SL hunting is weaker on event/low participation/high profit-holder days (DT-SL-094..102)."
Write-Host "- Target discipline: consider first capture near ~1R; runners only with holding evidence (DT-SL-067/088)."
Write-Host "- Fake confirmations happen: keep SL strict and treat them as cost (DT-SL-188)."
