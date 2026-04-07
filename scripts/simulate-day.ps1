param(
  # Previous session candles (used to compute PDH/PDL/prev close)
  [Parameter(Mandatory = $false)]
  [string]$PrevCsvPath,

  [Parameter(Mandatory = $false)]
  [string]$PrevCsvText,

  # Session to simulate
  [Parameter(Mandatory = $false)]
  [string]$DayCsvPath,

  [Parameter(Mandatory = $false)]
  [string]$DayCsvText,

  [Parameter(Mandatory = $false)]
  [string]$DefaultSymbol = "BANKNIFTY",

  [Parameter(Mandatory = $false)]
  [int]$Quantity = 65,

  [Parameter(Mandatory = $false)]
  [int]$GapThresholdPoints = 30,

  [Parameter(Mandatory = $false)]
  [int]$FlatThresholdPoints = 15,

  [Parameter(Mandatory = $false)]
  [int]$StopBufferPoints = 15,

  [Parameter(Mandatory = $false)]
  [int]$TargetStepPoints = 40,

  [Parameter(Mandatory = $false)]
  [ValidateSet("1R", "T1")]
  [string]$TargetMode = "1R",

  [Parameter(Mandatory = $false)]
  [ValidateSet("stop_first", "target_first")]
  [string]$AmbiguousFillPolicy = "stop_first",

  [Parameter(Mandatory = $false)]
  [ValidateRange(1, 5)]
  [int]$MaxAttemptsPerIdea = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-OneOf([string]$a, [string]$b, [string]$label) {
  $hasA = -not [string]::IsNullOrWhiteSpace($a)
  $hasB = -not [string]::IsNullOrWhiteSpace($b)
  if (-not $hasA -and -not $hasB) { throw "Provide either -$label`CsvPath or -$label`CsvText." }
  if ($hasA -and $hasB) { throw "Provide only one of -$label`CsvPath or -$label`CsvText (not both)." }
}

Ensure-OneOf $PrevCsvPath $PrevCsvText "Prev"
Ensure-OneOf $DayCsvPath $DayCsvText "Day"

if (-not [string]::IsNullOrWhiteSpace($PrevCsvPath) -and -not (Test-Path -LiteralPath $PrevCsvPath)) {
  throw "Prev CSV not found: $PrevCsvPath"
}
if (-not [string]::IsNullOrWhiteSpace($DayCsvPath) -and -not (Test-Path -LiteralPath $DayCsvPath)) {
  throw "Day CSV not found: $DayCsvPath"
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
    $symbol = $DefaultSymbol
    $date = $parts[0].Trim()
    $time = $parts[1].Trim()
    $o = $parts[2]
    $h = $parts[3]
    $l = $parts[4]
    $c = $parts[5]
    $vol = if ($parts.Length -ge 7) { $parts[6] } else { "0" }
  } else {
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

function Read-Candles([string]$csvPath, [string]$csvText) {
  $lines =
    if (-not [string]::IsNullOrWhiteSpace($csvText)) { $csvText -split "(\r\n|\n|\r)" } else { Get-Content -LiteralPath $csvPath }

  $candles = New-Object System.Collections.Generic.List[object]
  foreach ($line in $lines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.TrimStart().StartsWith("#")) { continue }
    if ($line -match "^(SYMBOL|NIFTY)\b" -and $line -match ",Date," ) { continue }
    if ($line -match "^(Date|DATE)\b" -and $line -match ",Time," ) { continue }

    $c = Parse-CandleLine $line
    if ($null -ne $c) { $candles.Add($c) }
  }

  $candles = $candles | Sort-Object Dt
  return ,$candles
}

function Only-OneDate($candles, [string]$label) {
  $dates = @($candles | ForEach-Object { $_.Dt.ToString("yyyy-MM-dd") } | Select-Object -Unique)
  if ($dates.Count -ne 1) {
    throw "$label candles must be a single session date. Found: $($dates -join ', ')"
  }
  return $dates[0]
}

function Classify-OpenScenario([double]$dayOpen, [double]$prevClose) {
  $gap = $dayOpen - $prevClose
  if ($gap -ge $GapThresholdPoints) { return "gap_up" }
  if ($gap -le (-1 * $GapThresholdPoints)) { return "gap_down" }
  if ([math]::Abs($gap) -le $FlatThresholdPoints) { return "flat" }
  return $(if ($gap -gt 0) { "mild_gap_up" } else { "mild_gap_down" })
}

function F([double]$x) { return $x.ToString("F2") }

function Target-For([string]$side, [double]$entry, [double]$stop, [double]$t1) {
  if ($TargetMode -eq "T1") { return $t1 }
  $risk = [math]::Abs($entry - $stop)
  if ($risk -le 0.0) { $risk = 1.0 }
  return $(if ($side -eq "BUY") { $entry + $risk } else { $entry - $risk })
}

$prevCandles = Read-Candles $PrevCsvPath $PrevCsvText
$dayCandles = Read-Candles $DayCsvPath $DayCsvText

if ($prevCandles.Count -lt 30) { throw "Prev session needs at least ~30 candles. Got $($prevCandles.Count)." }
if ($dayCandles.Count -lt 30) { throw "Day session needs at least ~30 candles. Got $($dayCandles.Count)." }

$prevDate = Only-OneDate $prevCandles "Prev"
$dayDate = Only-OneDate $dayCandles "Day"

$symbol = $dayCandles[0].Symbol

$pdh = ($prevCandles | Measure-Object -Property High -Maximum).Maximum
$pdl = ($prevCandles | Measure-Object -Property Low -Minimum).Minimum
$prevClose = $prevCandles[-1].Close

$dayOpen = $dayCandles[0].Open
$scenario = Classify-OpenScenario $dayOpen $prevClose

Write-Host ""
Write-Host "Nexus Ultra day simulator (deterministic; educational/backtest use; not financial advice)"
Write-Host "Symbol: $symbol  Prev: $prevDate  Day: $dayDate"
Write-Host ("Prev levels: PDH {0} | PDL {1} | PrevClose {2}" -f (F $pdh), (F $pdl), (F $prevClose))
Write-Host ("Open: {0} | Scenario: {1}" -f (F $dayOpen), $scenario)
Write-Host ("Qty: {0} | TargetMode: {1} | MaxAttemptsPerIdea: {2}" -f $Quantity, $TargetMode, $MaxAttemptsPerIdea)
Write-Host ""

# Define scenario idea (single-idea sim, max attempts)
$idea = $scenario
$attempts = 0

$pos = $null
$trades = New-Object System.Collections.Generic.List[object]

foreach ($c in $dayCandles) {
  if ($null -eq $pos) {
    if ($attempts -ge $MaxAttemptsPerIdea) { continue }

    $entry = $null
    $side = $null
    $stop = $null
    $t1 = $null
    $reason = $null

    if ($scenario -in @("gap_up", "mild_gap_up")) {
      # Short after failed acceptance above PDH
      if ($c.High -ge $pdh -and $c.Close -lt $pdh) {
        $entry = $c.Close
        $side = "SELL"
        $stop = $pdh + $StopBufferPoints
        $t1 = $prevClose
        $reason = "gap_up_failed_acceptance_above_pdh"
      }
    } elseif ($scenario -in @("gap_down", "mild_gap_down")) {
      # Do not sell immediately; sell pullback rejection near anchor
      $anchor = [math]::Min($prevClose, $pdl + $FlatThresholdPoints)
      if ($c.High -ge $anchor -and $c.Close -lt $anchor) {
        $entry = $c.Close
        $side = "SELL"
        $stop = $anchor + $StopBufferPoints
        $t1 = $pdl
        $reason = "gap_down_pullback_rejection"
      }
    } elseif ($scenario -eq "flat") {
      if ($c.Close -gt $pdh) {
        $entry = $c.Close
        $side = "BUY"
        $stop = $pdh - $StopBufferPoints
        $t1 = $pdh + $TargetStepPoints
        $reason = "flat_accept_above_pdh"
      } elseif ($c.Close -lt $pdl) {
        $entry = $c.Close
        $side = "SELL"
        $stop = $pdl + $StopBufferPoints
        $t1 = $pdl - $TargetStepPoints
        $reason = "flat_accept_below_pdl"
      }
    }

    if ($null -ne $entry) {
      $target = Target-For $side $entry $stop $t1
      $pos = [pscustomobject]@{
        Side = $side
        EntryDt = $c.Dt
        Entry = $entry
        Stop = $stop
        Target = $target
        Reason = $reason
      }
    }
    continue
  }

  # Manage position candle-by-candle
  $exit = $null
  $exitReason = $null

  if ($pos.Side -eq "BUY") {
    $hitStop = $c.Low -le $pos.Stop
    $hitTarget = $c.High -ge $pos.Target
    if ($hitStop -and $hitTarget) {
      if ($AmbiguousFillPolicy -eq "target_first") { $exit = $pos.Target; $exitReason = "TARGET(ambiguous)" } else { $exit = $pos.Stop; $exitReason = "STOP(ambiguous)" }
    } elseif ($hitStop) {
      $exit = $pos.Stop; $exitReason = "STOP"
    } elseif ($hitTarget) {
      $exit = $pos.Target; $exitReason = "TARGET"
    }
  } else {
    $hitStop = $c.High -ge $pos.Stop
    $hitTarget = $c.Low -le $pos.Target
    if ($hitStop -and $hitTarget) {
      if ($AmbiguousFillPolicy -eq "target_first") { $exit = $pos.Target; $exitReason = "TARGET(ambiguous)" } else { $exit = $pos.Stop; $exitReason = "STOP(ambiguous)" }
    } elseif ($hitStop) {
      $exit = $pos.Stop; $exitReason = "STOP"
    } elseif ($hitTarget) {
      $exit = $pos.Target; $exitReason = "TARGET"
    }
  }

  if ($null -ne $exit) {
    $points =
      if ($pos.Side -eq "BUY") { $exit - $pos.Entry } else { $pos.Entry - $exit }
    $pnl = $points * $Quantity

    $trades.Add([pscustomobject]@{
      Idea = $idea
      Side = $pos.Side
      EntryTime = $pos.EntryDt.ToString("HH:mm")
      Entry = [double]$pos.Entry
      Stop = [double]$pos.Stop
      Target = [double]$pos.Target
      ExitTime = $c.Dt.ToString("HH:mm")
      Exit = [double]$exit
      ExitReason = $exitReason
      Points = [double]$points
      PnL = [double]$pnl
      EntryReason = $pos.Reason
    })

    $pos = $null
    $attempts++
  }
}

if ($null -ne $pos) {
  $last = $dayCandles[-1]
  $exit = $last.Close
  $points =
    if ($pos.Side -eq "BUY") { $exit - $pos.Entry } else { $pos.Entry - $exit }
  $pnl = $points * $Quantity
  $trades.Add([pscustomobject]@{
    Idea = $idea
    Side = $pos.Side
    EntryTime = $pos.EntryDt.ToString("HH:mm")
    Entry = [double]$pos.Entry
    Stop = [double]$pos.Stop
    Target = [double]$pos.Target
    ExitTime = $last.Dt.ToString("HH:mm")
    Exit = [double]$exit
    ExitReason = "EOD"
    Points = [double]$points
    PnL = [double]$pnl
    EntryReason = $pos.Reason
  })
  $pos = $null
}

if ($trades.Count -lt 1) {
  Write-Host "No trades triggered for this scenario using the current deterministic rules."
  exit 0
}

Write-Host "Trades"
$trades |
  Select-Object Idea,Side,EntryTime,Entry,Stop,Target,ExitTime,Exit,ExitReason,Points,PnL,EntryReason |
  Format-Table -AutoSize

$sum = ($trades | Measure-Object -Property PnL -Sum).Sum
$w = @($trades | Where-Object { $_.PnL -gt 0 }).Count
$l = @($trades | Where-Object { $_.PnL -lt 0 }).Count

Write-Host ""
Write-Host ("Summary: Trades {0} | Wins {1} | Losses {2} | Total P&L {3}" -f $trades.Count, $w, $l, (F $sum))

Write-Host ""
Write-Host "Notes"
Write-Host "- Conservative fill: if stop+target touch in same candle, AmbiguousFillPolicy decides (default stop_first)."
Write-Host "- This is a starter deterministic policy; extend with full rulebook scoring/validity/comfort for production."
