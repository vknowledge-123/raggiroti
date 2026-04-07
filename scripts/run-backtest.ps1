param(
  [Parameter(Mandatory = $true)]
  [string]$CsvPath,

  [Parameter(Mandatory = $false)]
  [int]$Qty = 65
)

$ErrorActionPreference = "Stop"
py -m raggiroti.cli_backtest --csv $CsvPath --qty $Qty
