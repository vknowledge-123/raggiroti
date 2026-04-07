param(
  [Parameter(Mandatory = $true)]
  [string]$CsvPath,

  [Parameter(Mandatory = $false)]
  [int]$Qty = 65,

  [Parameter(Mandatory = $false)]
  [double]$Gap = 30,

  [Parameter(Mandatory = $false)]
  [double]$Flat = 15
  ,
  [Parameter(Mandatory = $false)]
  [string]$StartDate = "",
  [Parameter(Mandatory = $false)]
  [string]$EndDate = ""
)

$ErrorActionPreference = "Stop"
$args2 = @("--csv", $CsvPath, "--qty", $Qty, "--gap", $Gap, "--flat", $Flat)
if (-not [string]::IsNullOrWhiteSpace($StartDate)) { $args2 += @("--start", $StartDate) }
if (-not [string]::IsNullOrWhiteSpace($EndDate)) { $args2 += @("--end", $EndDate) }
py -m raggiroti.cli_backtest_multi @args2
