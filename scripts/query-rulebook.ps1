param(
  [Parameter(Mandatory = $false)]
  [string]$Path,

  [Parameter(Mandatory = $false)]
  [string]$Tag,

  [Parameter(Mandatory = $false)]
  [string]$Category,

  [Parameter(Mandatory = $false)]
  [string]$Text
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Path)) {
  $Path = Join-Path $scriptDir "..\\rulebook\\nexus_ultra_v2.rulebook.json"
}

if (-not (Test-Path -LiteralPath $Path)) {
  throw "Rulebook not found: $Path"
}

$rb = (Get-Content -LiteralPath $Path -Raw) | ConvertFrom-Json
$rules = $rb.rules

if ($Tag) {
  $rules = $rules | Where-Object { $_.tags -contains $Tag }
}
if ($Category) {
  $rules = $rules | Where-Object { $_.category -eq $Category }
}
if ($Text) {
  $rules = $rules | Where-Object {
    ($_.name -match [regex]::Escape($Text)) -or
    ($_.condition -match [regex]::Escape($Text)) -or
    ($_.interpretation -match [regex]::Escape($Text)) -or
    ($_.action -match [regex]::Escape($Text))
  }
}

$rules |
  Select-Object id, category, name, tags |
  Sort-Object id |
  Format-Table -AutoSize
