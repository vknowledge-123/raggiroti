param(
  [Parameter(Mandatory = $false)]
  [string]$Path
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

$raw = Get-Content -LiteralPath $Path -Raw
$rb = $raw | ConvertFrom-Json

function Assert-NotNullOrEmpty([string]$name, $value) {
  if ($null -eq $value) { throw "Missing required field: $name" }
  if ($value -is [string] -and [string]::IsNullOrWhiteSpace($value)) { throw "Empty required field: $name" }
}

Assert-NotNullOrEmpty "name" $rb.name
Assert-NotNullOrEmpty "version" $rb.version
Assert-NotNullOrEmpty "updated_at" $rb.updated_at

Assert-NotNullOrEmpty "rules" $rb.rules
if (-not ($rb.rules -is [System.Array])) { throw "'rules' must be an array" }

$ids = @{}
foreach ($rule in $rb.rules) {
  Assert-NotNullOrEmpty "rule.id" $rule.id
  Assert-NotNullOrEmpty "rule.category" $rule.category
  Assert-NotNullOrEmpty "rule.name" $rule.name
  Assert-NotNullOrEmpty "rule.condition" $rule.condition
  Assert-NotNullOrEmpty "rule.interpretation" $rule.interpretation
  Assert-NotNullOrEmpty "rule.action" $rule.action

  if ($ids.ContainsKey($rule.id)) {
    throw "Duplicate rule id: $($rule.id)"
  }
  $ids[$rule.id] = $true

  if ($null -ne $rule.tags -and -not ($rule.tags -is [System.Array])) {
    throw "rule.tags must be an array when present (rule id: $($rule.id))"
  }
}

Write-Host "OK: Parsed and validated rulebook '$($rb.name)' v$($rb.version) with $($rb.rules.Count) rules."
