param(
  [Parameter(Mandatory = $false)]
  [string]$Status = "draft",

  [Parameter(Mandatory = $false)]
  [string]$OutDir = ".\\rulebook\\proposals"
)

$ErrorActionPreference = "Stop"
py -m raggiroti.cli_export_proposals --status $Status --outdir $OutDir
