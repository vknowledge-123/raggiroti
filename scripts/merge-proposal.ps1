param(
  [Parameter(Mandatory = $true)]
  [string]$ProposalPath,

  [Parameter(Mandatory = $false)]
  [string]$RulebookPath = ".\\rulebook\\nexus_ultra_v2.rulebook.json",

  [Parameter(Mandatory = $false)]
  [ValidateSet("patch","minor","major")]
  [string]$Bump = "patch"
)

$ErrorActionPreference = "Stop"
py -m raggiroti.cli_merge_proposal --rulebook $RulebookPath --proposal $ProposalPath --bump $Bump
