param(
  [Parameter(Mandatory = $true)]
  [string]$File,

  [Parameter(Mandatory = $false)]
  [string]$Language = "hi",

  [Parameter(Mandatory = $false)]
  [string]$Tags = "",

  [Parameter(Mandatory = $false)]
  [switch]$Embed
)

$ErrorActionPreference = "Stop"
py -m raggiroti.cli_ingest_transcript --file $File --language $Language --tags $Tags @($Embed ? "--embed" : $null)
