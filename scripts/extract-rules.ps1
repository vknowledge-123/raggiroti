param(
  [Parameter(Mandatory = $true)]
  [string]$File,

  [Parameter(Mandatory = $false)]
  [string]$SourceTranscriptId = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($SourceTranscriptId)) {
  py -m raggiroti.cli_extract_rules --file $File
} else {
  py -m raggiroti.cli_extract_rules --file $File --source-transcript-id $SourceTranscriptId
}
