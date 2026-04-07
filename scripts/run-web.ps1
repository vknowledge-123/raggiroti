param(
  [Parameter(Mandatory = $false)]
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
py -m uvicorn raggiroti.web.app:app --host 127.0.0.1 --port $Port --reload

