Param(
  [string]$Python = "python",
  [string]$TestPath = "tests/e2e/test_mvp_slo.py",
  [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "Running SLO tests via pytest..."
if (!(Test-Path $TestPath)) {
  Write-Error "Test file not found: $TestPath"
  exit 2
}

# Create reports dir if needed (optional)
$reportDir = "reports/slo"
if (!(Test-Path $reportDir)) { New-Item -ItemType Directory -Force -Path $reportDir | Out-Null }

& $Python -m pytest $TestPath -q $ExtraArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Error "SLO tests failed with exit code $code"
}
exit $code
