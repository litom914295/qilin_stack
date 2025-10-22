$ErrorActionPreference = 'SilentlyContinue'

# 计算项目根目录与日志目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogsDir = Join-Path $ProjectRoot 'logs'

function Stop-ByPidFile([string]$Name){
  $pidFile = Join-Path $LogsDir ("$Name.pid")
  if (Test-Path $pidFile) {
    $pid = Get-Content $pidFile | Select-Object -First 1
    if ($pid) {
      try { Stop-Process -Id ([int]$pid) -Force -ErrorAction Stop; Write-Host "⏹️ 已停止 $Name (PID=$pid)" } catch {}
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  }
}

Stop-ByPidFile -Name 'adaptive_weights'
Stop-ByPidFile -Name 'dashboard'

Write-Host "✅ 已尝试停止所有服务（如未运行将忽略）。"
