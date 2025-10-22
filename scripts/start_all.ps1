Param(
  [int]$IntervalSeconds = 3600,
  [int]$LookbackDays = 3
)

$ErrorActionPreference = 'Stop'

# 计算项目根目录与日志目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogsDir = Join-Path $ProjectRoot 'logs'
New-Item -Force -ItemType Directory -Path $LogsDir | Out-Null

# 启动自适应权重守护服务
$weightsOut = Join-Path $LogsDir 'adaptive_weights.out'
$weightsErr = Join-Path $LogsDir 'adaptive_weights.err'
$weightsPid = Join-Path $LogsDir 'adaptive_weights.pid'

$weightsProc = Start-Process -FilePath 'python' -ArgumentList @('-m','decision_engine.adaptive_weights_runner','--interval',"$IntervalSeconds",'--lookback',"$LookbackDays") -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $weightsOut -RedirectStandardError $weightsErr
$weightsProc.Id | Out-File -FilePath $weightsPid -Encoding ascii -Force

# 启动Dashboard（使用 python -m streamlit 以避免PATH问题）
$dashOut = Join-Path $LogsDir 'dashboard.out'
$dashErr = Join-Path $LogsDir 'dashboard.err'
$dashPid = Join-Path $LogsDir 'dashboard.pid'

$dashProc = Start-Process -FilePath 'python' -ArgumentList @('-m','streamlit','run','web/unified_dashboard.py') -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $dashOut -RedirectStandardError $dashErr
$dashProc.Id | Out-File -FilePath $dashPid -Encoding ascii -Force

Write-Host "✅ 已启动："
Write-Host "  - 自适应权重守护 (PID=$($weightsProc.Id)) | 日志：$weightsOut"
Write-Host "  - Dashboard (PID=$($dashProc.Id)) | 日志：$dashOut"
Write-Host "📟 打开 http://localhost:8501 查看面板（默认端口）"
