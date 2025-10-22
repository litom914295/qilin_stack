#!/usr/bin/env pwsh
<#
.SYNOPSIS
    更新Kubernetes中的证书Secret
.DESCRIPTION
    读取本地证书文件并更新对应的Kubernetes Secret
#>

Param(
    [string]$CertDir = "D:\qilin-certs",
    [string]$Namespace = "default",
    [string]$Kubectl = "kubectl"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "=== Updating Kubernetes Secrets ===" -ForegroundColor Cyan
Write-Host "Cert Directory: $CertDir"
Write-Host "Namespace: $Namespace"
Write-Host ""

# 检查kubectl
try {
    & $Kubectl version --client | Out-Null
} catch {
    Write-Error "kubectl not found. Please install kubectl first."
    exit 1
}

# 检查证书目录
if (!(Test-Path $CertDir)) {
    Write-Error "Certificate directory not found: $CertDir"
    exit 1
}

# 更新CA证书
$caCertPath = Join-Path $CertDir "ca-cert.pem"
if (Test-Path $caCertPath) {
    Write-Host "Updating CA certificate secret..." -ForegroundColor Yellow
    
    $caCertBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($caCertPath))
    
    # 检查Secret是否存在
    $secretExists = & $Kubectl get secret qilin-ca-cert -n $Namespace 2>&1
    if ($LASTEXITCODE -eq 0) {
        # 更新现有Secret
        & $Kubectl patch secret qilin-ca-cert -n $Namespace --type='json' -p="[{'op': 'replace', 'path': '/data/ca-cert.pem', 'value': '$caCertBase64'}]"
    } else {
        # 创建新Secret
        & $Kubectl create secret generic qilin-ca-cert `
            --from-file=ca-cert.pem=$caCertPath `
            -n $Namespace
    }
    Write-Host "  ✓ CA certificate updated" -ForegroundColor Green
}

# 更新服务证书
$services = @("api-gateway", "feature-engine", "agent-orchestrator", "risk-controller")

foreach ($service in $services) {
    $certPath = Join-Path $CertDir "$service-cert.pem"
    $keyPath = Join-Path $CertDir "$service-key.pem"
    
    if ((Test-Path $certPath) -and (Test-Path $keyPath)) {
        Write-Host "Updating $service certificate..." -ForegroundColor Yellow
        
        $secretName = "$service-tls"
        
        # 检查Secret是否存在
        $secretExists = & $Kubectl get secret $secretName -n $Namespace 2>&1
        if ($LASTEXITCODE -eq 0) {
            # 删除旧Secret
            & $Kubectl delete secret $secretName -n $Namespace | Out-Null
        }
        
        # 创建新Secret
        & $Kubectl create secret tls $secretName `
            --cert=$certPath `
            --key=$keyPath `
            -n $Namespace
        
        # 添加标签
        & $Kubectl label secret $secretName `
            app=qilin-stack `
            service=$service `
            -n $Namespace `
            --overwrite | Out-Null
        
        Write-Host "  ✓ $service certificate updated" -ForegroundColor Green
    } else {
        Write-Warning "Certificate files not found for $service"
    }
}

Write-Host "`nAll Kubernetes secrets updated successfully!" -ForegroundColor Cyan
Write-Host "Note: You may need to restart pods to pick up new certificates." -ForegroundColor Yellow
