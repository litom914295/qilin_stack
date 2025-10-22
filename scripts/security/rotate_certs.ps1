#!/usr/bin/env pwsh
<#
.SYNOPSIS
    证书生成与轮换脚本（P0-2）
.DESCRIPTION
    生成CA和服务证书，并更新Kubernetes Secret
.PARAMETER Action
    provision: 生成所有证书
    check: 检查证书到期状态
    rotate: 轮换即将到期的证书
.EXAMPLE
    .\rotate_certs.ps1 -Action provision
#>

Param(
    [ValidateSet("provision", "check", "rotate", "metrics")]
    [string]$Action = "check",
    
    [string]$Python = "python",
    [string]$ConfigPath = "security/mtls/config.yaml",
    [string]$CertDir = "D:\qilin-certs",
    [switch]$UpdateK8s = $false
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

Write-Host "=== Qilin Certificate Management ===" -ForegroundColor Cyan
Write-Host "Action: $Action"
Write-Host "Config: $ConfigPath"
Write-Host "Cert Directory: $CertDir"
Write-Host ""

# 确保证书目录存在
if (!(Test-Path $CertDir)) {
    New-Item -ItemType Directory -Force -Path $CertDir | Out-Null
    Write-Host "Created cert directory: $CertDir" -ForegroundColor Green
}

# 设置环境变量
$env:CERT_DIR = $CertDir

Push-Location $projectRoot
try {
    switch ($Action) {
        "provision" {
            Write-Host "Provisioning all certificates..." -ForegroundColor Yellow
            & $Python security/mtls/cert_manager.py --provision --config $ConfigPath
            if ($LASTEXITCODE -ne 0) {
                throw "Certificate provisioning failed"
            }
            Write-Host "`nCertificates provisioned successfully!" -ForegroundColor Green
            
            if ($UpdateK8s) {
                Write-Host "`nUpdating Kubernetes secrets..." -ForegroundColor Yellow
                & $PSScriptRoot\update_k8s_certs.ps1 -CertDir $CertDir
            }
        }
        
        "check" {
            Write-Host "Checking certificate expiry..." -ForegroundColor Yellow
            
            $certFiles = Get-ChildItem -Path $CertDir -Filter "*-cert.pem"
            
            if ($certFiles.Count -eq 0) {
                Write-Warning "No certificates found in $CertDir"
                exit 0
            }
            
            $expiringCerts = @()
            
            foreach ($certFile in $certFiles) {
                $output = & $Python security/mtls/cert_manager.py --check $certFile.FullName --config $ConfigPath
                Write-Host "`n$($certFile.Name):" -ForegroundColor Cyan
                Write-Host $output
                
                # 检查是否需要续期
                if ($output -match "Days left: (\d+)") {
                    $daysLeft = [int]$matches[1]
                    if ($daysLeft -le 30) {
                        $expiringCerts += @{
                            File = $certFile.Name
                            DaysLeft = $daysLeft
                        }
                    }
                }
            }
            
            if ($expiringCerts.Count -gt 0) {
                Write-Host "`nWARNING: The following certificates need renewal:" -ForegroundColor Red
                foreach ($cert in $expiringCerts) {
                    Write-Host "  - $($cert.File): $($cert.DaysLeft) days left" -ForegroundColor Yellow
                }
                exit 1
            } else {
                Write-Host "`nAll certificates are valid." -ForegroundColor Green
            }
        }
        
        "rotate" {
            Write-Host "Rotating expiring certificates..." -ForegroundColor Yellow
            # 重新生成证书（cert_manager.py会自动检查并续期）
            & $Python security/mtls/cert_manager.py --provision --config $ConfigPath
            
            if ($LASTEXITCODE -ne 0) {
                throw "Certificate rotation failed"
            }
            
            Write-Host "`nCertificates rotated successfully!" -ForegroundColor Green
            
            if ($UpdateK8s) {
                Write-Host "`nUpdating Kubernetes secrets..." -ForegroundColor Yellow
                & $PSScriptRoot\update_k8s_certs.ps1 -CertDir $CertDir
            }
        }
        
        "metrics" {
            Write-Host "Exporting Prometheus metrics..." -ForegroundColor Yellow
            & $Python security/mtls/cert_manager.py --metrics --config $ConfigPath
        }
    }
} finally {
    Pop-Location
}

Write-Host "`nDone." -ForegroundColor Cyan
