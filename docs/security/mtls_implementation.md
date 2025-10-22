# mTLS 服务间通信实施（P0-2）

## 概述
实现服务间双向TLS认证，确保通信安全和身份验证。

## 组件
1. **证书管理器** (`security/mtls/cert_manager.py`)
   - 生成CA根证书
   - 颁发服务证书
   - 检查证书到期
   - 导出Prometheus指标

2. **Envoy代理** (`security/mtls/envoy-mtls.yaml`)
   - 终止mTLS连接
   - 验证客户端证书
   - 转发到后端服务

3. **Kubernetes集成**
   - Secret存储证书
   - CronJob自动轮换
   - RBAC权限控制

## 快速开始

### 1. 生成证书
```powershell
# Windows
.\scripts\security\rotate_certs.ps1 -Action provision -CertDir D:\qilin-certs

# Linux
python security/mtls/cert_manager.py --provision --config security/mtls/config.yaml
```

### 2. 更新Kubernetes Secret
```powershell
# 将证书上传到K8s
.\scripts\security\update_k8s_certs.ps1 -CertDir D:\qilin-certs -Namespace default
```

### 3. 部署Envoy Sidecar
在服务Deployment中添加Envoy sidecar容器，挂载证书Secret。

### 4. 验证mTLS连接
```bash
# 使用客户端证书访问
curl --cacert ca-cert.pem \
     --cert api-gateway-cert.pem \
     --key api-gateway-key.pem \
     https://feature-engine:8443/health
```

## 证书续期策略
- **有效期**: 服务证书1年，CA证书10年
- **续期阈值**: 到期前30天自动续期
- **轮换方式**: 
  - 自动：CronJob每天凌晨2点检查
  - 手动：`rotate_certs.ps1 -Action rotate`

## 监控与告警
- **指标**: `qilin_cert_expiry_days{service="..."}`
- **告警规则**: `monitoring/prometheus/rules/cert_expiry_alerts.yaml`
  - 30天内到期 → warning
  - 7天内到期 → page
  - 已过期 → page

## 故障排查
### 证书验证失败
1. 检查证书SAN是否包含服务DNS名
2. 验证CA证书是否一致
3. 确认证书未过期

### 自动轮换失败
1. 检查CronJob日志：`kubectl logs -l job=cert-rotation`
2. 验证RBAC权限
3. 确认PVC可写

## 安全最佳实践
1. **私钥保护**：使用Kubernetes Secret，设置严格的RBAC
2. **证书轮换**：定期轮换，不要等到即将过期
3. **监控告警**：配置Prometheus告警，提前发现问题
4. **访问审计**：记录所有证书访问和轮换操作

## 参考资料
- [Envoy TLS配置](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/transport_sockets/tls/v3/tls.proto)
- [K8s Secret管理](https://kubernetes.io/docs/concepts/configuration/secret/)
- [证书轮换最佳实践](https://wiki.qilin.internal/security/cert-rotation)
