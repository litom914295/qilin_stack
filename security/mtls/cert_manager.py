"""
mTLS 证书管理器（P0-2）
生成、轮换、监控服务间通信证书
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import yaml

logger = logging.getLogger(__name__)


class MTLSCertificateManager:
    """mTLS证书管理器"""
    
    def __init__(self, config_path: str = "security/mtls/config.yaml"):
        self.config = self._load_config(config_path)
        self.cert_dir = Path(self.config.get("cert_dir", "/etc/qilin/certs"))
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {
            "cert_dir": "/etc/qilin/certs",
            "ca_validity_days": 3650,  # 10年
            "cert_validity_days": 365,  # 1年
            "key_size": 2048,
            "renew_before_days": 30,  # 30天前续期
            "services": [
                "api-gateway",
                "feature-engine",
                "agent-orchestrator",
                "risk-controller"
            ]
        }
    
    def generate_ca(
        self,
        common_name: str = "Qilin Stack Root CA",
        output_key: Optional[str] = None,
        output_cert: Optional[str] = None
    ) -> tuple:
        """
        生成根CA证书
        
        Args:
            common_name: CN
            output_key: 私钥输出路径
            output_cert: 证书输出路径
            
        Returns:
            (private_key, certificate)
        """
        logger.info(f"Generating CA certificate: {common_name}")
        
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.get("key_size", 2048),
            backend=default_backend()
        
        # 构建证书
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Beijing"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Qilin Stack"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(
                datetime.utcnow() + timedelta(days=self.config.get("ca_validity_days", 3650))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256(), default_backend())
        
        # 保存到文件
        if output_key:
            key_path = Path(output_key)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption()
            logger.info(f"CA private key saved to {output_key}")
        
        if output_cert:
            cert_path = Path(output_cert)
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            logger.info(f"CA certificate saved to {output_cert}")
        
        return private_key, cert
    
    def generate_service_cert(
        self,
        service_name: str,
        ca_key: rsa.RSAPrivateKey,
        ca_cert: x509.Certificate,
        dns_names: Optional[List[str]] = None,
        output_key: Optional[str] = None,
        output_cert: Optional[str] = None
    ) -> tuple:
        """
        生成服务证书（由CA签发）
        
        Args:
            service_name: 服务名
            ca_key: CA私钥
            ca_cert: CA证书
            dns_names: SAN DNS列表
            output_key: 私钥输出路径
            output_cert: 证书输出路径
            
        Returns:
            (private_key, certificate)
        """
        logger.info(f"Generating service certificate: {service_name}")
        
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.get("key_size", 2048),
            backend=default_backend()
        
        # 默认SAN
        if dns_names is None:
            dns_names = [
                service_name,
                f"{service_name}.default.svc.cluster.local",
                f"{service_name}.default.svc",
                "localhost"
            ]
        
        # 构建证书
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Qilin Stack"),
            x509.NameAttribute(NameOID.COMMON_NAME, service_name),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(
                datetime.utcnow() + timedelta(days=self.config.get("cert_validity_days", 365))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(name) for name in dns_names]),
                critical=False,
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                ]),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256(), default_backend())
        
        # 保存到文件
        if output_key:
            key_path = Path(output_key)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption()
            logger.info(f"Service private key saved to {output_key}")
        
        if output_cert:
            cert_path = Path(output_cert)
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            logger.info(f"Service certificate saved to {output_cert}")
        
        return private_key, cert
    
    def check_cert_expiry(self, cert_path: str) -> Dict:
        """
        检查证书到期时间
        
        Args:
            cert_path: 证书路径
            
        Returns:
            证书信息字典
        """
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        
        now = datetime.utcnow()
        not_after = cert.not_valid_after
        days_left = (not_after - now).days
        
        renew_threshold = self.config.get("renew_before_days", 30)
        needs_renewal = days_left <= renew_threshold
        
        return {
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "serial_number": cert.serial_number,
            "not_valid_before": cert.not_valid_before,
            "not_valid_after": not_after,
            "days_left": days_left,
            "needs_renewal": needs_renewal,
            "expired": days_left < 0
        }
    
    def provision_all_certs(self):
        """为所有服务生成证书"""
        logger.info("Provisioning all service certificates")
        
        # 1. 生成CA
        ca_key_path = self.cert_dir / "ca-key.pem"
        ca_cert_path = self.cert_dir / "ca-cert.pem"
        
        if not ca_key_path.exists() or not ca_cert_path.exists():
            ca_key, ca_cert = self.generate_ca(
                output_key=str(ca_key_path),
                output_cert=str(ca_cert_path)
        else:
            logger.info("CA already exists, loading from disk")
            with open(ca_key_path, "rb") as f:
                ca_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
            with open(ca_cert_path, "rb") as f:
                ca_cert = x509.load_pem_x509_certificate(
                    f.read(), default_backend()
        
        # 2. 为每个服务生成证书
        services = self.config.get("services", [])
        for service_name in services:
            service_key_path = self.cert_dir / f"{service_name}-key.pem"
            service_cert_path = self.cert_dir / f"{service_name}-cert.pem"
            
            if service_key_path.exists() and service_cert_path.exists():
                # 检查是否需要续期
                cert_info = self.check_cert_expiry(str(service_cert_path))
                if not cert_info["needs_renewal"]:
                    logger.info(f"{service_name}: certificate valid for {cert_info['days_left']} days, skipping")
                    continue
                logger.info(f"{service_name}: certificate expires in {cert_info['days_left']} days, renewing")
            
            self.generate_service_cert(
                service_name=service_name,
                ca_key=ca_key,
                ca_cert=ca_cert,
                output_key=str(service_key_path),
                output_cert=str(service_cert_path)
        
        logger.info("All certificates provisioned successfully")
    
    def export_prometheus_metrics(self) -> str:
        """
        导出证书到期Prometheus指标
        
        Returns:
            Prometheus文本格式指标
        """
        metrics = []
        metrics.append("# HELP qilin_cert_expiry_days Days until certificate expiry")
        metrics.append("# TYPE qilin_cert_expiry_days gauge")
        
        for cert_file in self.cert_dir.glob("*-cert.pem"):
            try:
                cert_info = self.check_cert_expiry(str(cert_file))
                service_name = cert_file.stem.replace("-cert", "")
                metrics.append(
                    f'qilin_cert_expiry_days{{service="{service_name}"}} {cert_info["days_left"]}'
            except Exception as e:
                logger.error(f"Failed to check {cert_file}: {e}")
        
        return "\n".join(metrics)


def main():
    """CLI入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qilin mTLS Certificate Manager")
    parser.add_argument("--provision", action="store_true", help="Provision all certificates")
    parser.add_argument("--check", metavar="CERT", help="Check certificate expiry")
    parser.add_argument("--metrics", action="store_true", help="Export Prometheus metrics")
    parser.add_argument("--config", default="security/mtls/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    
    manager = MTLSCertificateManager(config_path=args.config)
    
    if args.provision:
        manager.provision_all_certs()
    elif args.check:
        info = manager.check_cert_expiry(args.check)
        logger.info(f"Subject: {info['subject']}")
        logger.info(f"Issuer: {info['issuer']}")
        logger.info(f"Valid from: {info['not_valid_before']}")
        logger.info(f"Valid until: {info['not_valid_after']}")
        logger.info(f"Days left: {info['days_left']}")
        logger.info(f"Needs renewal: {info['needs_renewal']}")
    elif args.metrics:
        # metrics文本需要标准输出以便被采集
        print(manager.export_prometheus_metrics())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
