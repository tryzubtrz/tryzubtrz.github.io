"""Generate self-signed TLS certificates for local HTTPS."""
from __future__ import annotations

import datetime
import ipaddress
import socket
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from config import CERTS_DIR


def generate_self_signed(
    cert_path: Path | None = None,
    key_path: Path | None = None,
    days: int = 365,
    common_name: str = "localhost",
) -> Tuple[Path, Path]:
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = cert_path or (CERTS_DIR / "localhost.crt")
    key_path = key_path or (CERTS_DIR / "localhost.key")

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Trading Platform"),
        ]
    )

    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        san_list.append(x509.IPAddress(ipaddress.ip_address(local_ip)))
    except Exception:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path
