"""
SSL-Zertifikat-Hilfsfunktionen für ice_brain.

Erstellt beim ersten Start ein selbstsigniertes Zertifikat (RSA 2048, 10 Jahre)
mit SAN für localhost, *.local, 127.0.0.1 und 0.0.0.0.

Reihenfolge:
1. Versucht das `cryptography`-Paket (bevorzugt).
2. Fällt zurück auf den `openssl`-CLI.

Ablage: ice_brain/certs/cert.pem und ice_brain/certs/key.pem
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Zertifikatsverzeichnis neben diesem Paket
_CERTS_DIR = Path(__file__).parent.parent / "certs"
_CERT_FILE = _CERTS_DIR / "cert.pem"
_KEY_FILE = _CERTS_DIR / "key.pem"

# Gültigkeitsdauer in Tagen (10 Jahre)
_CERT_VALIDITY_DAYS = 3650

# SAN-Einträge für das Zertifikat
_HOSTNAMES = ["localhost", "*.local"]
_IPS = ["127.0.0.1", "0.0.0.0"]


def _generate_with_cryptography() -> None:
    """Erstellt ein selbstsigniertes Zertifikat mit dem `cryptography`-Paket."""
    from cryptography import x509  # noqa: PLC0415
    from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: PLC0415
    from cryptography.x509.oid import NameOID  # noqa: PLC0415
    import ipaddress  # noqa: PLC0415

    # Schlüsselpaar generieren
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "ice_brain"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ice_brain self-signed"),
    ])

    # SAN-Erweiterung mit Hostnamen und IP-Adressen
    san_entries: list[x509.GeneralName] = []
    for hostname in _HOSTNAMES:
        san_entries.append(x509.DNSName(hostname))
    for ip in _IPS:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=_CERT_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(private_key, hashes.SHA256())
    )

    # Schlüssel schreiben
    _KEY_FILE.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    # Schlüsseldatei auf 0600 setzen (nur Besitzer darf lesen)
    _KEY_FILE.chmod(0o600)

    # Zertifikat schreiben
    _CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    logger.info("SSL-Zertifikat erstellt (cryptography): %s", _CERT_FILE)


def _generate_with_openssl() -> None:
    """Erstellt ein selbstsigniertes Zertifikat mit dem openssl-CLI als Fallback."""
    # SAN-String für openssl
    san_parts = [f"DNS:{h}" for h in _HOSTNAMES] + [f"IP:{ip}" for ip in _IPS]
    san_string = ",".join(san_parts)

    # openssl req + x509 in einem Schritt
    cmd = [
        "openssl", "req", "-x509", "-nodes",
        "-newkey", "rsa:2048",
        "-keyout", str(_KEY_FILE),
        "-out", str(_CERT_FILE),
        "-days", str(_CERT_VALIDITY_DAYS),
        "-subj", "/CN=ice_brain/O=ice_brain self-signed",
        "-addext", f"subjectAltName={san_string}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(
            f"openssl fehlgeschlagen (returncode={result.returncode}): {result.stderr.strip()}"
        )
    # Schlüsseldatei auf 0600 setzen
    _KEY_FILE.chmod(0o600)
    logger.info("SSL-Zertifikat erstellt (openssl CLI): %s", _CERT_FILE)


def ensure_ssl_cert() -> tuple[str, str] | None:
    """Stellt sicher, dass ein Zertifikat vorhanden ist.

    Gibt (cert_path, key_path) zurück oder None, wenn die Erstellung fehlschlug.
    Vorhandene Zertifikate werden nicht überschrieben.
    """
    _CERTS_DIR.mkdir(parents=True, exist_ok=True)

    if _CERT_FILE.exists() and _KEY_FILE.exists():
        logger.info("SSL-Zertifikat bereits vorhanden: %s", _CERT_FILE)
        return str(_CERT_FILE), str(_KEY_FILE)

    # Versuche cryptography-Paket zuerst
    try:
        _generate_with_cryptography()
        return str(_CERT_FILE), str(_KEY_FILE)
    except ImportError:
        logger.info("cryptography-Paket nicht verfügbar – versuche openssl CLI …")
    except Exception as exc:  # noqa: BLE001
        logger.warning("SSL-Zertifikat via cryptography fehlgeschlagen: %s", exc)

    # Fallback: openssl CLI
    try:
        _generate_with_openssl()
        return str(_CERT_FILE), str(_KEY_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SSL-Zertifikat via openssl CLI fehlgeschlagen: %s", exc)

    return None
