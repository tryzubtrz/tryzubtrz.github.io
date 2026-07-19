"""Fernet encryption helpers for .env secrets at rest."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from config import BASE_DIR


KEY_FILE = BASE_DIR / ".env.key"
ENC_ENV_FILE = BASE_DIR / ".env.enc"


def generate_key() -> bytes:
    return Fernet.generate_key()


def derive_key_from_password(password: str) -> bytes:
    """Derive a Fernet-compatible key from a passphrase (SHA-256 → urlsafe b64)."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def save_key(key: bytes, path: Path = KEY_FILE) -> None:
    path.write_bytes(key)
    os.chmod(path, 0o600)


def load_key(path: Path = KEY_FILE, env_var: str = "ENV_ENCRYPTION_KEY") -> bytes:
    env_key = os.getenv(env_var, "").strip()
    if env_key:
        # Accept raw Fernet key or passphrase
        try:
            if len(env_key) == 44:
                Fernet(env_key.encode())
                return env_key.encode()
        except Exception:
            pass
        return derive_key_from_password(env_key)
    if path.exists():
        return path.read_bytes().strip()
    key = generate_key()
    save_key(key, path)
    return key


def encrypt_env_file(env_path: Path = BASE_DIR / ".env", out_path: Path = ENC_ENV_FILE) -> Path:
    key = load_key()
    f = Fernet(key)
    plaintext = env_path.read_bytes()
    out_path.write_bytes(f.encrypt(plaintext))
    os.chmod(out_path, 0o600)
    return out_path


def decrypt_env_file(
    enc_path: Path = ENC_ENV_FILE,
    out_path: Optional[Path] = None,
) -> bytes:
    key = load_key()
    f = Fernet(key)
    try:
        plaintext = f.decrypt(enc_path.read_bytes())
    except InvalidToken as exc:
        raise ValueError("Failed to decrypt .env.enc — wrong key?") from exc
    if out_path is not None:
        out_path.write_bytes(plaintext)
        os.chmod(out_path, 0o600)
    return plaintext


def encrypt_value(value: str) -> str:
    f = Fernet(load_key())
    return f.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_value(token: str) -> str:
    f = Fernet(load_key())
    return f.decrypt(token.encode("utf-8")).decode("utf-8")


def ensure_encrypted_env() -> None:
    """If plaintext .env exists, create/update encrypted mirror."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        encrypt_env_file(env_path)
