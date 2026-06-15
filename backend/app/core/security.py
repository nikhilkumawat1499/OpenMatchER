from cryptography.fernet import Fernet

from app.core.config import get_settings


def encrypt_secret(value: str) -> str:
    return Fernet(get_settings().encryption_key.encode()).encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    return Fernet(get_settings().encryption_key.encode()).decrypt(value.encode()).decode()

