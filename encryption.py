import base64
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from config import AES_KEY

# AES-256 requires exactly 32 bytes — pad with zeros or truncate as needed
_KEY = bytes.fromhex(AES_KEY.ljust(64, '0')[:64])


def encrypt_data(plaintext: str) -> str:
    """Encrypt a string using AES-256-CBC. Returns base64-encoded ciphertext with IV prepended."""
    iv = os.urandom(16)
    cipher = AES.new(_KEY, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext.encode('utf-8'), AES.block_size))
    return base64.b64encode(iv + ciphertext).decode('utf-8')


def decrypt_data(encoded: str) -> str:
    """Decrypt a base64-encoded AES-256-CBC ciphertext. Returns original plaintext string."""
    raw = base64.b64decode(encoded)
    iv = raw[:16]
    ciphertext = raw[16:]
    cipher = AES.new(_KEY, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size).decode('utf-8')
