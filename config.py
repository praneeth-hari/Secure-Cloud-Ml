import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-secret-key-change-in-production')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'fallback-jwt-key-change-in-production')
AES_KEY = os.getenv('AES_KEY', '5f3a9c82b1d6e4f7a0b5c8d2e9f1a3b4c82b1d6e4f7a0b5c8d2e9f1a3b4c5d6e')
