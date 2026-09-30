import os

from dotenv import load_dotenv

load_dotenv()

FLASK_ENV = os.getenv('FLASK_ENV', 'development')
_REQUIRED_IN_PRODUCTION = ('SECRET_KEY', 'JWT_SECRET_KEY', 'AES_KEY', 'API_USERNAME', 'API_PASSWORD')

if FLASK_ENV == 'production':
    missing = [name for name in _REQUIRED_IN_PRODUCTION if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"FLASK_ENV=production requires these environment variables to be set: {', '.join(missing)}"
        )

SECRET_KEY = os.getenv('SECRET_KEY', 'dev-only-secret-key-do-not-use-in-production')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-only-jwt-key-do-not-use-in-production')
AES_KEY = os.getenv('AES_KEY', '5f3a9c82b1d6e4f7a0b5c8d2e9f1a3b4c82b1d6e4f7a0b5c8d2e9f1a3b4c5d6e')
API_USERNAME = os.getenv('API_USERNAME', 'admin')
API_PASSWORD = os.getenv('API_PASSWORD', 'dev-only-password-do-not-use-in-production')
DATASET_PATH = os.getenv('DATASET_PATH', 'Dataset/kdd_data.csv')
