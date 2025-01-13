from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
import jwt
from pydantic_settings import BaseSettings
import tomlkit as toml
from pathlib import Path
import logging
from .settings import auth_settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_days: int

    @classmethod
    def from_toml(cls, toml_path: str):
        print(f"[auth_service.py]Loading configuration from: {toml_path}")
        with open(toml_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        print(f"[auth_service.py]Loaded configuration: {config}")
        return cls(
            secret_key=config["app"]["secret_key"],
            algorithm=config["app"]["algorithm"],
            access_token_expire_minutes=config["app"]["access_token_expire_minutes"],
            refresh_token_expire_days=config["app"]["refresh_token_expire_days"],
        )

# 加载配置文件
config_path = Path(__file__).parent.parent / "dist" / "server_config.toml"
print(f"[auth_service.py]Config path: {config_path}")
settings = Settings.from_toml(config_path)
print(f"[auth_service.py]Settings loaded: {settings}")

# 密码加密工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
print("[auth_service.py]Password context initialized.")

class AuthService:
    def __init__(self):
        self.settings = auth_settings
        print("[auth_service.py]Initializing AuthService...")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        print(f"[auth_service.py]Verifying password for user...")
        result = pwd_context.verify(plain_password, hashed_password)
        print(f"[auth_service.py]Password verification result: {result}")
        return result

    def get_password_hash(self, password: str) -> str:
        print(f"[auth_service.py]Hashing password...")
        hashed_password = pwd_context.hash(password)
        print(f"[auth_service.py]Password hashed successfully.")
        return hashed_password

    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        print(f"[auth_service.py]Creating access token with data: {data}")
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        print(f"[auth_service.py]Access token created: {encoded_jwt}")
        return encoded_jwt

    def create_refresh_token(self, data: dict, expires_delta: timedelta = None):
        print(f"[auth_service.py]Creating refresh token with data: {data}")
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        print(f"[auth_service.py]Refresh token created: {encoded_jwt}")
        return encoded_jwt

    def decode_token(self, token: str):
        print(f"[auth_service.py]Decoding token: {token}")
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            print(f"[auth_service.py]Token decoded successfully: {payload}")
            return payload
        except jwt.PyJWTError as e:
            logger.error(f"[auth_service.py]Token decoded failed: {e}")
            print(f"[auth_service.py]Token decoding failed: {e}")
            return None