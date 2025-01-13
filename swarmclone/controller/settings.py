from pydantic_settings import BaseSettings
import tomlkit as toml
from pathlib import Path

class AuthSettings(BaseSettings):
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_days: int

    @classmethod
    def from_toml(cls, toml_path: str):
        with open(toml_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return cls(
            secret_key=config["app"]["secret_key"],
            algorithm=config["app"]["algorithm"],
            access_token_expire_minutes=config["app"]["access_token_expire_minutes"],
            refresh_token_expire_days=config["app"]["refresh_token_expire_days"],
        )

class DBSettings(BaseSettings):
    database_url: str

    @classmethod
    def from_toml(cls, toml_path: str):
        with open(toml_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return cls(database_url=config["app"]["database_url"])

config_path = Path(__file__).parent.parent / "dist" / "server_config.toml"

auth_settings = AuthSettings.from_toml(config_path)
db_settings = DBSettings.from_toml(config_path)