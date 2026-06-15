from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OpenMatchER"
    environment: str = "local"
    database_url: str = "sqlite:///./openmatcher.db"
    secret_key: str = Field(default="change-me-for-production")
    encryption_key: str = Field(default="MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=")
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
    ]
    upload_dir: str = "./data/uploads"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPENMATCHER_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
