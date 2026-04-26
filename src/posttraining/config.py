from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    tinker_api_key: SecretStr | None = Field(default=None)
    tinker_base_url: str | None = Field(default=None)
    default_model: str = Field(default="Qwen/Qwen3-0.6B")
    output_dir: Path = Field(default=Path("outputs"))

    @property
    def has_tinker_api_key(self) -> bool:
        return bool(self.tinker_api_key and self.tinker_api_key.get_secret_value())


def load_settings(env_file: Path | str = ".env") -> Settings:
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)

    return Settings(
        tinker_api_key=_optional_secret_env("TINKER_API_KEY"),
        tinker_base_url=_optional_env("TINKER_BASE_URL"),
        default_model=os.getenv("POSTTRAINING_DEFAULT_MODEL", "Qwen/Qwen3-0.6B"),
        output_dir=Path(os.getenv("POSTTRAINING_OUTPUT_DIR", "outputs")),
    )


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value


def _optional_secret_env(name: str) -> SecretStr | None:
    value = _optional_env(name)
    if value is None:
        return None
    return SecretStr(value)
