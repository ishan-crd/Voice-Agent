from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")

    # models
    models: str = "turbo,multilingual"  # comma-separated: turbo | multilingual
    device: str = "cuda"
    watermark: bool = True

    # voices
    voices_dir: Path = Path("voices")
    default_voice: str = "default"

    # server
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: str = ""  # comma-separated bearer tokens; empty = no auth

    # streaming
    first_chunk_words: int = 10
    max_chunk_chars: int = 250
    lookahead: int = 2

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
