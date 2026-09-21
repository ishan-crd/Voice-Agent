from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")

    # models
    models: str = "turbo,multilingual"  # comma-separated: turbo | multilingual
    device: str = "cuda"
    watermark: bool = True
    # output gain per model so voices sit at a similar level when switching languages
    gain_turbo: float = 1.0
    gain_multilingual: float = 0.4

    # voices
    voices_dir: Path = Path("voices")
    default_voice: str = "default"

    # server
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: str = ""  # comma-separated bearer tokens; empty = no auth (gateway=False only)

    # gateway: accounts, hashed API keys, quotas, metering (SQLite)
    gateway: bool = False
    db_path: Path = Path("data/voiceagent.db")
    admin_key: str = ""  # unlocks /admin/* and bypasses quotas
    public_url: str = ""  # shown in the dashboard / docs, e.g. https://tts.example.com
    cors_origins: str = "*"  # comma-separated; "*" for the beta, tighten in production

    # streaming
    first_chunk_words: int = 10
    max_chunk_chars: int = 250
    lookahead: int = 2

    # token-level streaming (turbo + multilingual)
    streaming: bool = True  # False = stock generate() per sentence (debugging only)
    first_block_tokens: int = 12  # speech tokens (25/s) before the first vocoder pass
    ref_seconds: int = 6  # S3Gen reference prompt length used per block (0 = full clip)
    t3_fp16: bool = True
    cuda_graph: bool = True

    # talk: speech-to-text + LLM for the live conversation demo
    stt: bool = True  # load Whisper at startup (~1.5 GB VRAM)
    stt_model: str = "openai/whisper-large-v3-turbo"
    llm_base_url: str = "http://127.0.0.1:11434/v1"  # any OpenAI-compatible chat endpoint (Ollama by default)
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    # keep this free of other languages' names: mentioning "Hindi"/"Devanagari" here
    # primes small models to answer in Hindi regardless of what the user spoke
    llm_system_prompt: str = (
        "You are a friendly, concise voice assistant on a phone call. Reply in one to three short sentences. "
        "Never use markdown, lists, emojis or symbols - only plain spoken sentences."
    )

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
