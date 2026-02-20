from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

load_dotenv()


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        print(f"FATAL: Environment variable '{key}' is required but not set.")
        sys.exit(1)
    return value or ""


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    result = []
    for item in raw.split(","):
        item = item.strip()
        if item.isdigit():
            result.append(int(item))
    return result


@dataclass(frozen=True)
class BotConfig:
    token: str


@dataclass(frozen=True)
class ApiConfig:
    openrouter_keys: list[str]
    gemini_keys: list[str]
    default_provider: str
    default_model: str
    key_cooldown_minutes: int
    request_timeout: int


@dataclass(frozen=True)
class DatabaseConfig:
    path: str


@dataclass(frozen=True)
class LogConfig:
    level: str
    file: str


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    provider: str


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash (free)", "openrouter"),
    ModelInfo("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (free)", "openrouter"),
    ModelInfo("mistralai/mistral-small-24b-instruct-2501:free", "Mistral Small 24B (free)", "openrouter"),
    ModelInfo("qwen/qwen2.5-vl-72b-instruct:free", "Qwen 2.5 VL 72B (free)", "openrouter"),
    ModelInfo("gemini-2.0-flash", "Gemini 2.0 Flash", "gemini"),
    ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash", "gemini"),
    ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro", "gemini"),
]


@dataclass(frozen=True)
class Config:
    bot: BotConfig
    api: ApiConfig
    db: DatabaseConfig
    log: LogConfig
    admin_ids: list[int]
    max_context_messages: int
    models: list[ModelInfo] = field(default_factory=lambda: AVAILABLE_MODELS)

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def get_provider_for_model(self, model_id: str) -> str | None:
        info = self.get_model_info(model_id)
        return info.provider if info else None

    def get_models_by_provider(self, provider: str) -> list[ModelInfo]:
        return [m for m in self.models if m.provider == provider]


def load_config() -> Config:
    bot_token = _get_env("BOT_TOKEN", required=True)
    openrouter_keys = _parse_list(_get_env("OPENROUTER_KEYS", ""))
    gemini_keys = _parse_list(_get_env("GEMINI_KEYS", ""))

    if not openrouter_keys and not gemini_keys:
        print("FATAL: At least one API key (OPENROUTER_KEYS or GEMINI_KEYS) must be provided.")
        sys.exit(1)

    admin_ids = _parse_int_list(_get_env("ADMIN_IDS", ""))
    default_provider = _get_env("DEFAULT_PROVIDER", "openrouter")
    default_model = _get_env("DEFAULT_MODEL", "google/gemini-2.0-flash-exp:free")
    key_cooldown = int(_get_env("KEY_COOLDOWN_MINUTES", "60"))
    max_context = int(_get_env("MAX_CONTEXT_MESSAGES", "15"))
    request_timeout = int(_get_env("REQUEST_TIMEOUT", "120"))
    db_path = _get_env("DATABASE_PATH", "data/bot.db")
    log_level = _get_env("LOG_LEVEL", "INFO")
    log_file = _get_env("LOG_FILE", "data/bot.log")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    return Config(
        bot=BotConfig(token=bot_token),
        api=ApiConfig(
            openrouter_keys=openrouter_keys,
            gemini_keys=gemini_keys,
            default_provider=default_provider,
            default_model=default_model,
            key_cooldown_minutes=key_cooldown,
            request_timeout=request_timeout,
        ),
        db=DatabaseConfig(path=db_path),
        log=LogConfig(level=log_level, file=log_file),
        admin_ids=admin_ids,
        max_context_messages=max_context,
    )