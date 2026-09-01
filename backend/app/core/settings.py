from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import rtoml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "api.toml"


@dataclass(frozen=True)
class AppSettings:
    name: str
    api_v1_prefix: str


@dataclass(frozen=True)
class TenancySettings:
    default_tenant_id: str
    default_user_id: str
    allow_client_tenant_id: bool


@dataclass(frozen=True)
class ChatSettings:
    top_k: int
    enable_long_term_memory: bool
    enable_semantic_memory_write: bool


@dataclass(frozen=True)
class ArtifactSettings:
    source_root: Path
    chunked_root: Path


@dataclass(frozen=True)
class RedisSettings:
    url: str
    channel_prefix: str


@dataclass(frozen=True)
class UploadSettings:
    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    stream_chunk_bytes: int
    allowed_suffixes: dict[str, str]


@dataclass(frozen=True)
class DedupeSettings:
    enabled: bool
    algorithm: str
    scope: str


@dataclass(frozen=True)
class MarkerSettings:
    ready: str
    done: str
    failed: str


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    tenancy: TenancySettings
    chat: ChatSettings
    artifacts: ArtifactSettings
    redis: RedisSettings
    uploads: UploadSettings
    dedupe: DedupeSettings
    markers: MarkerSettings

    @property
    def app_name(self) -> str:
        return self.app.name

    @property
    def api_v1_prefix(self) -> str:
        return self.app.api_v1_prefix

    @property
    def default_tenant_id(self) -> str:
        return self.tenancy.default_tenant_id

    @property
    def default_user_id(self) -> str:
        return self.tenancy.default_user_id

    @property
    def allow_client_tenant_id(self) -> bool:
        return self.tenancy.allow_client_tenant_id

    @property
    def chat_top_k(self) -> int:
        return self.chat.top_k

    @property
    def chat_enable_long_term_memory(self) -> bool:
        return self.chat.enable_long_term_memory

    @property
    def chat_enable_semantic_memory_write(self) -> bool:
        return self.chat.enable_semantic_memory_write

    @property
    def artifact_source_root(self) -> Path:
        return self.artifacts.source_root

    @property
    def artifact_chunked_root(self) -> Path:
        return self.artifacts.chunked_root

    @property
    def redis_url(self) -> str:
        return self.redis.url

    @property
    def redis_channel_prefix(self) -> str:
        return self.redis.channel_prefix

    @property
    def max_upload_files(self) -> int:
        return self.uploads.max_files

    @property
    def max_upload_file_bytes(self) -> int:
        return self.uploads.max_file_bytes

    @property
    def max_upload_total_bytes(self) -> int:
        return self.uploads.max_total_bytes

    @property
    def upload_stream_chunk_bytes(self) -> int:
        return self.uploads.stream_chunk_bytes


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(f"Expected TOML table: {key!r}")

    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string config value: {key!r}")

    return value.strip()


def _require_env(key: str) -> str:
    value = os.getenv(key, "").strip()

    if not value:
        raise ValueError(
            f"Expected non-empty environment variable: {key!r}"
        )

    return value


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean config value: {key!r}")

    return value


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: int = 1,
) -> int:
    value = data.get(key)

    if not isinstance(value, int):
        raise ValueError(f"Expected integer config value: {key!r}")

    if value < min_value:
        raise ValueError(f"Expected {key!r} >= {min_value}, got {value}")

    return value


def _require_suffix_mapping(data: Mapping[str, Any], key: str) -> dict[str, str]:
    value = data.get(key)

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"Expected non-empty TOML table: {key!r}")

    result: dict[str, str] = {}

    for raw_suffix, raw_media_type in value.items():
        suffix = str(raw_suffix or "").strip().lower()
        media_type = str(raw_media_type or "").strip()

        if not suffix.startswith("."):
            raise ValueError(f"Upload suffix must start with '.': {suffix!r}")

        if not media_type:
            raise ValueError(f"Upload media type cannot be empty for suffix: {suffix!r}")

        result[suffix] = media_type

    return result


def _load_config_data(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"API config not found: {config_path}")

    data = rtoml.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(data, Mapping):
        raise ValueError(f"API config must be a TOML object: {config_path}")

    return data


def _validate_settings(settings: Settings) -> Settings:
    if settings.dedupe.algorithm != "sha256":
        raise ValueError("dedupe.algorithm must be 'sha256'")

    if settings.dedupe.scope != "within_job":
        raise ValueError("dedupe.scope must be 'within_job' for now")

    return settings


def load_settings(config_path: Path | None = None) -> Settings:
    active_path = config_path or Path(
        os.getenv("DOCS_GPT_API_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )

    data = _load_config_data(active_path)

    app = _require_mapping(data, "app")
    tenancy = _require_mapping(data, "tenancy")
    chat = _require_mapping(data, "chat")
    artifacts = _require_mapping(data, "artifacts")
    redis = _require_mapping(data, "redis")
    uploads = _require_mapping(data, "uploads")
    dedupe = _require_mapping(data, "dedupe")
    markers = _require_mapping(data, "markers")

    return _validate_settings(
        Settings(
            app=AppSettings(
                name=_require_str(app, "name"),
                api_v1_prefix=_require_str(app, "api_v1_prefix"),
            ),
            tenancy=TenancySettings(
                default_tenant_id=_require_str(tenancy, "default_tenant_id"),
                default_user_id=_require_str(tenancy, "default_user_id"),
                allow_client_tenant_id=_require_bool(tenancy, "allow_client_tenant_id"),
            ),
            chat=ChatSettings(
                top_k=_require_int(chat, "top_k"),
                enable_long_term_memory=_require_bool(
                    chat,
                    "enable_long_term_memory",
                ),
                enable_semantic_memory_write=_require_bool(
                    chat,
                    "enable_semantic_memory_write",
                ),
            ),
            artifacts=ArtifactSettings(
                source_root=Path(_require_str(artifacts, "source_root")),
                chunked_root=Path(_require_str(artifacts, "chunked_root")),
            ),
            redis=RedisSettings(
                url=_require_env("REDIS_URL"),
                channel_prefix=_require_str(
                    redis,
                    "channel_prefix",
                ),
            ),
            uploads=UploadSettings(
                max_files=_require_int(uploads, "max_files"),
                max_file_bytes=_require_int(uploads, "max_file_bytes"),
                max_total_bytes=_require_int(uploads, "max_total_bytes"),
                stream_chunk_bytes=_require_int(uploads, "stream_chunk_bytes"),
                allowed_suffixes=_require_suffix_mapping(uploads, "allowed_suffixes"),
            ),
            dedupe=DedupeSettings(
                enabled=_require_bool(dedupe, "enabled"),
                algorithm=_require_str(dedupe, "algorithm"),
                scope=_require_str(dedupe, "scope"),
            ),
            markers=MarkerSettings(
                ready=_require_str(markers, "ready"),
                done=_require_str(markers, "done"),
                failed=_require_str(markers, "failed"),
            ),
        )
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
