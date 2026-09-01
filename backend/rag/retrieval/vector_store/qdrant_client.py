from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from qdrant_client import AsyncQdrantClient, QdrantClient


Payload = dict[str, Any]

_DOTENV_LOADED_PATHS: set[Path] = set()


@dataclass(frozen=True)
class QdrantConnectionConfig:
    url: str | None = None
    host: str | None = None
    port: int | None = None
    grpc_port: int | None = None
    api_key: str | None = None
    prefer_grpc: bool = False
    timeout: float | None = None
    https: bool | None = None
    collection_name: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        env_file: str | Path | None = None,
        load_dotenv: bool = True,
        override_env: bool = False,
    ) -> QdrantConnectionConfig:
        if load_dotenv:
            load_env_file(
                env_file=env_file,
                override=override_env,
            )

        config = cls(
            url=_optional_env_or_none("QDRANT_URL"),
            host=_optional_env_or_none("QDRANT_HOST"),
            port=_optional_env_int_or_none("QDRANT_PORT"),
            grpc_port=_optional_env_int_or_none("QDRANT_GRPC_PORT"),
            api_key=_optional_env_or_none("QDRANT_API_KEY"),
            prefer_grpc=_optional_env_bool("QDRANT_PREFER_GRPC", False),
            timeout=_optional_env_float_or_none("QDRANT_TIMEOUT"),
            https=_optional_env_bool_or_none("QDRANT_HTTPS"),
            collection_name=_optional_env("QDRANT_COLLECTION", ""),
        )
        config.validate()

        return config

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QdrantConnectionConfig:
        config = cls(
            url=_optional_mapping_str_or_none(data, "url"),
            host=_optional_mapping_str_or_none(data, "host"),
            port=_optional_mapping_int_or_none(data, "port"),
            grpc_port=_optional_mapping_int_or_none(data, "grpc_port"),
            api_key=_optional_mapping_str_or_none(data, "api_key"),
            prefer_grpc=_optional_mapping_bool(data, "prefer_grpc", default=False),
            timeout=_optional_mapping_float_or_none(data, "timeout"),
            https=_optional_mapping_bool_or_none(data, "https"),
            collection_name=_optional_mapping_str(
                data,
                "collection_name",
                default="",
            ),
        )
        config.validate()

        return config

    def validate(self) -> None:
        if self.url is None and self.host is None:
            raise ValueError("Qdrant config requires url or host.")

        if self.port is not None and self.port <= 0:
            raise ValueError("Qdrant port must be greater than zero.")

        if self.grpc_port is not None and self.grpc_port <= 0:
            raise ValueError("Qdrant grpc_port must be greater than zero.")

        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Qdrant timeout must be greater than zero.")

        if not isinstance(self.prefer_grpc, bool):
            raise ValueError("Qdrant prefer_grpc must be boolean.")

        if self.https is not None and not isinstance(self.https, bool):
            raise ValueError("Qdrant https must be boolean when provided.")

    def client_kwargs(self) -> Payload:
        kwargs: Payload = {
            "prefer_grpc": self.prefer_grpc,
        }

        if self.url:
            kwargs["url"] = self.url
        else:
            kwargs["host"] = self.host

        if self.port is not None:
            kwargs["port"] = self.port

        if self.grpc_port is not None:
            kwargs["grpc_port"] = self.grpc_port

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        if self.https is not None:
            kwargs["https"] = self.https

        return kwargs


class QdrantClientFactory:
    """
    Central Qdrant client factory for retrieval.

    Dense and sparse retrievers should receive clients from here instead of
    constructing their own connection logic.
    """

    def __init__(
        self,
        config: QdrantConnectionConfig | None = None,
    ) -> None:
        self.config = config or QdrantConnectionConfig.from_env()
        self.config.validate()

    def sync_client(self) -> QdrantClient:
        return QdrantClient(**self.config.client_kwargs())

    def async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(**self.config.client_kwargs())


def load_env_file(
    *,
    env_file: str | Path | None = None,
    override: bool = False,
) -> Path | None:
    path = _resolve_env_file(env_file)
    if path is None:
        return None

    if path in _DOTENV_LOADED_PATHS and not override:
        return path

    values = _parse_env_file(path)

    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value

    _DOTENV_LOADED_PATHS.add(path)

    return path


def _resolve_env_file(env_file: str | Path | None) -> Path | None:
    if env_file is not None:
        path = Path(env_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f".env file not found: {path}")
        return path

    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate.resolve()

    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line.removeprefix("export ").strip()

        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number} in {path}: missing '='.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value.strip())

        if not key:
            raise ValueError(f"Invalid .env line {line_number} in {path}: empty key.")

        values[key] = value

    return values


def _clean_env_value(value: str) -> str:
    if not value:
        return ""

    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    comment_index = value.find(" #")
    if comment_index >= 0:
        return value[:comment_index].strip()

    return value


def _optional_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _optional_env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _optional_env_int_or_none(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _optional_env_float_or_none(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return float(value)


def _optional_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean-like value.")


def _optional_env_bool_or_none(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None

    return _optional_env_bool(name, False)


def _optional_mapping_str(
    data: Mapping[str, Any],
    key: str,
    *,
    default: str,
) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _optional_mapping_str_or_none(
    data: Mapping[str, Any],
    key: str,
) -> str | None:
    value = data.get(key)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_mapping_int_or_none(
    data: Mapping[str, Any],
    key: str,
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return int(value)


def _optional_mapping_float_or_none(
    data: Mapping[str, Any],
    key: str,
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def _optional_mapping_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = data.get(key)
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"{key} must be a boolean-like value.")


def _optional_mapping_bool_or_none(
    data: Mapping[str, Any],
    key: str,
) -> bool | None:
    value = data.get(key)
    if value is None:
        return None

    return _optional_mapping_bool(data, key, default=False)
