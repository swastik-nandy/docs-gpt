from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rtoml


@dataclass(frozen=True, slots=True)
class ModelsRuntimeConfig:
    registry_path: Path


@dataclass(frozen=True, slots=True)
class QdrantMemoryRuntimeConfig:
    url: str
    collection_name: str
    vector_size: int
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class MemoryRuntimeConfig:
    qdrant: QdrantMemoryRuntimeConfig


@dataclass(frozen=True, slots=True)
class GenerationRuntimeConfig:
    models: ModelsRuntimeConfig
    memory: MemoryRuntimeConfig

    @classmethod
    def from_toml(cls, path: str | Path | None = None) -> "GenerationRuntimeConfig":
        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).resolve().parent / "config" / "runtime.toml"
        )
        config_path = config_path.resolve()
        raw = rtoml.load(config_path)

        raw_models = raw.get("models", {})
        raw_qdrant = raw.get("memory", {}).get("qdrant", {})

        registry_path = Path(raw_models["registry_path"])
        if not registry_path.is_absolute():
            registry_path = config_path.parent / registry_path

        return cls(
            models=ModelsRuntimeConfig(
                registry_path=registry_path.resolve(),
            ),
            memory=MemoryRuntimeConfig(
                qdrant=QdrantMemoryRuntimeConfig(
                    url=str(raw_qdrant["url"]),
                    collection_name=str(raw_qdrant["collection_name"]),
                    vector_size=int(raw_qdrant["vector_size"]),
                    timeout_seconds=float(raw_qdrant.get("timeout_seconds", 10.0)),
                )
            ),
        )
