# rag/ingestion/data_transformation/chunking/tokenization/registry.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import TypedDict

from rag.ingestion.embeddings.config import get_embedding_pipeline_config


#--------------------PATHS--------------------

BACKEND_ROOT = (
    Path(__file__)
    .resolve()
    .parents[5]
)

DEFAULT_MODELS_ROOT = (
    BACKEND_ROOT
    / ".models"
)

REGISTRY_FILENAME = "registry.json"

SUPPORTED_ROLES = frozenset(
    {
        "dense",
        "sparse",
    }
)

REGISTRY_SCHEMA_VERSION = (
    "local_embedding_model_registry_v1"
)


#--------------------REGISTRY PAYLOADS--------------------

class RegistryInstallationPayload(TypedDict):
    requested_revision: str
    path: str
    tokenizer_path: str
    offline_runtime: bool


class RegistryModelPayload(TypedDict):
    installations: dict[
        str,
        RegistryInstallationPayload,
    ]


class RegistryRolePayload(TypedDict):
    model_id: str
    resolved_revision: str
    path: str


class ModelRegistryPayload(TypedDict, total=False):
    schema_version: str
    models: dict[
        str,
        RegistryModelPayload,
    ]
    roles: dict[
        str,
        RegistryRolePayload,
    ]
    updated_at_utc: str


#--------------------RUNTIME MODEL--------------------

@dataclass(frozen=True)
class LocalModelAsset:
    role: str
    model_id: str
    revision: str
    model_path: Path
    tokenizer_path: Path
    offline_runtime: bool


#--------------------RAW VALUE HELPERS--------------------

def _require_mapping(
    value: Any,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise RuntimeError(
            f"{field} must be an object"
        )

    return value


def _require_string(
    value: Any,
    *,
    field: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise RuntimeError(
            f"{field} must be a string"
        )

    cleaned = value.strip()

    if not cleaned:
        raise RuntimeError(
            f"{field} cannot be empty"
        )

    return cleaned


def _require_boolean(
    value: Any,
    *,
    field: str,
) -> bool:
    if not isinstance(
        value,
        bool,
    ):
        raise RuntimeError(
            f"{field} must be a boolean"
        )

    return value


#--------------------PAYLOAD PARSING--------------------

def _parse_installation_payload(
    raw: Any,
    *,
    model_id: str,
    revision: str,
) -> RegistryInstallationPayload:
    data = _require_mapping(
        raw,
        field=(
            "registry.models."
            f"{model_id}.installations.{revision}"
        ),
    )

    return RegistryInstallationPayload(
        requested_revision=_require_string(
            data.get(
                "requested_revision"
            ),
            field=(
                "registry.models."
                f"{model_id}.installations."
                f"{revision}.requested_revision"
            ),
        ),
        path=_require_string(
            data.get(
                "path"
            ),
            field=(
                "registry.models."
                f"{model_id}.installations."
                f"{revision}.path"
            ),
        ),
        tokenizer_path=_require_string(
            data.get(
                "tokenizer_path"
            ),
            field=(
                "registry.models."
                f"{model_id}.installations."
                f"{revision}.tokenizer_path"
            ),
        ),
        offline_runtime=_require_boolean(
            data.get(
                "offline_runtime"
            ),
            field=(
                "registry.models."
                f"{model_id}.installations."
                f"{revision}.offline_runtime"
            ),
        ),
    )


def _parse_model_payload(
    raw: Any,
    *,
    model_id: str,
) -> RegistryModelPayload:
    data = _require_mapping(
        raw,
        field=(
            f"registry.models.{model_id}"
        ),
    )

    raw_installations = _require_mapping(
        data.get(
            "installations"
        ),
        field=(
            "registry.models."
            f"{model_id}.installations"
        ),
    )

    installations: dict[
        str,
        RegistryInstallationPayload,
    ] = {}

    for raw_revision, raw_installation in (
        raw_installations.items()
    ):
        revision = _require_string(
            raw_revision,
            field=(
                "registry.models."
                f"{model_id}.installations revision"
            ),
        )

        installations[
            revision
        ] = _parse_installation_payload(
            raw_installation,
            model_id=model_id,
            revision=revision,
        )

    if not installations:
        raise RuntimeError(
            "No local installations registered for "
            f"embedding model {model_id!r}"
        )

    return RegistryModelPayload(
        installations=installations
    )


def _parse_role_payload(
    raw: Any,
    *,
    role: str,
) -> RegistryRolePayload:
    data = _require_mapping(
        raw,
        field=(
            f"registry.roles.{role}"
        ),
    )

    return RegistryRolePayload(
        model_id=_require_string(
            data.get(
                "model_id"
            ),
            field=(
                f"registry.roles.{role}.model_id"
            ),
        ),
        resolved_revision=_require_string(
            data.get(
                "resolved_revision"
            ),
            field=(
                "registry.roles."
                f"{role}.resolved_revision"
            ),
        ),
        path=_require_string(
            data.get(
                "path"
            ),
            field=(
                f"registry.roles.{role}.path"
            ),
        ),
    )


def _parse_registry_payload(
    raw: Any,
) -> ModelRegistryPayload:
    data = _require_mapping(
        raw,
        field="registry",
    )

    schema_version = _require_string(
        data.get(
            "schema_version"
        ),
        field="registry.schema_version",
    )

    if (
        schema_version
        != REGISTRY_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Unsupported local embedding model registry "
            f"schema: {schema_version!r}"
        )

    raw_models = _require_mapping(
        data.get(
            "models"
        ),
        field="registry.models",
    )

    raw_roles = _require_mapping(
        data.get(
            "roles"
        ),
        field="registry.roles",
    )

    models: dict[
        str,
        RegistryModelPayload,
    ] = {}

    for raw_model_id, raw_model in (
        raw_models.items()
    ):
        model_id = _require_string(
            raw_model_id,
            field="registry model id",
        )

        models[
            model_id
        ] = _parse_model_payload(
            raw_model,
            model_id=model_id,
        )

    roles: dict[
        str,
        RegistryRolePayload,
    ] = {}

    for raw_role, raw_role_payload in (
        raw_roles.items()
    ):
        role = _require_string(
            raw_role,
            field="registry role",
        ).lower()

        if role not in SUPPORTED_ROLES:
            raise RuntimeError(
                "Unsupported model registry role: "
                f"{role!r}"
            )

        roles[
            role
        ] = _parse_role_payload(
            raw_role_payload,
            role=role,
        )

    if not models:
        raise RuntimeError(
            "Local embedding model registry "
            "contains no models"
        )

    if not roles:
        raise RuntimeError(
            "Local embedding model registry "
            "contains no roles"
        )

    payload = ModelRegistryPayload(
        schema_version=schema_version,
        models=models,
        roles=roles,
    )

    updated_at = data.get(
        "updated_at_utc"
    )

    if updated_at is not None:
        payload[
            "updated_at_utc"
        ] = _require_string(
            updated_at,
            field="registry.updated_at_utc",
        )

    return payload


#--------------------REGISTRY LOADING--------------------

def _load_registry(
    models_root: Path,
) -> ModelRegistryPayload:
    registry_path = (
        models_root
        / REGISTRY_FILENAME
    )

    if not registry_path.is_file():
        raise RuntimeError(
            "Local embedding model registry not found: "
            f"{registry_path}. "
            "Provision the configured embedding model first."
        )

    try:
        raw_text = registry_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to read local embedding model registry: "
            f"{registry_path}"
        ) from exc

    try:
        raw_payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Local embedding model registry contains "
            f"invalid JSON: {registry_path}"
        ) from exc

    return _parse_registry_payload(
        raw_payload
    )


#--------------------MODEL CONFIGURATION--------------------

def _configured_model_id(
    role: str,
) -> str:
    pipeline_config = (
        get_embedding_pipeline_config()
    )

    embedding_config = (
        pipeline_config.embedding
    )

    if role == "dense":
        model_id = (
            embedding_config.model_name
        )
    else:
        model_id = (
            embedding_config.sparse_model_name
        )

    if not isinstance(
        model_id,
        str,
    ):
        raise RuntimeError(
            "Configured embedding model id for "
            f"role {role!r} must be a string"
        )

    cleaned = model_id.strip()

    if not cleaned:
        raise RuntimeError(
            "No embedding model configured for "
            f"role {role!r}"
        )

    return cleaned


#--------------------PATH RESOLUTION--------------------

def _normalize_models_root(
    models_root: str | Path | None,
) -> Path:
    root = (
        DEFAULT_MODELS_ROOT
        if models_root is None
        else Path(
            models_root
        )
    )

    resolved = (
        root
        .expanduser()
        .resolve()
    )

    if not resolved.is_dir():
        raise RuntimeError(
            "Local embedding model directory "
            f"does not exist: {resolved}"
        )

    return resolved


def _resolve_registered_path(
    raw_path: str,
    *,
    models_root: Path,
) -> Path:
    configured = Path(
        raw_path
    ).expanduser()

    if configured.is_absolute():
        candidate = configured.resolve()
    elif (
        configured.parts
        and configured.parts[0]
        == models_root.name
    ):
        candidate = (
            models_root.parent
            / configured
        ).resolve()
    else:
        candidate = (
            models_root
            / configured
        ).resolve()

    try:
        candidate.relative_to(
            models_root
        )
    except ValueError as exc:
        raise RuntimeError(
            "Registered model path escapes the local "
            f"models directory: {raw_path!r}"
        ) from exc

    return candidate


#--------------------ASSET VALIDATION--------------------

def _has_tokenizer_asset(
    tokenizer_path: Path,
) -> bool:
    candidates = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
    )

    return any(
        (
            tokenizer_path
            / filename
        ).is_file()
        for filename in candidates
    )


def _has_model_weights(
    model_path: Path,
) -> bool:
    return any(
        model_path.glob(
            "*.safetensors"
        )
    ) or any(
        model_path.glob(
            "*.bin"
        )
    )


def _validate_asset_files(
    *,
    model_path: Path,
    tokenizer_path: Path,
) -> None:
    if not model_path.is_dir():
        raise RuntimeError(
            "Registered local model path "
            f"does not exist: {model_path}"
        )

    if not tokenizer_path.is_dir():
        raise RuntimeError(
            "Registered local tokenizer path "
            f"does not exist: {tokenizer_path}"
        )

    if not (
        model_path
        / "config.json"
    ).is_file():
        raise RuntimeError(
            "Registered local model is missing "
            f"config.json: {model_path}"
        )

    if not _has_tokenizer_asset(
        tokenizer_path
    ):
        raise RuntimeError(
            "Registered local tokenizer directory "
            "contains no supported tokenizer asset: "
            f"{tokenizer_path}"
        )

    if not _has_model_weights(
        model_path
    ):
        raise RuntimeError(
            "Registered local model directory "
            "contains no model weights: "
            f"{model_path}"
        )


#--------------------ASSET RESOLUTION--------------------

def _resolve_asset(
    *,
    role: str,
    models_root: Path,
    registry: ModelRegistryPayload,
) -> LocalModelAsset:
    configured_model_id = (
        _configured_model_id(
            role
        )
    )

    roles = registry.get(
        "roles"
    )

    if roles is None:
        raise RuntimeError(
            "Local embedding model registry "
            "is missing roles"
        )

    role_payload = roles.get(
        role
    )

    if role_payload is None:
        raise RuntimeError(
            "No locally provisioned model registered "
            f"for role {role!r}"
        )

    registered_model_id = (
        role_payload[
            "model_id"
        ]
    )

    if (
        registered_model_id
        != configured_model_id
    ):
        raise RuntimeError(
            "Configured embedding model is not the "
            "model currently provisioned for "
            f"role {role!r}: configured="
            f"{configured_model_id!r}, "
            f"registered={registered_model_id!r}. "
            "Run scripts/"
            "download_embedding_model_assets.py "
            "after changing the embedding model."
        )

    revision = role_payload[
        "resolved_revision"
    ]

    models = registry.get(
        "models"
    )

    if models is None:
        raise RuntimeError(
            "Local embedding model registry "
            "is missing models"
        )

    model_payload = models.get(
        configured_model_id
    )

    if model_payload is None:
        raise RuntimeError(
            "Provisioned role references a model "
            "missing from registry.models: "
            f"{configured_model_id!r}"
        )

    installation = (
        model_payload[
            "installations"
        ].get(
            revision
        )
    )

    if installation is None:
        raise RuntimeError(
            "Provisioned role references an unknown "
            "model revision: "
            f"{configured_model_id!r}@{revision}"
        )

    role_model_path = (
        _resolve_registered_path(
            role_payload[
                "path"
            ],
            models_root=models_root,
        )
    )

    installation_model_path = (
        _resolve_registered_path(
            installation[
                "path"
            ],
            models_root=models_root,
        )
    )

    if (
        role_model_path
        != installation_model_path
    ):
        raise RuntimeError(
            "Model registry role path does not match "
            "its installation path for "
            f"{configured_model_id!r}@{revision}"
        )

    tokenizer_path = (
        _resolve_registered_path(
            installation[
                "tokenizer_path"
            ],
            models_root=models_root,
        )
    )

    if not installation[
        "offline_runtime"
    ]:
        raise RuntimeError(
            "Provisioned embedding model is not marked "
            "for offline runtime use: "
            f"{configured_model_id!r}@{revision}"
        )

    _validate_asset_files(
        model_path=installation_model_path,
        tokenizer_path=tokenizer_path,
    )

    return LocalModelAsset(
        role=role,
        model_id=configured_model_id,
        revision=revision,
        model_path=installation_model_path,
        tokenizer_path=tokenizer_path,
        offline_runtime=True,
    )


#--------------------PUBLIC API--------------------

def resolve_local_model_asset(
    *,
    role: str = "dense",
    models_root: str | Path | None = None,
) -> LocalModelAsset:
    normalized_role = (
        str(
            role
        )
        .strip()
        .lower()
    )

    if normalized_role not in SUPPORTED_ROLES:
        raise ValueError(
            "Embedding model role must be "
            "'dense' or 'sparse'"
        )

    root = _normalize_models_root(
        models_root
    )

    registry = _load_registry(
        root
    )

    return _resolve_asset(
        role=normalized_role,
        models_root=root,
        registry=registry,
    )


def resolve_embedding_model_assets(
    *,
    models_root: str | Path | None = None,
) -> dict[str, LocalModelAsset]:
    root = _normalize_models_root(
        models_root
    )

    registry = _load_registry(
        root
    )

    assets: dict[
        str,
        LocalModelAsset,
    ] = {}

    for role in sorted(
        SUPPORTED_ROLES
    ):
        assets[
            role
        ] = _resolve_asset(
            role=role,
            models_root=root,
            registry=registry,
        )

    return assets