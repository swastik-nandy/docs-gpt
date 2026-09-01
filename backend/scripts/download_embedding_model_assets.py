# backend/scripts/download_embedding_model_assets.py

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENV_FILE = BACKEND_ROOT / ".env"
DEFAULT_MODELS_ROOT = BACKEND_ROOT / ".models"

REGISTRY_FILE_NAME = "registry.json"
MANIFEST_FILE_NAME = "_MODEL_ASSET_MANIFEST.json"

DEFAULT_REVISION = "main"

DENSE_MODEL_ENV = "EMBEDDING_MODEL"
DENSE_REVISION_ENV = "EMBEDDING_MODEL_REVISION"

SPARSE_MODEL_ENV = "SPARSE_EMBEDDING_MODEL"
SPARSE_REVISION_ENV = "SPARSE_EMBEDDING_MODEL_REVISION"

HF_TOKEN_ENV = "HF_TOKEN"

DEFAULT_IGNORE_PATTERNS = (
    "onnx/**",
    "openvino/**",
    "*.onnx",
    "*.tflite",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
)

TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
)

MODEL_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
)


#------------------------models-------------------------

@dataclass(frozen=True)
class ModelRequest:
    model_id: str
    revision: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class InstalledModel:
    model_id: str
    requested_revision: str
    resolved_revision: str
    local_path: Path
    roles: tuple[str, ...]


#------------------------arguments-------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Provision embedding model assets for completely offline "
            "runtime ingestion."
        )
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"Environment file. Default: {DEFAULT_ENV_FILE}",
    )

    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help=f"Local model registry root. Default: {DEFAULT_MODELS_ROOT}",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force model files to be downloaded again.",
    )

    parser.add_argument(
        "--include-all",
        action="store_true",
        help=(
            "Download every repository file including ONNX and "
            "other alternate runtime formats."
        ),
    )

    return parser.parse_args()


#------------------------environment loading-------------------------

def _parse_env_value(
    raw_value: str,
) -> str:
    value = raw_value.strip()

    if not value:
        return ""

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]

    comment_index = value.find(" #")

    if comment_index >= 0:
        value = value[:comment_index]

    return value.strip()


def load_env_file(
    path: Path,
) -> dict[str, str]:
    if not path.exists():
        return {}

    if not path.is_file():
        raise RuntimeError(
            f"Environment path is not a file: {path}"
        )

    values: dict[str, str] = {}

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, raw_value = line.split(
            "=",
            1,
        )

        key = key.strip()

        if not key:
            raise RuntimeError(
                f"Invalid environment entry at {path}:{line_number}"
            )

        values[key] = _parse_env_value(
            raw_value
        )

    return values


def environment_value(
    key: str,
    file_values: Mapping[str, str],
    default: str = "",
) -> str:
    process_value = os.getenv(
        key
    )

    if process_value is not None:
        return process_value.strip()

    return str(
        file_values.get(
            key,
            default,
        )
    ).strip()


#------------------------request resolution-------------------------

def collect_model_requests(
    file_values: Mapping[str, str],
) -> tuple[ModelRequest, ...]:
    dense_model = environment_value(
        DENSE_MODEL_ENV,
        file_values,
    )

    if not dense_model:
        raise RuntimeError(
            f"{DENSE_MODEL_ENV} is not configured"
        )

    dense_revision = environment_value(
        DENSE_REVISION_ENV,
        file_values,
        DEFAULT_REVISION,
    )

    sparse_model = environment_value(
        SPARSE_MODEL_ENV,
        file_values,
        dense_model,
    )

    sparse_revision = environment_value(
        SPARSE_REVISION_ENV,
        file_values,
        dense_revision,
    )

    grouped: dict[
        tuple[str, str],
        list[str],
    ] = {}

    grouped.setdefault(
        (
            dense_model,
            dense_revision,
        ),
        [],
    ).append(
        "dense"
    )

    grouped.setdefault(
        (
            sparse_model,
            sparse_revision,
        ),
        [],
    ).append(
        "sparse"
    )

    requests: list[ModelRequest] = []

    for (
        model_id,
        revision,
    ), roles in grouped.items():
        requests.append(
            ModelRequest(
                model_id=model_id,
                revision=revision,
                roles=tuple(
                    sorted(
                        set(roles)
                    )
                ),
            )
        )

    return tuple(
        requests
    )


#------------------------hub dependency-------------------------

def _load_hub_dependencies():
    try:
        from huggingface_hub import HfApi
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model provisioning requires the "
            "'huggingface_hub' package"
        ) from exc

    return HfApi, snapshot_download


#------------------------revision resolution-------------------------

def resolve_revision(
    *,
    model_id: str,
    revision: str,
    token: str | None,
) -> str:
    HfApi, _ = _load_hub_dependencies()

    api = HfApi(
        token=token
    )

    info = api.model_info(
        repo_id=model_id,
        revision=revision,
        token=token,
    )

    resolved_revision = str(
        getattr(
            info,
            "sha",
            "",
        )
        or ""
    ).strip()

    if not resolved_revision:
        raise RuntimeError(
            "Unable to resolve immutable revision for "
            f"{model_id!r} at {revision!r}"
        )

    return resolved_revision


#------------------------local paths-------------------------

def safe_model_directory_name(
    model_id: str,
) -> str:
    normalized = model_id.strip().replace(
        "/",
        "--",
    )

    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        normalized,
    )

    normalized = normalized.strip(
        "._-"
    )

    if not normalized:
        raise RuntimeError(
            f"Unable to create local directory for model {model_id!r}"
        )

    return normalized


def installation_path(
    *,
    models_root: Path,
    model_id: str,
    resolved_revision: str,
) -> Path:
    return (
        models_root
        / safe_model_directory_name(model_id)
        / resolved_revision
    )


#------------------------download-------------------------

def download_model(
    *,
    request: ModelRequest,
    models_root: Path,
    force: bool,
    include_all: bool,
    token: str | None,
) -> InstalledModel:
    _, snapshot_download = (
        _load_hub_dependencies()
    )

    resolved_revision = resolve_revision(
        model_id=request.model_id,
        revision=request.revision,
        token=token,
    )

    local_path = installation_path(
        models_root=models_root,
        model_id=request.model_id,
        resolved_revision=resolved_revision,
    )

    local_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    ignore_patterns = (
        None
        if include_all
        else list(DEFAULT_IGNORE_PATTERNS)
    )

    snapshot_download(
        repo_id=request.model_id,
        revision=resolved_revision,
        local_dir=local_path,
        force_download=force,
        token=token,
        ignore_patterns=ignore_patterns,
    )

    installed = InstalledModel(
        model_id=request.model_id,
        requested_revision=request.revision,
        resolved_revision=resolved_revision,
        local_path=local_path.resolve(),
        roles=request.roles,
    )

    validate_installation(
        installed
    )

    write_model_manifest(
        installed
    )

    return installed


#------------------------installation validation-------------------------

def _repository_files(
    root: Path,
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and ".cache" not in path.parts
        )
    )


def validate_installation(
    installed: InstalledModel,
) -> None:
    root = installed.local_path

    if not root.is_dir():
        raise RuntimeError(
            f"Model installation directory is missing: {root}"
        )

    config_path = root / "config.json"

    if not config_path.is_file():
        raise RuntimeError(
            f"Model {installed.model_id!r} has no config.json"
        )

    tokenizer_found = any(
        (
            root
            / asset_name
        ).is_file()
        for asset_name in TOKENIZER_ASSET_NAMES
    )

    if not tokenizer_found:
        raise RuntimeError(
            "No supported tokenizer assets were found for "
            f"{installed.model_id!r}"
        )

    repository_files = _repository_files(
        root
    )

    weight_found = any(
        path.suffix.lower()
        in MODEL_WEIGHT_SUFFIXES
        for path in repository_files
    )

    if not weight_found:
        raise RuntimeError(
            "No supported model weight files were found for "
            f"{installed.model_id!r}"
        )


#------------------------model manifest-------------------------

def relative_to_backend(
    path: Path,
) -> str:
    resolved = path.resolve()

    try:
        return resolved.relative_to(
            BACKEND_ROOT
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_model_manifest(
    installed: InstalledModel,
) -> Path:
    repository_files = _repository_files(
        installed.local_path
    )

    manifest = {
        "schema_version": "local_model_assets_v1",
        "model_id": installed.model_id,
        "requested_revision": installed.requested_revision,
        "resolved_revision": installed.resolved_revision,
        "roles": list(installed.roles),
        "local_path": relative_to_backend(
            installed.local_path
        ),
        "tokenizer_path": relative_to_backend(
            installed.local_path
        ),
        "offline_runtime": True,
        "file_count": len(repository_files),
        "installed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    manifest_path = (
        installed.local_path
        / MANIFEST_FILE_NAME
    )

    write_json_atomic(
        manifest_path,
        manifest,
    )

    return manifest_path


#------------------------registry-------------------------

def load_registry(
    registry_path: Path,
) -> dict[str, Any]:
    if not registry_path.exists():
        return {
            "schema_version": "local_embedding_model_registry_v1",
            "models": {},
            "roles": {},
        }

    try:
        value = json.loads(
            registry_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read model registry: {registry_path}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Invalid model registry: {registry_path}"
        )

    return value


def update_registry(
    *,
    models_root: Path,
    installed_models: tuple[InstalledModel, ...],
) -> Path:
    registry_path = (
        models_root
        / REGISTRY_FILE_NAME
    )

    registry = load_registry(
        registry_path
    )

    models = registry.setdefault(
        "models",
        {},
    )

    roles = registry.setdefault(
        "roles",
        {},
    )

    for installed in installed_models:
        model_entry = models.setdefault(
            installed.model_id,
            {
                "installations": {},
            },
        )

        installations = model_entry.setdefault(
            "installations",
            {},
        )

        installations[
            installed.resolved_revision
        ] = {
            "requested_revision": installed.requested_revision,
            "path": relative_to_backend(
                installed.local_path
            ),
            "tokenizer_path": relative_to_backend(
                installed.local_path
            ),
            "offline_runtime": True,
        }

        for role in installed.roles:
            roles[role] = {
                "model_id": installed.model_id,
                "resolved_revision": installed.resolved_revision,
                "path": relative_to_backend(
                    installed.local_path
                ),
            }

    registry[
        "schema_version"
    ] = "local_embedding_model_registry_v1"

    registry[
        "updated_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    write_json_atomic(
        registry_path,
        registry,
    )

    return registry_path


#------------------------filesystem helpers-------------------------

def write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def remove_partial_installation(
    path: Path,
) -> None:
    if not path.exists():
        return

    shutil.rmtree(
        path,
        ignore_errors=True,
    )


#------------------------main-------------------------

def main() -> int:
    args = parse_args()

    env_file = args.env_file.resolve()
    models_root = args.models_root.resolve()

    file_values = load_env_file(
        env_file
    )

    requests = collect_model_requests(
        file_values
    )

    token = environment_value(
        HF_TOKEN_ENV,
        file_values,
    ) or None

    models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Environment: {env_file}"
    )
    print(
        f"Models root: {models_root}"
    )

    installed_models: list[InstalledModel] = []

    for request in requests:
        print()
        print(
            f"Model:    {request.model_id}"
        )
        print(
            f"Revision: {request.revision}"
        )
        print(
            f"Roles:    {', '.join(request.roles)}"
        )
        print(
            "Resolving and downloading immutable snapshot..."
        )

        try:
            installed = download_model(
                request=request,
                models_root=models_root,
                force=args.force,
                include_all=args.include_all,
                token=token,
            )
        except Exception:
            raise

        installed_models.append(
            installed
        )

        print(
            f"Installed: {installed.local_path}"
        )
        print(
            f"Commit:    {installed.resolved_revision}"
        )

    registry_path = update_registry(
        models_root=models_root,
        installed_models=tuple(
            installed_models
        ),
    )

    print()
    print(
        "Embedding model provisioning complete."
    )
    print(
        f"Registry: {registry_path}"
    )
    print(
        "Runtime ingestion can now resolve these model assets locally."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nProvisioning interrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nProvisioning failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)