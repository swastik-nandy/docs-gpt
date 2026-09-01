from __future__ import annotations

import os
from pathlib import Path

from rag.generation.llm.config import ModelRegistry


def main() -> None:
    _load_local_env(Path.cwd() / ".env")

    registry_path = Path(__file__).resolve().parents[1] / "config" / "models.toml"
    registry = ModelRegistry.from_toml(registry_path)

    print(f"Loaded model registry: {registry_path}")
    print()

    for role_name, role in registry.roles.items():
        print(f"[role] {role_name}")
        for slot_name in role.slots:
            try:
                resolved = registry.resolve_slot(slot_name)
                print(
                    f"  slot={slot_name} env={resolved.env_name} "
                    f"provider={resolved.provider} model={resolved.model}"
                )
            except Exception as exc:
                print(f"  slot={slot_name} unresolved: {type(exc).__name__}: {exc}")
        print()


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()
