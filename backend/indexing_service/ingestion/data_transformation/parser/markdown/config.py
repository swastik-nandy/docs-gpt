from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import tomllib


class ConfigNode(dict[str, Any]):
    def __getattr__(
        self,
        name: str,
    ) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _node(
    value: Any,
) -> Any:
    if isinstance(value, dict):
        return ConfigNode(
            {
                key: _node(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(
            _node(item)
            for item in value
        )

    return value


@lru_cache(maxsize=1)
def get_markdown_config() -> ConfigNode:
    path = Path(__file__).with_name(
        "config.toml"
    )

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    return _node(data["markdown"])
