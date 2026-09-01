from __future__ import annotations

import os
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rtoml


#--------------------TYPES--------------------

ProviderKind = Literal[
    "google_genai",
    "groq",
]


#--------------------ERRORS--------------------

class ModelResolutionError(
    ValueError
):
    pass


#--------------------MODELS--------------------

@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    kind: ProviderKind
    library: str
    api_key_env: str
    timeout_seconds: float
    model_patterns: tuple[
        re.Pattern[str],
        ...,
    ]
    api_version: str | None = None

    def matches_model(
        self,
        model_id: str,
    ) -> bool:
        normalized = model_id.strip()

        return any(
            pattern.fullmatch(
                normalized
            )
            for pattern
            in self.model_patterns
        )


@dataclass(frozen=True, slots=True)
class ResolvedModelConfig:
    slot_name: str
    env_name: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class RoleConfig:
    name: str
    slots: tuple[str, ...]
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    slots: dict[str, str]

    slot_providers: dict[
        str,
        str,
    ]

    providers: dict[
        str,
        ProviderConfig,
    ]

    roles: dict[
        str,
        RoleConfig,
    ]

    #--------------------LOADING--------------------

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
    ) -> "ModelRegistry":
        config_path = Path(
            path
        )

        raw = rtoml.load(
            config_path
        )

        slots = {
            str(
                slot_name
            ).strip(): str(
                env_name
            ).strip()
            for slot_name, env_name
            in raw.get(
                "slots",
                {},
            ).items()
        }

        if not slots:
            raise ValueError(
                f"No [slots] configured in "
                f"{config_path}."
            )

        providers = _load_providers(
            raw,
            config_path,
        )

        slot_providers = {
            str(
                slot_name
            ).strip(): str(
                provider_name
            ).strip()
            for slot_name, provider_name
            in raw.get(
                "slot_providers",
                {},
            ).items()
        }

        unknown_provider_slots = (
            set(
                slot_providers
            )
            - set(
                slots
            )
        )

        if unknown_provider_slots:
            raise ValueError(
                "[slot_providers] contains unknown slots: "
                f"{sorted(unknown_provider_slots)}"
            )

        for (
            slot_name,
            provider_name,
        ) in slot_providers.items():
            if provider_name not in providers:
                raise ValueError(
                    f"Slot '{slot_name}' references "
                    f"unknown provider "
                    f"'{provider_name}'."
                )

        roles = _load_roles(
            raw,
            slots=slots,
            config_path=config_path,
        )

        return cls(
            slots=slots,
            slot_providers=slot_providers,
            providers=providers,
            roles=roles,
        )

    #--------------------RESOLUTION--------------------

    def resolve_slot(
        self,
        slot_name: str,
    ) -> ResolvedModelConfig:
        env_name = self.slots.get(
            slot_name
        )

        if not env_name:
            raise ModelResolutionError(
                f"Unknown LLM slot "
                f"'{slot_name}'."
            )

        model_id = os.getenv(
            env_name,
            "",
        ).strip()

        if not model_id:
            raise ModelResolutionError(
                f"LLM slot '{slot_name}' "
                f"points to env '{env_name}', "
                "but it is empty."
            )

        explicit_provider = (
            self.slot_providers.get(
                slot_name
            )
        )

        if explicit_provider:
            return ResolvedModelConfig(
                slot_name=slot_name,
                env_name=env_name,
                provider=explicit_provider,
                model=model_id,
            )

        provider = self._infer_provider(
            model_id
        )

        return ResolvedModelConfig(
            slot_name=slot_name,
            env_name=env_name,
            provider=provider.name,
            model=model_id,
        )

    def _infer_provider(
        self,
        model_id: str,
    ) -> ProviderConfig:
        matches = [
            provider
            for provider
            in self.providers.values()
            if provider.matches_model(
                model_id
            )
        ]

        if not matches:
            raise ModelResolutionError(
                "Could not infer provider for "
                f"model id '{model_id}'. "
                "Known provider patterns: "
                f"{_describe_provider_patterns(self.providers)}"
            )

        if len(
            matches
        ) > 1:
            names = [
                provider.name
                for provider
                in matches
            ]

            raise ModelResolutionError(
                f"Model id '{model_id}' "
                "matched multiple providers: "
                f"{names}."
            )

        return matches[
            0
        ]


#--------------------PROVIDERS--------------------

def _load_providers(
    raw: dict,
    config_path: Path,
) -> dict[str, ProviderConfig]:
    providers: dict[
        str,
        ProviderConfig,
    ] = {}

    for name, item in raw.get(
        "providers",
        {},
    ).items():
        provider_name = str(
            name
        ).strip()

        provider_kind = _provider_kind(
            provider_name
        )

        patterns = tuple(
            re.compile(
                str(
                    pattern
                )
            )
            for pattern
            in item.get(
                "model_patterns",
                [],
            )
        )

        if not patterns:
            raise ValueError(
                f"Provider '{provider_name}' "
                "must define model_patterns."
            )

        providers[
            provider_name
        ] = ProviderConfig(
            name=provider_name,
            kind=provider_kind,
            library=str(
                item[
                    "library"
                ]
            ),
            api_key_env=str(
                item[
                    "api_key_env"
                ]
            ),
            timeout_seconds=float(
                item.get(
                    "timeout_seconds",
                    45.0,
                )
            ),
            api_version=item.get(
                "api_version"
            ),
            model_patterns=patterns,
        )

    if not providers:
        raise ValueError(
            f"No [providers] configured in "
            f"{config_path}."
        )

    return providers


#--------------------ROLES--------------------

def _load_roles(
    raw: dict,
    *,
    slots: dict[str, str],
    config_path: Path,
) -> dict[str, RoleConfig]:
    roles: dict[
        str,
        RoleConfig,
    ] = {}

    for name, item in raw.get(
        "roles",
        {},
    ).items():
        role_name = str(
            name
        ).strip()

        role_slots = tuple(
            str(
                slot
            ).strip()
            for slot
            in item.get(
                "slots",
                [],
            )
        )

        if not role_slots:
            raise ValueError(
                f"Role '{role_name}' must "
                "define at least one slot."
            )

        unknown_slots = (
            set(
                role_slots
            )
            - set(
                slots
            )
        )

        if unknown_slots:
            raise ValueError(
                f"Role '{role_name}' "
                "references unknown slots: "
                f"{sorted(unknown_slots)}"
            )

        roles[
            role_name
        ] = RoleConfig(
            name=role_name,
            slots=role_slots,
            temperature=float(
                item.get(
                    "temperature",
                    0.2,
                )
            ),
            max_output_tokens=int(
                item.get(
                    "max_output_tokens",
                    2048,
                )
            ),
        )

    if not roles:
        raise ValueError(
            f"No [roles] configured in "
            f"{config_path}."
        )

    return roles


#--------------------HELPERS--------------------

def _provider_kind(
    provider_name: str,
) -> ProviderKind:
    if provider_name == "google_genai":
        return "google_genai"

    if provider_name == "groq":
        return "groq"

    raise ValueError(
        f"Unsupported provider "
        f"'{provider_name}'."
    )


def _describe_provider_patterns(
    providers: dict[
        str,
        ProviderConfig,
    ],
) -> dict[str, list[str]]:
    return {
        provider_name: [
            pattern.pattern
            for pattern
            in provider.model_patterns
        ]
        for (
            provider_name,
            provider,
        )
        in providers.items()
    }
