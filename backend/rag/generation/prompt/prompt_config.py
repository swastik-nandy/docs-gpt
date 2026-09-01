from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rtoml


#--------------------MODELS--------------------

@dataclass(frozen=True, slots=True)
class PromptTemplate:
    system: str
    user_template: str


@dataclass(frozen=True, slots=True)
class PromptRendering:
    empty_value: str
    context_header_template: str
    chunk_part_template: str
    score_part_template: str
    provenance_item_template: str


@dataclass(frozen=True, slots=True)
class PromptFallbacks:
    insufficient_grounding: str
    generation_unavailable: str


@dataclass(frozen=True, slots=True)
class PromptRegistry:
    rag_answer: PromptTemplate
    turn_planning: PromptTemplate
    memory_extraction: PromptTemplate
    rendering: PromptRendering
    fallbacks: PromptFallbacks

    @classmethod
    def from_toml(
        cls,
        path: str | Path | None = None,
    ) -> "PromptRegistry":
        config_path = (
            Path(
                path
            )
            if path is not None
            else (
                Path(
                    __file__
                )
                .resolve()
                .parents[
                    1
                ]
                / "config"
                / "prompts.toml"
            )
        )

        raw = rtoml.load(
            config_path
        )

        return cls(
            rag_answer=_load_prompt_template(
                raw,
                "rag_answer",
            ),
            turn_planning=_load_prompt_template(
                raw,
                "turn_planning",
            ),
            memory_extraction=_load_prompt_template(
                raw,
                "memory_extraction",
            ),
            rendering=_load_rendering(
                raw
            ),
            fallbacks=_load_fallbacks(
                raw
            ),
        )


#--------------------PROMPT LOADING--------------------

def _load_prompt_template(
    raw: dict,
    key: str,
) -> PromptTemplate:
    section = raw.get(
        key
    )

    if not isinstance(
        section,
        dict,
    ):
        raise ValueError(
            f"Missing prompt section "
            f"[{key}]."
        )

    system = str(
        section.get(
            "system"
        )
        or ""
    ).strip()

    user_template = str(
        section.get(
            "user_template"
        )
        or ""
    ).strip()

    if not system:
        raise ValueError(
            f"Prompt section [{key}] "
            "must define non-empty system."
        )

    if not user_template:
        raise ValueError(
            f"Prompt section [{key}] "
            "must define non-empty "
            "user_template."
        )

    return PromptTemplate(
        system=system,
        user_template=user_template,
    )


#--------------------RENDERING--------------------

def _load_rendering(
    raw: dict,
) -> PromptRendering:
    section = raw.get(
        "rendering"
    )

    if not isinstance(
        section,
        dict,
    ):
        raise ValueError(
            "Missing prompt section "
            "[rendering]."
        )

    return PromptRendering(
        empty_value=_required_string(
            section,
            "empty_value",
            "rendering",
        ),
        context_header_template=_required_string(
            section,
            "context_header_template",
            "rendering",
        ),
        chunk_part_template=_required_string(
            section,
            "chunk_part_template",
            "rendering",
        ),
        score_part_template=_required_string(
            section,
            "score_part_template",
            "rendering",
        ),
        provenance_item_template=_required_string(
            section,
            "provenance_item_template",
            "rendering",
        ),
    )


#--------------------FALLBACKS--------------------

def _load_fallbacks(
    raw: dict,
) -> PromptFallbacks:
    section = raw.get(
        "fallbacks"
    )

    if not isinstance(
        section,
        dict,
    ):
        raise ValueError(
            "Missing prompt section "
            "[fallbacks]."
        )

    return PromptFallbacks(
        insufficient_grounding=_required_string(
            section,
            "insufficient_grounding",
            "fallbacks",
        ),
        generation_unavailable=_required_string(
            section,
            "generation_unavailable",
            "fallbacks",
        ),
    )


#--------------------VALIDATION--------------------

def _required_string(
    section: dict,
    key: str,
    section_name: str,
) -> str:
    value = str(
        section.get(
            key
        )
        or ""
    ).strip()

    if not value:
        raise ValueError(
            f"Prompt section "
            f"[{section_name}] "
            "must define non-empty "
            f"{key}."
        )

    return value
