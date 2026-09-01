from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.config.models.shared import validate_bool


@dataclass(frozen=True)
class ValidationConfig:
    strict_enums: bool
    reject_blank_optional_strings: bool
    reject_non_finite_scores: bool
    reject_empty_context_blocks: bool
    reject_empty_candidates: bool

    def validate(self) -> None:
        validate_bool("validation.strict_enums", self.strict_enums)
        validate_bool(
            "validation.reject_blank_optional_strings",
            self.reject_blank_optional_strings,
        )
        validate_bool(
            "validation.reject_non_finite_scores",
            self.reject_non_finite_scores,
        )
        validate_bool(
            "validation.reject_empty_context_blocks",
            self.reject_empty_context_blocks,
        )
        validate_bool(
            "validation.reject_empty_candidates",
            self.reject_empty_candidates,
        )