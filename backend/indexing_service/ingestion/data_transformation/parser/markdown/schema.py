from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MarkdownParsedUnit:
    unit_id: str
    doc_id: str
    source_doc_id: str
    parse_strategy: str

    sections: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    blocks: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    meta: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    warnings: list[
        str
    ] = field(
        default_factory=list
    )

    @property
    def section_count(
        self,
    ) -> int:
        return len(
            self.sections
        )

    @property
    def block_count(
        self,
    ) -> int:
        return len(
            self.blocks
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "unit_id":
                self.unit_id,

            "doc_id":
                self.doc_id,

            "source_doc_id":
                self.source_doc_id,

            "parse_strategy":
                self.parse_strategy,

            "section_count":
                self.section_count,

            "block_count":
                self.block_count,

            "sections":
                self.sections,

            "blocks":
                self.blocks,

            "meta":
                self.meta,

            "warnings":
                self.warnings,
        }


@dataclass(frozen=True)
class MarkdownParseResult:
    doc_id: str
    source_path: str

    units: list[
        MarkdownParsedUnit
    ]

    parser_family: str = "markdown"
    parser_variant: str = "markdown"
    parse_strategy: str = "single_document"

    warnings: list[
        str
    ] = field(
        default_factory=list
    )

    metrics: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def iter_sections(
        self,
    ) -> Iterable[
        dict[str, Any]
    ]:
        for unit in self.units:
            yield from unit.sections

    def iter_blocks(
        self,
    ) -> Iterable[
        dict[str, Any]
    ]:
        for unit in self.units:
            yield from unit.blocks

    @property
    def unit_count(
        self,
    ) -> int:
        return len(
            self.units
        )

    @property
    def section_count(
        self,
    ) -> int:
        return sum(
            unit.section_count
            for unit in self.units
        )

    @property
    def block_count(
        self,
    ) -> int:
        return sum(
            unit.block_count
            for unit in self.units
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "doc_id":
                self.doc_id,

            "source_path":
                self.source_path,

            "parser_family":
                self.parser_family,

            "parser_variant":
                self.parser_variant,

            "parse_strategy":
                self.parse_strategy,

            "units": [
                unit.to_dict()
                for unit in self.units
            ],

            "warnings":
                self.warnings,

            "metrics": {
                **self.metrics,

                "unit_count":
                    self.unit_count,

                "section_count":
                    self.section_count,

                "block_count":
                    self.block_count,
            },
        }
