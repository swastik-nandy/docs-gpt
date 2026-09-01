from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.retrieval.contracts.request import SourceConstraints, TenantContext


_DEV_CONFIG_PATH = Path(__file__).with_name("dev.json")


@dataclass(frozen=True)
class DevIdentityConfig:
    """
    Shared local-dev identity.

    This is intentionally outside rag.retrieval because the same identity will be
    used by retrieval, Qdrant filters, and future Postgres conversation storage.
    """

    tenant_id: str
    job_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    source_id: str | None = None
    parser_doc_id: str | None = None
    section_id: str | None = None
    ir_id: str | None = None

    def tenant_context(self) -> TenantContext:
        return TenantContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id,
        )

    def source_constraints(self) -> SourceConstraints:
        return SourceConstraints(
            job_id=self.job_id,
            source_id=self.source_id,
            parser_doc_id=self.parser_doc_id,
            section_id=self.section_id,
            ir_id=self.ir_id,
        )

    def postgres_scope(self) -> dict[str, str | None]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
        }

    def retrieval_metadata(self) -> dict[str, Any]:
        return {
            "identity_source": "tenancy.dev",
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "job_id": self.job_id,
        }


def load_dev_identity_config(
    path: str | Path | None = None,
) -> DevIdentityConfig:
    config_path = Path(path) if path is not None else _DEV_CONFIG_PATH

    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))

    tenant_id = (
        _env("DEV_TENANT_ID", "DEV_RETRIEVAL_TENANT_ID")
        or raw.get("tenant_id")
        or "debug-tenant"
    )

    return DevIdentityConfig(
        tenant_id=str(tenant_id),
        job_id=_env("DEV_JOB_ID", "DEV_RETRIEVAL_JOB_ID") or raw.get("job_id"),
        user_id=_env("DEV_USER_ID", "DEV_RETRIEVAL_USER_ID") or raw.get("user_id"),
        session_id=_env("DEV_SESSION_ID", "DEV_RETRIEVAL_SESSION_ID") or raw.get("session_id"),
        conversation_id=(
            _env("DEV_CONVERSATION_ID", "DEV_RETRIEVAL_CONVERSATION_ID")
            or raw.get("conversation_id")
        ),
        source_id=_env("DEV_SOURCE_ID", "DEV_RETRIEVAL_SOURCE_ID") or raw.get("source_id"),
        parser_doc_id=(
            _env("DEV_PARSER_DOC_ID", "DEV_RETRIEVAL_PARSER_DOC_ID")
            or raw.get("parser_doc_id")
        ),
        section_id=_env("DEV_SECTION_ID", "DEV_RETRIEVAL_SECTION_ID") or raw.get("section_id"),
        ir_id=_env("DEV_IR_ID", "DEV_RETRIEVAL_IR_ID") or raw.get("ir_id"),
    )


def load_dev_retrieval_identity_config(
    path: str | Path | None = None,
) -> DevIdentityConfig:
    return load_dev_identity_config(path)


DevRetrievalIdentityConfig = DevIdentityConfig


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()

    return None
