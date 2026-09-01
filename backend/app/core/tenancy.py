from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.settings import Settings


TENANT_ID_RE = re.compile(r"^tnt_[a-z0-9][a-z0-9_-]{6,80}$")


class TenantResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    source: str


def validate_tenant_id(value: str) -> str:
    tenant_id = str(value or "").strip().lower()

    if not tenant_id:
        raise TenantResolutionError("tenant_id cannot be empty")

    if not TENANT_ID_RE.match(tenant_id):
        raise TenantResolutionError(
            "Invalid tenant_id. Expected opaque tenant id like "
            "'tnt_local_000000000001'. Do not use names or emails."
        )

    return tenant_id


def resolve_tenant_context(
    *,
    requested_tenant_id: str | None,
    settings: Settings,
) -> TenantContext:
    """
    Resolve tenant identity for artifact paths.

    Current local/dev behavior:
    - use settings.default_tenant_id
    - optionally allow explicit tenant_id only when enabled

    Future production behavior:
    - replace this with auth/session/DB tenant lookup
    - never trust arbitrary tenant names from client input
    """

    if settings.allow_client_tenant_id and requested_tenant_id:
        return TenantContext(
            tenant_id=validate_tenant_id(requested_tenant_id),
            source="client_override",
        )

    return TenantContext(
        tenant_id=validate_tenant_id(settings.default_tenant_id),
        source="local_default",
    )