from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator
from collections.abc import Mapping
from typing import Protocol

from rag.generation.llm.config import ModelRegistry
from rag.generation.llm.config import ProviderConfig
from rag.generation.llm.config import ResolvedModelConfig
from rag.generation.llm.providers.google_genai_provider import GoogleGenAIProvider
from rag.generation.llm.providers.groq_provider import GroqProvider
from rag.generation.llm.schema import ModelProviderError
from rag.generation.llm.schema import ModelRequest
from rag.generation.llm.schema import ModelResponse
from rag.generation.llm.schema import ModelRouterError
from rag.generation.llm.schema import ModelStreamEvent
from rag.generation.llm.schema import ModelStreamInterruptedError
from rag.generation.llm.schema import ModelUsage
from rag.generation.llm.schema import ProviderStreamChunk


#--------------------PROVIDER CONTRACT--------------------

class ModelProvider(
    Protocol
):
    @property
    def name(
        self,
    ) -> str:
        ...

    async def generate(
        self,
        request: ModelRequest,
        model: ResolvedModelConfig,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        ...

    def stream(
        self,
        request: ModelRequest,
        model: ResolvedModelConfig,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[
        ProviderStreamChunk
    ]:
        ...

    async def close(
        self,
    ) -> None:
        ...


#--------------------ROUTER--------------------

class ModelRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: Mapping[
            str,
            ModelProvider,
        ],
    ) -> None:
        self._registry = registry
        self._providers = dict(
            providers
        )

    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
    ) -> "ModelRouter":
        providers: dict[
            str,
            ModelProvider,
        ] = {}

        for (
            provider_name,
            provider_config,
        ) in registry.providers.items():
            providers[
                provider_name
            ] = _build_provider(
                provider_config
            )

        return cls(
            registry=registry,
            providers=providers,
        )

    #--------------------FULL GENERATION--------------------

    async def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        role_config = (
            self._resolve_role(
                request
            )
        )

        failures: list[str] = []

        for slot_name in role_config.slots:
            resolved = self._resolve_slot(
                slot_name=slot_name,
                failures=failures,
            )

            if resolved is None:
                continue

            provider = self._providers.get(
                resolved.provider
            )

            if provider is None:
                failures.append(
                    f"Provider "
                    f"'{resolved.provider}' "
                    "is not initialized."
                )

                continue

            provider_config = (
                self._registry.providers[
                    resolved.provider
                ]
            )

            try:
                return await asyncio.wait_for(
                    provider.generate(
                        request=request,
                        model=resolved,
                        temperature=(
                            request.temperature
                            if request.temperature
                            is not None
                            else role_config.temperature
                        ),
                        max_output_tokens=(
                            request.max_output_tokens
                            if request.max_output_tokens
                            is not None
                            else role_config.max_output_tokens
                        ),
                    ),
                    timeout=(
                        provider_config
                        .timeout_seconds
                    ),
                )

            except Exception as exc:
                failures.append(
                    _failure_text(
                        resolved,
                        exc,
                    )
                )

        raise ModelRouterError(
            "All model slots failed for "
            f"role '{request.role}'. "
            "Failures: "
            f"{' | '.join(failures)}"
        )

    #--------------------STREAMING--------------------

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[
        ModelStreamEvent
    ]:
        role_config = (
            self._resolve_role(
                request
            )
        )

        failures: list[str] = []

        for slot_index, slot_name in enumerate(
            role_config.slots
        ):
            resolved = self._resolve_slot(
                slot_name=slot_name,
                failures=failures,
            )

            if resolved is None:
                continue

            provider = self._providers.get(
                resolved.provider
            )

            if provider is None:
                failures.append(
                    f"Provider "
                    f"'{resolved.provider}' "
                    "is not initialized."
                )

                continue

            provider_config = (
                self._registry.providers[
                    resolved.provider
                ]
            )

            provider_stream = (
                provider.stream(
                    request=request,
                    model=resolved,
                    temperature=(
                        request.temperature
                        if request.temperature
                        is not None
                        else role_config.temperature
                    ),
                    max_output_tokens=(
                        request.max_output_tokens
                        if request.max_output_tokens
                        is not None
                        else role_config.max_output_tokens
                    ),
                )
            )

            committed = False

            latest_usage = (
                ModelUsage()
            )

            finish_reason: str | None = None

            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            anext(
                                provider_stream
                            ),
                            timeout=(
                                provider_config
                                .timeout_seconds
                            ),
                        )

                    except StopAsyncIteration:
                        break

                    if (
                        chunk.usage
                        is not None
                        and _usage_has_values(
                            chunk.usage
                        )
                    ):
                        latest_usage = (
                            chunk.usage
                        )

                    if chunk.raw_finish_reason:
                        finish_reason = (
                            chunk.raw_finish_reason
                        )

                    if not chunk.text:
                        continue

                    if not committed:
                        committed = True

                        yield ModelStreamEvent(
                            event_type="started",
                            provider=provider.name,
                            model=resolved.model,
                            role=request.role,
                            slot_name=slot_name,
                            fallback_used=(
                                slot_index
                                > 0
                            ),
                        )

                    yield ModelStreamEvent(
                        event_type="delta",
                        provider=provider.name,
                        model=resolved.model,
                        role=request.role,
                        slot_name=slot_name,
                        fallback_used=(
                            slot_index
                            > 0
                        ),
                        text=chunk.text,
                    )

                if not committed:
                    raise ModelProviderError(
                        f"Provider "
                        f"'{provider.name}' model "
                        f"'{resolved.model}' returned "
                        "an empty stream."
                    )

                yield ModelStreamEvent(
                    event_type="completed",
                    provider=provider.name,
                    model=resolved.model,
                    role=request.role,
                    slot_name=slot_name,
                    fallback_used=(
                        slot_index
                        > 0
                    ),
                    usage=latest_usage,
                    raw_finish_reason=(
                        finish_reason
                    ),
                )

                return

            except Exception as exc:
                await _close_stream_safely(
                    provider_stream
                )

                if committed:
                    raise ModelStreamInterruptedError(
                        "Committed model stream "
                        "was interrupted for "
                        f"{resolved.env_name}="
                        f"{resolved.model} via "
                        f"{resolved.provider}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ) from exc

                failures.append(
                    _failure_text(
                        resolved,
                        exc,
                    )
                )

                continue

        raise ModelRouterError(
            "All model slots failed for "
            f"role '{request.role}'. "
            "Failures: "
            f"{' | '.join(failures)}"
        )

    #--------------------LIFECYCLE--------------------

    async def close(
        self,
    ) -> None:
        for provider in (
            self._providers.values()
        ):
            await provider.close()

    #--------------------RESOLUTION--------------------

    def _resolve_role(
        self,
        request: ModelRequest,
    ):
        role_config = (
            self._registry.roles.get(
                request.role
            )
        )

        if role_config is None:
            raise ModelRouterError(
                "No model role configured "
                f"for '{request.role}'."
            )

        return role_config

    def _resolve_slot(
        self,
        *,
        slot_name: str,
        failures: list[str],
    ) -> ResolvedModelConfig | None:
        try:
            return self._registry.resolve_slot(
                slot_name
            )

        except Exception as exc:
            failures.append(
                f"slot '{slot_name}' "
                "resolution failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return None


#--------------------HELPERS--------------------

def _failure_text(
    resolved: ResolvedModelConfig,
    exc: Exception,
) -> str:
    return (
        f"{resolved.env_name}="
        f"{resolved.model} via "
        f"{resolved.provider} failed: "
        f"{type(exc).__name__}: "
        f"{exc}"
    )


def _usage_has_values(
    usage: ModelUsage,
) -> bool:
    return any(
        value is not None
        for value
        in (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )
    )


async def _close_stream_safely(
    stream: AsyncIterator[
        ProviderStreamChunk
    ],
) -> None:
    close_fn = getattr(
        stream,
        "aclose",
        None,
    )

    if close_fn is None:
        return

    try:
        await close_fn()

    except Exception:
        return


def _build_provider(
    config: ProviderConfig,
) -> ModelProvider:
    if config.kind == "google_genai":
        return GoogleGenAIProvider(
            config
        )

    if config.kind == "groq":
        return GroqProvider(
            config
        )

    raise ModelRouterError(
        "Unsupported provider kind: "
        f"{config.kind}"
    )
