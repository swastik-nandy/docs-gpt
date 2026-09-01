from __future__ import annotations

import inspect
import os

from collections.abc import AsyncIterator

from groq import APIError
from groq import AsyncGroq

from rag.generation.llm.config import ProviderConfig
from rag.generation.llm.config import ResolvedModelConfig
from rag.generation.llm.schema import ChatMessage
from rag.generation.llm.schema import ModelProviderError
from rag.generation.llm.schema import ModelRequest
from rag.generation.llm.schema import ModelResponse
from rag.generation.llm.schema import ModelUsage
from rag.generation.llm.schema import ProviderStreamChunk


class GroqProvider:
    def __init__(
        self,
        config: ProviderConfig,
    ) -> None:
        self._config = config
        self._client: AsyncGroq | None = None

    @property
    def name(
        self,
    ) -> str:
        return self._config.name

    #--------------------FULL GENERATION--------------------

    async def generate(
        self,
        request: ModelRequest,
        model: ResolvedModelConfig,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        client = self._get_client()

        messages = _serialize_messages(
            request.system,
            request.messages,
        )

        try:
            response = (
                await client.chat.completions.create(
                    model=model.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=(
                        max_output_tokens
                    ),
                    **_reasoning_options(
                        model.model
                    ),
                )
            )

        except APIError as exc:
            raise ModelProviderError(
                f"Groq request failed for "
                f"model '{model.model}': "
                f"{exc}"
            ) from exc

        except Exception as exc:
            raise ModelProviderError(
                f"Unexpected Groq failure for "
                f"model '{model.model}': "
                f"{exc}"
            ) from exc

        choices = getattr(
            response,
            "choices",
            None,
        ) or []

        if not choices:
            raise ModelProviderError(
                f"Groq model '{model.model}' "
                "returned no choices."
            )

        text = (
            choices[
                0
            ].message.content
            or ""
        )

        if not text.strip():
            raise ModelProviderError(
                f"Groq model '{model.model}' "
                "returned empty text."
            )

        return ModelResponse(
            text=text,
            provider=self.name,
            model=model.model,
            role=request.role,
            usage=_usage_from_groq(
                getattr(
                    response,
                    "usage",
                    None,
                )
            ),
            raw_finish_reason=(
                choices[
                    0
                ].finish_reason
            ),
            raw=response,
        )

    #--------------------STREAMING--------------------

    async def stream(
        self,
        request: ModelRequest,
        model: ResolvedModelConfig,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> AsyncIterator[
        ProviderStreamChunk
    ]:
        client = self._get_client()

        messages = _serialize_messages(
            request.system,
            request.messages,
        )

        try:
            stream = (
                await client.chat.completions.create(
                    model=model.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=(
                        max_output_tokens
                    ),
                    stream=True,
                    **_reasoning_options(
                        model.model
                    ),
                )
            )

            async for chunk in stream:
                choices = getattr(
                    chunk,
                    "choices",
                    None,
                ) or []

                text = ""
                finish_reason: str | None = None

                if choices:
                    choice = choices[
                        0
                    ]

                    delta = getattr(
                        choice,
                        "delta",
                        None,
                    )

                    text = str(
                        getattr(
                            delta,
                            "content",
                            "",
                        )
                        or ""
                    )

                    raw_finish_reason = getattr(
                        choice,
                        "finish_reason",
                        None,
                    )

                    if raw_finish_reason is not None:
                        finish_reason = str(
                            raw_finish_reason
                        )

                yield ProviderStreamChunk(
                    text=text,
                    usage=_usage_from_groq(
                        getattr(
                            chunk,
                            "usage",
                            None,
                        )
                    ),
                    raw_finish_reason=finish_reason,
                    raw=chunk,
                )

        except APIError as exc:
            raise ModelProviderError(
                f"Groq stream failed for "
                f"model '{model.model}': "
                f"{exc}"
            ) from exc

        except Exception as exc:
            if isinstance(
                exc,
                ModelProviderError,
            ):
                raise

            raise ModelProviderError(
                f"Unexpected Groq stream "
                f"failure for model "
                f"'{model.model}': {exc}"
            ) from exc

    #--------------------LIFECYCLE--------------------

    async def close(
        self,
    ) -> None:
        if self._client is None:
            return

        close_fn = getattr(
            self._client,
            "close",
            None,
        )

        if close_fn is not None:
            result = close_fn()

            if inspect.isawaitable(
                result
            ):
                await result

        self._client = None

    #--------------------CLIENT--------------------

    def _get_client(
        self,
    ) -> AsyncGroq:
        if self._client is not None:
            return self._client

        api_key = os.getenv(
            self._config.api_key_env
        )

        if not api_key:
            raise ModelProviderError(
                "Missing API key env var "
                f"'{self._config.api_key_env}' "
                f"for provider '{self.name}'."
            )

        self._client = AsyncGroq(
            api_key=api_key
        )

        return self._client


#--------------------REASONING--------------------

def _reasoning_options(
    model_id: str,
) -> dict[str, bool]:
    normalized = str(
        model_id
    ).strip().lower()

    if normalized.startswith(
        "openai/gpt-oss-"
    ):
        return {
            "include_reasoning": False,
        }

    return {}


#--------------------MESSAGES--------------------

def _serialize_messages(
    system: str | None,
    messages: list[ChatMessage]
    | tuple[ChatMessage, ...],
) -> list[dict[str, str]]:
    rendered: list[
        dict[str, str]
    ] = []

    if system and system.strip():
        rendered.append(
            {
                "role": "system",
                "content": system.strip(),
            }
        )

    for message in messages:
        content = message.content.strip()

        if not content:
            continue

        role = message.role

        if role == "tool":
            role = "user"

        rendered.append(
            {
                "role": role,
                "content": content,
            }
        )

    return rendered


#--------------------USAGE--------------------

def _usage_from_groq(
    usage: object,
) -> ModelUsage:
    if usage is None:
        return ModelUsage()

    return ModelUsage(
        input_tokens=getattr(
            usage,
            "prompt_tokens",
            None,
        ),
        output_tokens=getattr(
            usage,
            "completion_tokens",
            None,
        ),
        total_tokens=getattr(
            usage,
            "total_tokens",
            None,
        ),
    )
