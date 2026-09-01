from __future__ import annotations

import os

from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors
from google.genai import types

from rag.generation.llm.config import ProviderConfig
from rag.generation.llm.config import ResolvedModelConfig
from rag.generation.llm.schema import ChatMessage
from rag.generation.llm.schema import ModelProviderError
from rag.generation.llm.schema import ModelRequest
from rag.generation.llm.schema import ModelResponse
from rag.generation.llm.schema import ModelUsage
from rag.generation.llm.schema import ProviderStreamChunk


class GoogleGenAIProvider:
    def __init__(
        self,
        config: ProviderConfig,
    ) -> None:
        self._config = config
        self._client: genai.Client | None = None
        self._async_client: Any | None = None

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
        async_client = (
            self._get_async_client()
        )

        config = _generation_config(
            request=request,
            temperature=temperature,
            max_output_tokens=(
                max_output_tokens
            ),
        )

        try:
            response = (
                await async_client.models.generate_content(
                    model=model.model,
                    contents=self._serialize_messages(
                        request.messages
                    ),
                    config=config,
                )
            )

        except errors.APIError as exc:
            raise ModelProviderError(
                "Google GenAI request failed "
                f"for model '{model.model}': "
                f"{exc}"
            ) from exc

        except Exception as exc:
            raise ModelProviderError(
                "Unexpected Google GenAI failure "
                f"for model '{model.model}': "
                f"{exc}"
            ) from exc

        text = self._extract_text(
            response
        )

        if not text.strip():
            raise ModelProviderError(
                f"Google GenAI model "
                f"'{model.model}' returned "
                "empty text. "
                "debug="
                f"{self._debug_response_summary(response)}"
            )

        return ModelResponse(
            text=text,
            provider=self.name,
            model=model.model,
            role=request.role,
            usage=self._usage_from_response(
                response
            ),
            raw_finish_reason=(
                self._finish_reason_from_response(
                    response
                )
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
        async_client = (
            self._get_async_client()
        )

        config = _generation_config(
            request=request,
            temperature=temperature,
            max_output_tokens=(
                max_output_tokens
            ),
        )

        try:
            stream = (
                await async_client.models.generate_content_stream(
                    model=model.model,
                    contents=self._serialize_messages(
                        request.messages
                    ),
                    config=config,
                )
            )

            async for response in stream:
                yield ProviderStreamChunk(
                    text=self._extract_stream_text(
                        response
                    ),
                    usage=self._usage_from_response(
                        response
                    ),
                    raw_finish_reason=(
                        self._finish_reason_from_response(
                            response
                        )
                    ),
                    raw=response,
                )

        except errors.APIError as exc:
            raise ModelProviderError(
                "Google GenAI stream failed "
                f"for model '{model.model}': "
                f"{exc}"
            ) from exc

        except Exception as exc:
            if isinstance(
                exc,
                ModelProviderError,
            ):
                raise

            raise ModelProviderError(
                "Unexpected Google GenAI "
                "stream failure for model "
                f"'{model.model}': {exc}"
            ) from exc

    #--------------------LIFECYCLE--------------------

    async def close(
        self,
    ) -> None:
        if self._async_client is not None:
            close_fn = getattr(
                self._async_client,
                "aclose",
                None,
            )

            if close_fn is not None:
                await close_fn()

            self._async_client = None

        if self._client is not None:
            close_fn = getattr(
                self._client,
                "close",
                None,
            )

            if close_fn is not None:
                close_fn()

            self._client = None

    #--------------------CLIENT--------------------

    def _get_async_client(
        self,
    ) -> Any:
        if self._async_client is not None:
            return self._async_client

        api_key = os.getenv(
            self._config.api_key_env
        )

        if not api_key:
            raise ModelProviderError(
                "Missing API key env var "
                f"'{self._config.api_key_env}' "
                f"for provider '{self.name}'."
            )

        http_options = (
            types.HttpOptions(
                api_version=(
                    self._config.api_version
                )
            )
            if self._config.api_version
            else None
        )

        self._client = genai.Client(
            api_key=api_key,
            http_options=http_options,
        )

        self._async_client = (
            self._client.aio
        )

        return self._async_client

    #--------------------MESSAGES--------------------

    @staticmethod
    def _serialize_messages(
        messages: list[ChatMessage]
        | tuple[ChatMessage, ...],
    ) -> str:
        rendered: list[str] = []

        for message in messages:
            content = (
                message.content.strip()
            )

            if not content:
                continue

            rendered.append(
                f"{message.role.upper()}:\n"
                f"{content}"
            )

        return "\n\n".join(
            rendered
        ).strip()

    #--------------------TEXT EXTRACTION--------------------

    @staticmethod
    def _extract_stream_text(
        response: Any,
    ) -> str:
        output: list[str] = []

        candidates = getattr(
            response,
            "candidates",
            None,
        ) or []

        for candidate in candidates:
            content = getattr(
                candidate,
                "content",
                None,
            )

            parts = getattr(
                content,
                "parts",
                None,
            ) or []

            for part in parts:
                if bool(
                    getattr(
                        part,
                        "thought",
                        False,
                    )
                ):
                    continue

                text = getattr(
                    part,
                    "text",
                    None,
                )

                if (
                    isinstance(
                        text,
                        str,
                    )
                    and text
                ):
                    output.append(
                        text
                    )

        return "".join(
            output
        )

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:
        value = (
            GoogleGenAIProvider
            ._extract_stream_text(
                response
            )
        )

        return value.strip()

    #--------------------METADATA--------------------

    @staticmethod
    def _usage_from_response(
        response: Any,
    ) -> ModelUsage:
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage is None:
            return ModelUsage()

        return ModelUsage(
            input_tokens=getattr(
                usage,
                "prompt_token_count",
                None,
            ),
            output_tokens=getattr(
                usage,
                "candidates_token_count",
                None,
            ),
            total_tokens=getattr(
                usage,
                "total_token_count",
                None,
            ),
        )

    @staticmethod
    def _finish_reason_from_response(
        response: Any,
    ) -> str | None:
        candidates = getattr(
            response,
            "candidates",
            None,
        ) or []

        if not candidates:
            return None

        value = getattr(
            candidates[
                0
            ],
            "finish_reason",
            None,
        )

        if value is None:
            return None

        return str(
            value
        )

    @staticmethod
    def _debug_response_summary(
        response: Any,
    ) -> dict[str, Any]:
        candidates = getattr(
            response,
            "candidates",
            None,
        ) or []

        return {
            "candidate_count": len(
                candidates
            ),
            "prompt_feedback": str(
                getattr(
                    response,
                    "prompt_feedback",
                    None,
                )
            ),
            "usage_metadata": str(
                getattr(
                    response,
                    "usage_metadata",
                    None,
                )
            ),
        }


#--------------------CONFIG--------------------

def _generation_config(
    *,
    request: ModelRequest,
    temperature: float,
    max_output_tokens: int,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=request.system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=(
            request.response_mime_type
        ),
    )
