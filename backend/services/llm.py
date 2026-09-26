from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field


from backend.config import Settings, get_settings


logger = logging.getLogger(__name__)

SchemaType = TypeVar(
    "SchemaType",
    bound=BaseModel,
)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMCallResult(BaseModel):
    agent_name: str
    provider: Literal["mock", "azure"]
    deployment: str

    status: Literal["completed", "failed"]
    content: Any = None

    latency_ms: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)

    error: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class LLMService:
    """
    Shared language-model service used by all CORTEX agents.

    It supports:

    - Deterministic mock execution
    - Azure OpenAI execution
    - Plain-text responses
    - Pydantic structured responses
    - Token and latency tracking
    - Standardized error handling
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._azure_model: AzureChatOpenAI | None = None

    @property
    def provider(self) -> Literal["mock", "azure"]:
        return self.settings.llm_provider

    @property
    def deployment(self) -> str:
        if self.provider == "mock":
            return "cortex-mock"

        return self.settings.azure_openai_deployment

    def _validate_azure_configuration(self) -> None:
        missing: list[str] = []

        if not self.settings.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")

        if not self.settings.azure_api_key_value:
            missing.append("AZURE_OPENAI_API_KEY")

        if not self.settings.azure_openai_deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")

        if not self.settings.azure_openai_api_version:
            missing.append("AZURE_OPENAI_API_VERSION")

        if missing:
            missing_names = ", ".join(missing)

            raise RuntimeError(
                "Azure OpenAI configuration is incomplete. "
                f"Missing: {missing_names}"
            )

    def _get_azure_model(self) -> AzureChatOpenAI:
        if self._azure_model is not None:
            return self._azure_model

        self._validate_azure_configuration()

        self._azure_model = AzureChatOpenAI(
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_key=self.settings.azure_api_key_value,
            azure_deployment=self.settings.azure_openai_deployment,
            api_version=self.settings.azure_openai_api_version,
            temperature=self.settings.azure_openai_temperature,
            timeout=self.settings.azure_openai_timeout_seconds,
            max_retries=self.settings.azure_openai_max_retries,
        )

        return self._azure_model

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
    ) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_prompt.strip()),
        ]

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    text_value = item.get("text")

                    if text_value:
                        parts.append(str(text_value))
                    else:
                        parts.append(str(item))

                    continue

                text_value = getattr(item, "text", None)

                if text_value:
                    parts.append(str(text_value))
                else:
                    parts.append(str(item))

            return "\n".join(parts).strip()

        if content is None:
            return ""

        return str(content).strip()

    @staticmethod
    def _extract_usage(message: Any) -> TokenUsage:
        if message is None:
            return TokenUsage()

        usage_metadata = getattr(
            message,
            "usage_metadata",
            None,
        ) or {}

        response_metadata = getattr(
            message,
            "response_metadata",
            None,
        ) or {}

        token_usage = (
            response_metadata.get("token_usage")
            or response_metadata.get("usage")
            or {}
        )

        input_tokens = int(
            usage_metadata.get(
                "input_tokens",
                token_usage.get(
                    "prompt_tokens",
                    token_usage.get("input_tokens", 0),
                ),
            )
            or 0
        )

        output_tokens = int(
            usage_metadata.get(
                "output_tokens",
                token_usage.get(
                    "completion_tokens",
                    token_usage.get("output_tokens", 0),
                ),
            )
            or 0
        )

        total_tokens = int(
            usage_metadata.get(
                "total_tokens",
                token_usage.get(
                    "total_tokens",
                    input_tokens + output_tokens,
                ),
            )
            or input_tokens + output_tokens
        )

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _extract_response_metadata(
        message: Any,
    ) -> dict[str, Any]:
        if message is None:
            return {}

        metadata = getattr(
            message,
            "response_metadata",
            None,
        )

        if not isinstance(metadata, dict):
            return {}

        safe_metadata: dict[str, Any] = {}

        for key in (
            "model_name",
            "model",
            "finish_reason",
            "system_fingerprint",
            "prompt_filter_results",
            "content_filter_results",
        ):
            if key in metadata:
                safe_metadata[key] = metadata[key]

        return safe_metadata

    def invoke_text(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        mock_text: str | None = None,
    ) -> LLMCallResult:
        started_at = perf_counter()

        if self.provider == "mock":
            content = (
                mock_text
                or "CORTEX mock LLM operational."
            )

            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            return LLMCallResult(
                agent_name=agent_name,
                provider="mock",
                deployment=self.deployment,
                status="completed",
                content=content,
                latency_ms=latency_ms,
                usage=TokenUsage(),
            )

        try:
            model = self._get_azure_model()

            messages = self._build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            response = model.invoke(messages)

            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            return LLMCallResult(
                agent_name=agent_name,
                provider="azure",
                deployment=self.deployment,
                status="completed",
                content=self._normalize_content(
                    response.content
                ),
                latency_ms=latency_ms,
                usage=self._extract_usage(response),
                response_metadata=(
                    self._extract_response_metadata(response)
                ),
            )

        except Exception as exc:
            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            logger.exception(
                "Azure OpenAI text invocation failed for agent %s",
                agent_name,
            )

            return LLMCallResult(
                agent_name=agent_name,
                provider="azure",
                deployment=self.deployment,
                status="failed",
                latency_ms=latency_ms,
                error=str(exc),
            )

    def invoke_structured(
        self,
        *,
        agent_name: str,
        schema: type[SchemaType],
        system_prompt: str,
        user_prompt: str,
        mock_value: SchemaType | dict[str, Any] | None = None,
    ) -> LLMCallResult:
        started_at = perf_counter()

        if self.provider == "mock":
            if mock_value is None:
                latency_ms = int(
                    (perf_counter() - started_at) * 1000
                )

                return LLMCallResult(
                    agent_name=agent_name,
                    provider="mock",
                    deployment=self.deployment,
                    status="failed",
                    latency_ms=latency_ms,
                    error=(
                        "mock_value is required for structured "
                        "execution in mock mode."
                    ),
                )

            try:
                if isinstance(mock_value, schema):
                    parsed_mock = mock_value
                else:
                    parsed_mock = schema.model_validate(
                        mock_value
                    )

            except Exception as exc:
                latency_ms = int(
                    (perf_counter() - started_at) * 1000
                )

                return LLMCallResult(
                    agent_name=agent_name,
                    provider="mock",
                    deployment=self.deployment,
                    status="failed",
                    latency_ms=latency_ms,
                    error=f"Invalid mock response: {exc}",
                )

            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            return LLMCallResult(
                agent_name=agent_name,
                provider="mock",
                deployment=self.deployment,
                status="completed",
                content=parsed_mock,
                latency_ms=latency_ms,
                usage=TokenUsage(),
            )

        try:
            model = self._get_azure_model()

            structured_model = model.with_structured_output(
                schema,
                method="function_calling",
                include_raw=True,
            )

            messages = self._build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            response = structured_model.invoke(messages)

            raw_message = response.get("raw")
            parsed_value = response.get("parsed")
            parsing_error = response.get("parsing_error")

            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            if parsing_error is not None:
                return LLMCallResult(
                    agent_name=agent_name,
                    provider="azure",
                    deployment=self.deployment,
                    status="failed",
                    latency_ms=latency_ms,
                    usage=self._extract_usage(raw_message),
                    error=(
                        "Structured response parsing failed: "
                        f"{parsing_error}"
                    ),
                    response_metadata=(
                        self._extract_response_metadata(
                            raw_message
                        )
                    ),
                )

            if parsed_value is None:
                return LLMCallResult(
                    agent_name=agent_name,
                    provider="azure",
                    deployment=self.deployment,
                    status="failed",
                    latency_ms=latency_ms,
                    usage=self._extract_usage(raw_message),
                    error=(
                        "Azure OpenAI returned no structured value."
                    ),
                    response_metadata=(
                        self._extract_response_metadata(
                            raw_message
                        )
                    ),
                )

            return LLMCallResult(
                agent_name=agent_name,
                provider="azure",
                deployment=self.deployment,
                status="completed",
                content=parsed_value,
                latency_ms=latency_ms,
                usage=self._extract_usage(raw_message),
                response_metadata=(
                    self._extract_response_metadata(
                        raw_message
                    )
                ),
            )

        except Exception as exc:
            latency_ms = int(
                (perf_counter() - started_at) * 1000
            )

            logger.exception(
                "Azure OpenAI structured invocation failed "
                "for agent %s",
                agent_name,
            )

            return LLMCallResult(
                agent_name=agent_name,
                provider="azure",
                deployment=self.deployment,
                status="failed",
                latency_ms=latency_ms,
                error=str(exc),
            )

    def health_check(self) -> dict[str, Any]:
        if self.provider == "mock":
            return {
                "status": "ready",
                "provider": "mock",
                "deployment": self.deployment,
                "message": "CORTEX mock LLM operational.",
                "latency_ms": 0,
            }

        result = self.invoke_text(
            agent_name="llm_health_check",
            system_prompt=(
                "You are a service health-check assistant. "
                "Follow the requested output exactly."
            ),
            user_prompt=(
                "Reply with exactly: CORTEX AZURE READY"
            ),
        )

        return {
            "status": (
                "ready"
                if result.status == "completed"
                else "failed"
            ),
            "provider": result.provider,
            "deployment": result.deployment,
            "message": (
                result.content
                if result.status == "completed"
                else result.error
            ),
            "latency_ms": result.latency_ms,
            "usage": result.usage.model_dump(),
        }


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()