from typing import Literal

from pydantic import BaseModel, Field

from backend.services.llm import get_llm_service


class StructuredHealthResponse(BaseModel):
    status: Literal["ready"] = Field(
        description="The service status."
    )

    component: str = Field(
        description="The component being tested."
    )

    message: str = Field(
        description="A short health-check message."
    )


def print_separator() -> None:
    print("-" * 64)


def main() -> None:
    service = get_llm_service()

    print_separator()
    print("CORTEX SHARED LLM SERVICE TEST")
    print_separator()

    print("Provider   :", service.provider)
    print("Deployment :", service.deployment)

    print_separator()
    print("1. Health check")
    print_separator()

    health = service.health_check()

    for key, value in health.items():
        print(f"{key:12}: {value}")

    if health["status"] != "ready":
        raise SystemExit(
            "LLM health check failed."
        )

    print_separator()
    print("2. Plain-text invocation")
    print_separator()

    text_result = service.invoke_text(
        agent_name="test_agent",
        system_prompt=(
            "You are the CORTEX test agent. "
            "Return a short operational confirmation."
        ),
        user_prompt=(
            "Confirm that the shared LLM service is operational."
        ),
        mock_text="CORTEX mock LLM operational.",
    )

    print("Status     :", text_result.status)
    print("Response   :", text_result.content)
    print("Latency ms :", text_result.latency_ms)
    print("Tokens     :", text_result.usage.total_tokens)

    if text_result.status != "completed":
        print("Error      :", text_result.error)
        raise SystemExit(
            "Plain-text LLM invocation failed."
        )

    print_separator()
    print("3. Structured invocation")
    print_separator()

    structured_result = service.invoke_structured(
        agent_name="test_structured_agent",
        schema=StructuredHealthResponse,
        system_prompt=(
            "You are validating the CORTEX structured-output "
            "service. Return only the requested structured data."
        ),
        user_prompt=(
            "Return status ready, component Shared LLM Service, "
            "and a short message confirming successful operation."
        ),
        mock_value={
            "status": "ready",
            "component": "Shared LLM Service",
            "message": (
                "CORTEX structured mock response operational."
            ),
        },
    )

    print("Status     :", structured_result.status)
    print("Latency ms :", structured_result.latency_ms)
    print("Tokens     :", structured_result.usage.total_tokens)

    if structured_result.status != "completed":
        print("Error      :", structured_result.error)
        raise SystemExit(
            "Structured LLM invocation failed."
        )

    structured_content = structured_result.content

    if isinstance(
        structured_content,
        StructuredHealthResponse,
    ):
        print(
            structured_content.model_dump_json(
                indent=2
            )
        )
    else:
        print(structured_content)

    print_separator()
    print("CORTEX shared LLM service test completed successfully.")
    print_separator()


if __name__ == "__main__":
    main()