from backend.services.workflow_service import get_workflow_runtime


def main() -> None:
    runtime = get_workflow_runtime()
    print("CORTEX LangGraph checkpointer is ready.")
    print("Mode       :", runtime.settings.workflow_checkpointer)
    print("Persistent :", runtime.settings.workflow_checkpointer == "postgres")
    runtime.close()
    get_workflow_runtime.cache_clear()


if __name__ == "__main__":
    main()
