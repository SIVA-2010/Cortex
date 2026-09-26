from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "CORTEX"
    app_env: str = "development"
    debug: bool = True
    api_base_url: str = "http://127.0.0.1:8000"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cortex_db"
    postgres_user: str = "cortex_user"
    postgres_password: str = Field(repr=False)

    jwt_secret_key: str = Field(min_length=32, repr=False)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 240
    cors_origins: str = "http://localhost:8501"

    llm_provider: Literal["mock", "azure"] = "mock"
    openai_model: str = "gpt-5.4-mini"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr | None = Field(default=None, repr=False)
    azure_openai_deployment: str = "cortex-llm"
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_temperature: float = 0.0
    azure_openai_timeout_seconds: int = 60
    azure_openai_max_retries: int = 2
    azure_openai_input_cost_per_1m: float = 0.0
    azure_openai_output_cost_per_1m: float = 0.0
    azure_openai_shadow_deployment: str = ""
    shadowbench_use_main_if_unset: bool = True

    embedding_provider: Literal["local", "azure"] = "local"
    azure_openai_embedding_deployment: str = ""
    chroma_persist_dir: str = "./vector_store"
    chroma_collection: str = "cortex_enterprise_knowledge"

    max_workflow_retries: int = 2
    workflow_checkpointer: Literal["memory", "postgres"] = "postgres"
    workflow_poll_seconds: int = 3

    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 25
    default_complaint_csv: str = (
        "./sample_data/complaints/customer_complaints_100.csv"
    )
    default_operations_dir: str = "./sample_data/operations"

    enable_external_evaluation: bool = False

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator(
        "llm_provider",
        "embedding_provider",
        "workflow_checkpointer",
        mode="before",
    )
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return str(value).strip().lower()

    @field_validator("azure_openai_endpoint", mode="before")
    @classmethod
    def normalize_azure_endpoint(cls, value: str | None) -> str:
        if not value:
            return ""
        return f"{str(value).strip().rstrip('/')}/"

    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    @property
    def langgraph_database_uri(self) -> str:
        url = URL.create(
            drivername="postgresql",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
        return url.render_as_string(hide_password=False)

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @staticmethod
    def _resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def chroma_path(self) -> Path:
        return self._resolve_path(self.chroma_persist_dir)

    @property
    def upload_path(self) -> Path:
        return self._resolve_path(self.upload_dir)

    @property
    def default_complaint_path(self) -> Path:
        return self._resolve_path(self.default_complaint_csv)

    @property
    def default_operations_path(self) -> Path:
        return self._resolve_path(self.default_operations_dir)

    @property
    def azure_api_key_value(self) -> str:
        if self.azure_openai_api_key is None:
            return ""
        return self.azure_openai_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
