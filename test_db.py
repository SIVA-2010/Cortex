from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def main() -> None:
    settings = Settings()

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
    )

    print(
        "Testing:",
        database_url.render_as_string(hide_password=True),
    )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT
                        current_user AS database_user,
                        current_database() AS database_name,
                        version() AS postgres_version
                    """
                )
            ).mappings().one()

            print("\nDatabase connection successful")
            print("User:", result["database_user"])
            print("Database:", result["database_name"])
            print("PostgreSQL:", result["postgres_version"].split(",")[0])

    except SQLAlchemyError as exc:
        print("\nDatabase connection failed")
        print(str(exc))
        raise SystemExit(1) from exc
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()