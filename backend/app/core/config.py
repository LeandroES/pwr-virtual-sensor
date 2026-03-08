from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg2://pwr_user:pwr_secret@db:5432/pwr_twin"
    postgres_user: str = "pwr_user"
    postgres_password: str = "pwr_secret"
    postgres_db: str = "pwr_twin"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    secret_key: str = "change_me_in_production"

    # CORS — comma-separated list of allowed origins.
    # In Docker the frontend Nginx proxies /api → FastAPI on the internal
    # network, so no browser CORS header is needed.  These origins cover the
    # Vite dev server and any direct API access during local development.
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
