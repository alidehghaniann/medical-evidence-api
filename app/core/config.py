from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
    perplexity_api_key: str | None = None
    request_timeout_seconds: float = 15.0
    max_search_results: int = 10
    minimum_evidence_score: float = 0.25

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
