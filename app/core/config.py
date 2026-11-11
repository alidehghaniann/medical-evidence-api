from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
    perplexity_api_key: str | None = None
    answer_provider: Literal["perplexity", "dev"] = "perplexity"
    query_understanding_provider: Literal["ollama", "perplexity"] | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None
    request_timeout_seconds: float = 15.0
    max_search_results: int = 10
    minimum_evidence_score: float = 0.25

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def effective_query_understanding_provider(self) -> Literal["ollama", "perplexity"]:
        if self.query_understanding_provider is not None:
            return self.query_understanding_provider
        return "ollama" if self.app_env.strip().casefold() in {"development", "test"} else "perplexity"


@lru_cache
def get_settings() -> Settings:
    return Settings()
