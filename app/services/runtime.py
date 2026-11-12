from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.config import Settings
from app.providers.answer_base import AnswerProvider
from app.providers.claim_verifier_base import ClaimVerifier
from app.providers.dev_answer import DevAnswerProvider
from app.providers.dev_claim_verifier import DevClaimVerifier
from app.providers.europepmc import EuropePMCProvider
from app.providers.ollama_query_understanding import OllamaQueryUnderstandingProvider
from app.providers.perplexity_answer import PerplexityAnswerProvider
from app.providers.perplexity_claim_verifier import PerplexityClaimVerifier
from app.providers.pubmed import PubMedProvider
from app.services.medical_application import MedicalApplicationError
from app.services.query_understanding import QueryUnderstander
from app.services.search import MedicalSearchService


@asynccontextmanager
async def search_service_context(settings: Settings) -> AsyncIterator[MedicalSearchService]:
    providers = [PubMedProvider(settings), EuropePMCProvider(settings)]
    try:
        async with query_understander_context(settings) as query_understander:
            yield MedicalSearchService(providers, settings, query_understander=query_understander)
    finally:
        for provider in providers:
            await provider.close()


@asynccontextmanager
async def query_understander_context(settings: Settings) -> AsyncIterator[QueryUnderstander | None]:
    """Select the configured local or hosted query-understanding provider."""
    if settings.effective_query_understanding_provider == "ollama":
        if not settings.ollama_model or not settings.ollama_model.strip():
            yield None
            return
        provider = OllamaQueryUnderstandingProvider(settings)
    else:
        if not settings.perplexity_api_key:
            yield None
            return
        provider = PerplexityAnswerProvider(settings)
    try:
        yield provider
    finally:
        await provider.close()


@asynccontextmanager
async def pubmed_provider_context(settings: Settings) -> AsyncIterator[PubMedProvider]:
    provider = PubMedProvider(settings)
    try:
        yield provider
    finally:
        await provider.close()


@asynccontextmanager
async def europepmc_provider_context(settings: Settings) -> AsyncIterator[EuropePMCProvider]:
    provider = EuropePMCProvider(settings)
    try:
        yield provider
    finally:
        await provider.close()


@asynccontextmanager
async def answer_provider_context(settings: Settings) -> AsyncIterator[AnswerProvider]:
    if settings.answer_provider == "dev":
        _require_development_environment(settings)
        provider = DevAnswerProvider()
        try:
            yield provider
        finally:
            await provider.close()
        return
    if not settings.perplexity_api_key:
        raise MedicalApplicationError(503, "No answer provider is configured.", "answer_provider_not_configured")
    provider = PerplexityAnswerProvider(settings)
    try:
        yield provider
    finally:
        await provider.close()


@asynccontextmanager
async def claim_verifier_context(settings: Settings) -> AsyncIterator[ClaimVerifier]:
    if settings.answer_provider == "dev":
        _require_development_environment(settings)
        verifier = DevClaimVerifier()
        try:
            yield verifier
        finally:
            await verifier.close()
        return
    if not settings.perplexity_api_key:
        raise MedicalApplicationError(503, "No claim verifier is configured.", "claim_verification_failed")
    verifier = PerplexityClaimVerifier(settings)
    try:
        yield verifier
    finally:
        await verifier.close()


def _require_development_environment(settings: Settings) -> None:
    if settings.app_env.strip().casefold() not in {"development", "test"}:
        raise MedicalApplicationError(
            503,
            "Development answer and verification providers are available only when APP_ENV is development or test.",
        )
