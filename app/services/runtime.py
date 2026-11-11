from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.config import Settings
from app.providers.answer_base import AnswerProvider
from app.providers.claim_verifier_base import ClaimVerifier
from app.providers.europepmc import EuropePMCProvider
from app.providers.perplexity import PerplexityProvider
from app.providers.perplexity_answer import PerplexityAnswerProvider
from app.providers.perplexity_claim_verifier import PerplexityClaimVerifier
from app.providers.pubmed import PubMedProvider
from app.services.medical_application import MedicalApplicationError
from app.services.search import MedicalSearchService


@asynccontextmanager
async def search_service_context(settings: Settings) -> AsyncIterator[MedicalSearchService]:
    providers = [PubMedProvider(settings)]
    if settings.perplexity_api_key:
        providers.append(PerplexityProvider(settings))
    try:
        yield MedicalSearchService(providers, settings)
    finally:
        for provider in providers:
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
    if not settings.perplexity_api_key:
        raise MedicalApplicationError(503, "No answer provider is configured.")
    provider = PerplexityAnswerProvider(settings)
    try:
        yield provider
    finally:
        await provider.close()


@asynccontextmanager
async def claim_verifier_context(settings: Settings) -> AsyncIterator[ClaimVerifier]:
    if not settings.perplexity_api_key:
        raise MedicalApplicationError(503, "No claim verifier is configured.")
    verifier = PerplexityClaimVerifier(settings)
    try:
        yield verifier
    finally:
        await verifier.close()
