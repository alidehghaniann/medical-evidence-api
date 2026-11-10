from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.evidence import SearchRequest, SearchResponse
from app.providers.perplexity import PerplexityProvider
from app.providers.pubmed import PubMedProvider
from app.services.search import MedicalSearchService

router = APIRouter()


def get_service(settings: Settings = Depends(get_settings)) -> MedicalSearchService:
    providers = [PubMedProvider(settings)]
    if settings.perplexity_api_key:
        providers.append(PerplexityProvider(settings))
    return MedicalSearchService(providers, settings)


@router.get("/")
async def root() -> dict[str, str]:
    return {"service": "Medical Evidence API", "status": "running", "docs": "/docs"}


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/medical/search", response_model=SearchResponse)
async def medical_search(request: SearchRequest, service: MedicalSearchService = Depends(get_service)) -> SearchResponse:
    try:
        return await service.search(request)
    finally:
        for provider in service.providers:
            close = getattr(provider, "close", None)
            if close:
                await close()
