from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.models.answer import AskResponse
from app.models.evidence import SearchRequest, SearchResponse
from app.providers.answer_base import AnswerProvider
from app.providers.claim_verifier_base import ClaimVerifier
from app.services.medical_application import MedicalApplicationError, medical_ask, medical_search
from app.services.runtime import answer_provider_context, claim_verifier_context, search_service_context
from app.services.search import MedicalSearchService

router = APIRouter()
DEMO_PAGE = Path(__file__).resolve().parents[1] / "static" / "index.html"


async def get_service(settings: Settings = Depends(get_settings)) -> AsyncGenerator[MedicalSearchService, None]:
    async with search_service_context(settings) as service:
        yield service


async def get_answer_provider(settings: Settings = Depends(get_settings)) -> AsyncGenerator[AnswerProvider, None]:
    try:
        async with answer_provider_context(settings) as provider:
            yield provider
    except MedicalApplicationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


async def get_claim_verifier(settings: Settings = Depends(get_settings)) -> AsyncGenerator[ClaimVerifier, None]:
    try:
        async with claim_verifier_context(settings) as verifier:
            yield verifier
    except MedicalApplicationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/")
async def root() -> dict[str, str]:
    return {"service": "Medical Evidence API", "status": "running", "docs": "/docs"}


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/diagnostics")
async def diagnostics(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    """Report local provider configuration without making network calls or exposing secrets."""
    query_provider = settings.effective_query_understanding_provider
    if query_provider == "ollama":
        query_status = "configured" if settings.ollama_model and settings.ollama_model.strip() else "not_configured"
    else:
        query_status = "configured" if settings.perplexity_api_key else "not_configured"

    answer_status = "configured" if settings.answer_provider == "dev" or settings.perplexity_api_key else "not_configured"
    return {
        "api": "ok",
        "providers": {
            "pubmed": "configured",
            "europepmc": "configured",
            "query_understanding": {
                "provider": query_provider,
                "status": query_status,
            },
            "answer": {
                "provider": settings.answer_provider,
                "status": answer_status,
                "development_only": settings.answer_provider == "dev",
            },
        },
    }


@router.get("/demo", include_in_schema=False)
async def demo_page() -> FileResponse:
    return FileResponse(DEMO_PAGE, media_type="text/html; charset=utf-8")


@router.post(
    "/v1/medical/search",
    response_model=SearchResponse,
    operation_id="medical_search_v1_medical_search_post",
)
async def medical_search_route(
    request: SearchRequest,
    service: MedicalSearchService = Depends(get_service),
) -> SearchResponse:
    return await medical_search(request, service)


@router.post(
    "/v1/medical/ask",
    response_model=AskResponse,
    operation_id="medical_ask_v1_medical_ask_post",
)
async def medical_ask_route(
    request: SearchRequest,
    service: MedicalSearchService = Depends(get_service),
    answer_provider: AnswerProvider = Depends(get_answer_provider),
    claim_verifier: ClaimVerifier = Depends(get_claim_verifier),
) -> AskResponse:
    try:
        return await medical_ask(request, service, answer_provider, claim_verifier)
    except MedicalApplicationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
