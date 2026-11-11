from contextlib import AsyncExitStack
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from app.core.config import get_settings
from app.models.answer import AskResponse
from app.models.evidence import SearchRequest, SearchResponse
from app.services.medical_application import MedicalApplicationError, medical_ask as run_medical_ask
from app.services.medical_application import europepmc_search as run_europepmc_search
from app.services.medical_application import medical_search as run_medical_search
from app.services.medical_application import pubmed_search as run_pubmed_search
from app.services.runtime import answer_provider_context, claim_verifier_context, europepmc_provider_context, pubmed_provider_context, search_service_context

mcp = FastMCP("Medical Evidence API")

Query = Annotated[str, Field(min_length=3, max_length=1000)]
Language = Literal["en", "fa"]


@mcp.tool()
async def medical_search(query: Query, language: Language = "en") -> SearchResponse:
    """Retrieve and rank medical evidence using the existing evidence core."""
    request = SearchRequest(query=query, language=language)
    try:
        async with search_service_context(get_settings()) as service:
            return await run_medical_search(request, service)
    except MedicalApplicationError as exc:
        raise ToolError(f"{exc.status_code}: {exc.detail}") from exc


@mcp.tool()
async def medical_ask(query: Query, language: Language = "en") -> AskResponse:
    """Return only verified, cited claims or the core's explicit abstention."""
    request = SearchRequest(query=query, language=language)
    settings = get_settings()
    try:
        async with AsyncExitStack() as stack:
            service = await stack.enter_async_context(search_service_context(settings))
            answer_provider = await stack.enter_async_context(answer_provider_context(settings))
            claim_verifier = await stack.enter_async_context(claim_verifier_context(settings))
            return await run_medical_ask(request, service, answer_provider, claim_verifier)
    except MedicalApplicationError as exc:
        raise ToolError(f"{exc.status_code}: {exc.detail}") from exc


@mcp.tool()
async def pubmed_search(query: Query, language: Language = "en") -> SearchResponse:
    """Retrieve PubMed metadata and abstracts through NCBI E-utilities."""
    request = SearchRequest(query=query, language=language)
    try:
        async with pubmed_provider_context(get_settings()) as provider:
            return await run_pubmed_search(request, provider)
    except MedicalApplicationError as exc:
        raise ToolError(f"{exc.status_code}: {exc.detail}") from exc


@mcp.tool()
async def europepmc_search(query: Query, language: Language = "en") -> SearchResponse:
    """Retrieve Europe PMC publication metadata and abstracts."""
    request = SearchRequest(query=query, language=language)
    try:
        async with europepmc_provider_context(get_settings()) as provider:
            return await run_europepmc_search(request, provider)
    except MedicalApplicationError as exc:
        raise ToolError(f"{exc.status_code}: {exc.detail}") from exc


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
