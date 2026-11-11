import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.models.medical_query import MedicalQuery
from app.models.evidence import SearchRequest
from app.providers.ollama_query_understanding import OllamaQueryUnderstandingProvider
from app.providers.perplexity_answer import PerplexityAnswerProvider
from app.services.runtime import query_understander_context
from app.services.search import MedicalSearchService


PERSIAN = "درمان فشار خون بالا در سالمندان چیست؟"
MIXED = "آیا metformin برای بیماری مزمن کلیه ایمن است؟"


def output_for(query: str, search_query: str, terms: list[str], intent: str) -> dict:
    from app.models.medical_query import normalize_query_text

    return {
        "original_query": query,
        "language": "fa",
        "normalized_query": normalize_query_text(query),
        "search_query": search_query,
        "medical_terms": terms,
        "intent": intent,
    }


@pytest.mark.parametrize(("query", "search_query", "terms", "intent"), [
    (PERSIAN, "hypertension treatment older adults", ["hypertension", "treatment", "older adults"], "treatment"),
    (MIXED, "metformin chronic kidney disease safety", ["metformin", "chronic kidney disease", "safety"], "safety"),
])
def test_ollama_returns_valid_medical_query_and_uses_configured_local_api(query, search_query, terms, intent):
    def handler(request):
        assert str(request.url) == "http://ollama.test:11434/api/generate"
        body = json.loads(request.content)
        assert body["model"] == "local-test-model"
        assert body["stream"] is False
        assert body["format"] == "json"
        return httpx.Response(200, json={"response": json.dumps(output_for(query, search_query, terms, intent), ensure_ascii=False)})

    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OllamaQueryUnderstandingProvider(
            Settings(ollama_base_url="http://ollama.test:11434", ollama_model="local-test-model"),
            client=client,
        )
        try:
            result = await provider.understand_query(query, output_for(query, "unused query", ["unused"], "general")["normalized_query"])
            return result
        finally:
            await client.aclose()

    result = asyncio.run(exercise())
    assert isinstance(result, MedicalQuery)
    assert result.original_query == query
    assert result.language == "fa"
    assert result.search_query == search_query
    assert result.medical_terms == terms
    assert result.intent == intent


def test_ollama_invalid_json_fails_closed():
    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"response": "not json"})
        ))
        provider = OllamaQueryUnderstandingProvider(Settings(ollama_model="local-test-model"), client=client)
        try:
            with pytest.raises(ValueError):
                await provider.understand_query(PERSIAN, PERSIAN)
        finally:
            await client.aclose()

    asyncio.run(exercise())


def test_ollama_unavailable_fails_closed_and_missing_model_is_controlled():
    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(httpx.ConnectError("offline"))
        ))
        provider = OllamaQueryUnderstandingProvider(Settings(ollama_model="local-test-model"), client=client)
        missing_model = OllamaQueryUnderstandingProvider(Settings())
        try:
            with pytest.raises(httpx.ConnectError):
                await provider.understand_query(PERSIAN, PERSIAN)
            with pytest.raises(ValueError, match="OLLAMA_MODEL"):
                await missing_model.understand_query(PERSIAN, PERSIAN)
        finally:
            await client.aclose()
            await missing_model.close()

    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_provider_selection_defaults_by_environment_and_perplexity_remains_selectable():
    async with query_understander_context(Settings(app_env="development", ollama_model="local-test-model")) as provider:
        assert isinstance(provider, OllamaQueryUnderstandingProvider)
    async with query_understander_context(Settings(app_env="production", perplexity_api_key="test-key")) as provider:
        assert isinstance(provider, PerplexityAnswerProvider)
    async with query_understander_context(Settings(
        app_env="development", query_understanding_provider="perplexity", perplexity_api_key="test-key"
    )) as provider:
        assert isinstance(provider, PerplexityAnswerProvider)


def test_query_understanding_provider_environment_configuration(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("QUERY_UNDERSTANDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "model-from-environment")
    settings = Settings(_env_file=None)
    assert settings.effective_query_understanding_provider == "ollama"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model == "model-from-environment"


def test_search_service_sends_only_ollama_search_ready_query_to_retrieval_providers():
    query = PERSIAN
    expected_search = "hypertension treatment older adults"

    class CaptureProvider:
        def __init__(self, name):
            self.name = name
            self.queries = []

        async def search(self, search_query, limit=10):
            self.queries.append(search_query)
            return []

    def handler(_request):
        return httpx.Response(200, json={
            "response": json.dumps(output_for(query, expected_search, ["hypertension", "treatment", "older adults"], "treatment"))
        })

    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        settings = Settings(ollama_model="local-test-model", minimum_evidence_score=0.0)
        understander = OllamaQueryUnderstandingProvider(settings, client=client)
        providers = [CaptureProvider("pubmed"), CaptureProvider("europepmc")]
        service = MedicalSearchService(providers, settings, query_understander=understander)
        try:
            await service.search(SearchRequest(query=query))
        finally:
            await client.aclose()
        return providers

    providers = asyncio.run(exercise())
    assert [provider.queries for provider in providers] == [[expected_search], [expected_search]]
