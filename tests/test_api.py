import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service
from app.core.config import Settings
from app.main import app
from app.models.evidence import EvidenceRecord
from app.services.search import MedicalSearchService


class FakeProvider:
    name = "fake"

    def __init__(self, records=None, fail=False):
        self.records = records or []
        self.fail = fail

    async def search(self, query, limit=10):
        if self.fail:
            raise httpx.ConnectError("offline")
        return self.records[:limit]


def sample_record():
    return EvidenceRecord(source="pubmed", source_id="123", title="Treatment evidence", abstract="Relevant study",
        url="https://pubmed.ncbi.nlm.nih.gov/123/", retrieved_at="2026-01-01T00:00:00Z", relevance_score=0.0)


def client_with(providers):
    service = MedicalSearchService(providers, Settings(minimum_evidence_score=0.0))
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_health_and_root():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/").json()["service"] == "Medical Evidence API"


def test_diagnostics_reports_configuration_without_secrets():
    from app.api.routes import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(
        app_env="development", answer_provider="dev", ollama_model="local-model",
        perplexity_api_key="must-not-leak",
    )
    try:
        with TestClient(app) as client:
            response = client.get("/diagnostics")
        assert response.status_code == 200
        assert response.json() == {
            "api": "ok",
            "providers": {
                "pubmed": "configured",
                "europepmc": "configured",
                "query_understanding": {"provider": "ollama", "status": "configured"},
                "answer": {"provider": "dev", "status": "configured", "development_only": True},
            },
        }
        assert "must-not-leak" not in response.text
    finally:
        app.dependency_overrides.clear()


def test_diagnostics_marks_missing_query_and_answer_configuration():
    from app.api.routes import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(
        app_env="production", query_understanding_provider="ollama", answer_provider="perplexity",
    )
    try:
        with TestClient(app) as client:
            body = client.get("/diagnostics").json()
        assert body["providers"]["query_understanding"] == {"provider": "ollama", "status": "not_configured"}
        assert body["providers"]["answer"] == {
            "provider": "perplexity", "status": "not_configured", "development_only": False,
        }
    finally:
        app.dependency_overrides.clear()


def test_medical_route_openapi_operation_ids_remain_stable():
    paths = app.openapi()["paths"]
    assert paths["/v1/medical/search"]["post"]["operationId"] == "medical_search_v1_medical_search_post"
    assert paths["/v1/medical/ask"]["post"]["operationId"] == "medical_ask_v1_medical_ask_post"


def test_request_validation():
    with client_with([FakeProvider()]) as client:
        assert client.post("/v1/medical/search", json={"query": "x", "language": "en"}).status_code == 422
        assert client.post("/v1/medical/search", json={"query": "valid question", "language": "xx"}).status_code == 422
    app.dependency_overrides.clear()


def test_provider_failure_returns_abstention_instead_of_server_error():
    with client_with([FakeProvider(fail=True)]) as client:
        response = client.post("/v1/medical/search", json={"query": "valid medical question"})
        assert response.status_code == 200
        assert response.json()["abstained"] is True
        assert response.json()["provider_status"] == {"fake": "unavailable"}
    app.dependency_overrides.clear()


def test_missing_optional_key_and_insufficient_evidence_abstain():
    settings = Settings(perplexity_api_key=None)
    from app.providers.perplexity import PerplexityProvider
    provider = PerplexityProvider(settings, client=httpx.AsyncClient())
    with client_with([FakeProvider()]) as client:
        response = client.post("/v1/medical/search", json={"query": "unmatched query"})
        assert response.json()["abstained"] is True
    app.dependency_overrides.clear()
    import asyncio
    asyncio.run(provider.close())


def test_evidence_response_includes_traceable_source():
    with client_with([FakeProvider([sample_record()])]) as client:
        response = client.post("/v1/medical/search", json={"query": "treatment evidence"})
        assert response.status_code == 200
        assert response.json()["abstained"] is False
        assert response.json()["evidence"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/123/"
        assert response.json()["disclaimer"]
    app.dependency_overrides.clear()
