from contextlib import asynccontextmanager

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

import app.mcp_server as mcp_server
from app.core.config import Settings
from app.models.answer import AnswerClaim, AnswerResult, ClaimVerification, ProviderCitation
from app.models.evidence import EvidenceRecord, SearchRequest, SearchResponse
from app.services import medical_application
from app.services import runtime as provider_runtime
from app.services.search import MedicalSearchService


def evidence_record():
    return EvidenceRecord(
        source="pubmed", source_id="123", title="Treatment evidence", abstract="Relevant study",
        url="https://pubmed.ncbi.nlm.nih.gov/123/", retrieved_at="2026-01-01T00:00:00Z",
    )


class FakeEvidenceProvider:
    name = "fake"

    def __init__(self, fail=False):
        self.fail = fail

    async def search(self, query, limit=10):
        if self.fail:
            raise httpx.ConnectError("offline")
        return [evidence_record()]


class FakeAnswerProvider:
    def __init__(self, result=None, fail=False):
        self.fail = fail
        self.result = result or AnswerResult(
            answer="The supplied evidence describes the treatment. [evidence-1]",
            confidence=0.8,
            abstained=False,
            citations=[ProviderCitation(evidence_id="evidence-1")],
            claims=[AnswerClaim(
                text="The supplied evidence describes the treatment.",
                citation_ids=["evidence-1"],
            )],
        )

    async def answer(self, query, evidence, language):
        if self.fail:
            raise httpx.ConnectError("offline")
        return self.result


class FakeVerifier:
    def __init__(self, status="supported"):
        self.status = status
        self.calls = 0

    async def verify(self, claims):
        self.calls += 1
        return [ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            citation_ids=claim.citation_ids,
            status=self.status,
            confidence=0.9,
        ) for claim in claims]


@pytest.fixture
def runtime(monkeypatch):
    settings = Settings(perplexity_api_key="test-key", minimum_evidence_score=0.0)
    service = MedicalSearchService([FakeEvidenceProvider()], settings)
    answer_provider = FakeAnswerProvider()
    verifier = FakeVerifier()

    @asynccontextmanager
    async def service_context(_settings):
        yield service

    @asynccontextmanager
    async def answer_context(_settings):
        yield answer_provider

    @asynccontextmanager
    async def verifier_context(_settings):
        yield verifier

    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_server, "search_service_context", service_context)
    monkeypatch.setattr(mcp_server, "answer_provider_context", answer_context)
    monkeypatch.setattr(mcp_server, "claim_verifier_context", verifier_context)
    return service, answer_provider, verifier


def test_mcp_server_imports_and_publishes_typed_tools():
    assert isinstance(mcp_server.mcp, FastMCP)

    async def inspect_tools():
        return await mcp_server.mcp.list_tools()

    import asyncio

    tools = asyncio.run(inspect_tools())
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"medical_search", "medical_ask", "pubmed_search", "europepmc_search"}
    assert by_name["medical_search"].inputSchema["properties"]["query"]["type"] == "string"
    assert "outputSchema" in by_name["medical_search"].model_dump(by_alias=True)
    assert by_name["pubmed_search"].inputSchema["properties"]["language"]["enum"] == ["en", "fa"]
    assert by_name["europepmc_search"].inputSchema["properties"]["language"]["enum"] == ["en", "fa"]


@pytest.mark.asyncio
async def test_medical_search_tool_calls_shared_search_service(runtime):
    service, _, _ = runtime
    response = await mcp_server.medical_search("treatment evidence")
    assert response.provider_status == {"fake": "ok"}
    assert response.evidence[0].source_id == "123"


@pytest.mark.asyncio
async def test_pubmed_mcp_tool_calls_shared_application_and_provider(monkeypatch):
    class FakePubMedProvider:
        async def search(self, query, limit=10):
            assert query == "PubMed asthma studies"
            assert limit == 10
            return [evidence_record()]

    @asynccontextmanager
    async def provider_context(_settings):
        yield FakePubMedProvider()

    monkeypatch.setattr(mcp_server, "pubmed_provider_context", provider_context)
    response = await mcp_server.pubmed_search("PubMed asthma studies", "en")
    assert response.evidence[0].source_id == "123"
    assert response.provider_status == {"pubmed": "ok"}
    assert mcp_server.run_pubmed_search is medical_application.pubmed_search


@pytest.mark.asyncio
async def test_europepmc_mcp_tool_calls_shared_application_and_provider(monkeypatch):
    class FakeEuropePMCProvider:
        async def search(self, query, limit=10):
            assert query == "Europe PMC asthma literature"
            assert limit == 10
            return [evidence_record()]

    @asynccontextmanager
    async def provider_context(_settings):
        yield FakeEuropePMCProvider()

    monkeypatch.setattr(mcp_server, "europepmc_provider_context", provider_context)
    response = await mcp_server.europepmc_search("Europe PMC asthma literature", "fa")
    assert response.evidence[0].source_id == "123"
    assert response.language == "en"
    assert response.provider_status == {"europepmc": "ok"}
    assert mcp_server.run_europepmc_search is medical_application.europepmc_search


@pytest.mark.asyncio
async def test_medical_ask_tool_uses_shared_application_and_returns_verified_result(runtime):
    _, _, verifier = runtime
    response = await mcp_server.medical_ask("treatment evidence")
    assert response.abstained is False
    assert response.answer == "The supplied evidence describes the treatment. [evidence-1]"
    assert response.claim_verifications[0].status == "supported"
    assert verifier.calls == 1
    assert mcp_server.run_medical_ask is medical_application.medical_ask


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unsupported", "uncertain"])
async def test_medical_ask_preserves_core_abstention(runtime, status):
    _, _, verifier = runtime
    verifier.status = status
    response = await mcp_server.medical_ask("treatment evidence")
    assert response.abstained is True
    assert response.answer == "Insufficient evidence was retrieved to provide a reliable answer."
    assert response.claim_verifications[0].status == status


@pytest.mark.asyncio
async def test_medical_ask_preserves_search_provider_failure(runtime):
    service, _, _ = runtime
    service.providers = [FakeEvidenceProvider(fail=True)]
    with pytest.raises(ToolError, match="503: Evidence search providers are unavailable"):
        await mcp_server.medical_ask("treatment evidence")


@pytest.mark.asyncio
async def test_medical_ask_preserves_answer_provider_failure(runtime):
    _, answer_provider, _ = runtime
    answer_provider.fail = True
    with pytest.raises(ToolError, match="502: Answer provider failed"):
        await mcp_server.medical_ask("treatment evidence")


@pytest.mark.asyncio
async def test_medical_ask_preserves_unknown_citation_error(runtime):
    _, answer_provider, _ = runtime
    answer_provider.result = AnswerResult(
        answer="Unsupported citation [evidence-99]",
        confidence=0.8,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-99")],
        claims=[AnswerClaim(text="Unsupported citation", citation_ids=["evidence-99"])],
    )
    with pytest.raises(ToolError, match="502: Answer provider returned an invalid citation"):
        await mcp_server.medical_ask("treatment evidence")


@pytest.mark.asyncio
async def test_medical_ask_missing_answer_provider_is_controlled_error(runtime, monkeypatch):
    monkeypatch.setattr(mcp_server, "get_settings", lambda: Settings(perplexity_api_key=None))
    monkeypatch.setattr(mcp_server, "answer_provider_context", provider_runtime.answer_provider_context)
    with pytest.raises(ToolError, match="503: No answer provider is configured"):
        await mcp_server.medical_ask("treatment evidence")
