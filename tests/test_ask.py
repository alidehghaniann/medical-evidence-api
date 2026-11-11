import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_answer_provider, get_claim_verifier, get_service
from app.core.config import Settings
from app.main import app
from app.models.answer import AnswerClaim, AnswerResult, ClaimVerification, ProviderCitation
from app.models.evidence import EvidenceRecord
from app.providers.perplexity_answer import PerplexityAnswerProvider
from app.services.search import MedicalSearchService


def evidence_record():
    return EvidenceRecord(
        source="pubmed", source_id="123", title="Treatment evidence", abstract="Relevant study",
        url="https://pubmed.ncbi.nlm.nih.gov/123/", retrieved_at="2026-01-01T00:00:00Z",
    )


class FakeSearchProvider:
    name = "fake"

    def __init__(self, records=None, fail=False):
        self.records = records or []
        self.fail = fail

    async def search(self, query, limit=10):
        if self.fail:
            raise httpx.ConnectError("offline")
        return self.records[:limit]


class FakeAnswerProvider:
    def __init__(self, result=None, fail=False):
        self.result = result or AnswerResult(
            answer="The supplied evidence describes the treatment. [evidence-1]",
            confidence=0.7,
            abstained=False,
            citations=[ProviderCitation(evidence_id="evidence-1")],
            claims=[AnswerClaim(text="The supplied evidence describes the treatment.", citation_ids=["evidence-1"])],
        )
        self.fail = fail

    async def answer(self, query, evidence, language):
        if self.fail:
            raise httpx.ConnectError("offline")
        return self.result


class FakeClaimVerifier:
    def __init__(self, status="supported", confidence=0.95, fail=False):
        self.status = status
        self.confidence = confidence
        self.fail = fail
        self.calls = 0

    async def verify(self, claims):
        self.calls += 1
        if self.fail:
            raise httpx.ConnectError("offline")
        return [ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            citation_ids=claim.citation_ids,
            status=self.status,
            confidence=self.confidence,
        ) for claim in claims]


def setup_client(records=None, answer_provider=None, search_fail=False, minimum_score=0.0, claim_verifier=None):
    search_service = MedicalSearchService(
        [FakeSearchProvider(records, fail=search_fail)],
        Settings(minimum_evidence_score=minimum_score),
    )
    app.dependency_overrides[get_service] = lambda: search_service
    if answer_provider is not None:
        app.dependency_overrides[get_answer_provider] = lambda: answer_provider
        verifier = claim_verifier or FakeClaimVerifier()
        app.dependency_overrides[get_claim_verifier] = lambda: verifier
    return TestClient(app)


def clear_overrides():
    app.dependency_overrides.clear()


def test_ask_english_success_and_response_schema():
    with setup_client([evidence_record()], FakeAnswerProvider()) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence", "language": "en"})
    clear_overrides()
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The supplied evidence describes the treatment. [evidence-1]"
    assert body["language"] == "en"
    assert body["citations"] == [{
        "evidence_id": "evidence-1", "claim": None, "title": "Treatment evidence",
        "source": "pubmed", "source_id": "123", "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
    }]
    assert body["confidence"] == 0.7
    assert body["abstained"] is False
    assert body["evidence_count"] == 1
    assert body["claim_verifications"][0]["status"] == "supported"


def test_ask_persian_answer_uses_requested_language():
    result = AnswerResult(answer="پاسخ بر اساس شواهد [evidence-1]", confidence=0.6,
        abstained=False, citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(text="پاسخ بر اساس شواهد", citation_ids=["evidence-1"])])
    with setup_client([evidence_record()], FakeAnswerProvider(result)) as client:
        response = client.post("/v1/medical/ask", json={"query": "سوال پزشکی", "language": "fa"})
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["language"] == "fa"
    assert response.json()["answer"].startswith("پاسخ")


def test_insufficient_evidence_abstains_without_calling_answer_provider():
    provider = FakeAnswerProvider(fail=True)
    with setup_client([], provider) as client:
        response = client.post("/v1/medical/ask", json={"query": "unmatched question"})
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["confidence"] == 0.0
    assert response.json()["citations"] == []
    assert response.json()["evidence_count"] == 0


def test_answer_provider_failure_is_controlled_error():
    with setup_client([evidence_record()], FakeAnswerProvider(fail=True)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502


def test_search_provider_failure_is_controlled_error():
    with setup_client([], FakeAnswerProvider(), search_fail=True) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 503


def test_unknown_citation_is_rejected():
    result = AnswerResult(answer="An answer [evidence-999]", confidence=0.8, abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-999")],
        claims=[AnswerClaim(text="An answer", citation_ids=["evidence-999"])])
    with setup_client([evidence_record()], FakeAnswerProvider(result)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502
    assert "invalid citation" in response.json()["detail"]


def test_extra_factual_answer_statement_not_in_claims_abstains():
    result = AnswerResult(
        answer="Drug X is effective. Drug X also reduces mortality.",
        confidence=0.9,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(text="Drug X is effective", citation_ids=["evidence-1"])],
    )
    with setup_client([evidence_record()], FakeAnswerProvider(result)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert "reduces mortality" not in response.json()["answer"]


def test_claim_without_citation_abstains():
    result = AnswerResult(
        answer="Drug X is effective.",
        confidence=0.9,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(text="Drug X is effective", citation_ids=[])],
    )
    with setup_client([evidence_record()], FakeAnswerProvider(result)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["abstained"] is True


def test_unrecognized_source_url_in_answer_is_rejected():
    result = AnswerResult(answer="See https://untrusted.example/article [evidence-1]", confidence=0.8,
        abstained=False, citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(text="See a source", citation_ids=["evidence-1"])])
    with setup_client([evidence_record()], FakeAnswerProvider(result)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502


@pytest.mark.parametrize("status", ["unsupported", "uncertain"])
def test_unsupported_or_uncertain_claim_abstains(status):
    verifier = FakeClaimVerifier(status=status, confidence=0.8)
    with setup_client([evidence_record()], FakeAnswerProvider(), claim_verifier=verifier) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["answer"] == "Insufficient evidence was retrieved to provide a reliable answer."
    assert response.json()["claim_verifications"][0]["status"] == status


def test_multiple_claims_with_mixed_support_abstain_and_preserve_audit():
    result = AnswerResult(
        answer="Claim one. Claim two.", confidence=0.9, abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[
            AnswerClaim(text="Claim one", citation_ids=["evidence-1"]),
            AnswerClaim(text="Claim two", citation_ids=["evidence-1"]),
        ],
    )

    class MixedVerifier(FakeClaimVerifier):
        async def verify(self, claims):
            self.calls += 1
            return [ClaimVerification(
                claim_id=claim.claim_id, claim_text=claim.text, citation_ids=claim.citation_ids,
                status="supported" if index == 0 else "unsupported", confidence=0.95,
            ) for index, claim in enumerate(claims)]

    with setup_client([evidence_record()], FakeAnswerProvider(result), claim_verifier=MixedVerifier()) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["answer"] == "Insufficient evidence was retrieved to provide a reliable answer."
    assert [item["status"] for item in body["claim_verifications"]] == ["supported", "unsupported"]


def test_verifier_failure_fails_closed_as_uncertain():
    with setup_client([evidence_record()], FakeAnswerProvider(), claim_verifier=FakeClaimVerifier(fail=True)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["claim_verifications"][0]["status"] == "uncertain"


def test_low_verifier_confidence_is_downgraded_to_uncertain():
    with setup_client([evidence_record()], FakeAnswerProvider(), claim_verifier=FakeClaimVerifier(confidence=0.4)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.json()["abstained"] is True
    assert response.json()["claim_verifications"][0]["status"] == "uncertain"


def test_citation_mapping_uses_only_retrieved_metadata():
    from app.services.citation_validation import map_citations

    citations = map_citations(FakeAnswerProvider().result, [evidence_record()])
    assert citations[0].source_id == "123"
    assert str(citations[0].url) == "https://pubmed.ncbi.nlm.nih.gov/123/"


def test_empty_query_is_rejected_by_request_validation():
    with setup_client([evidence_record()], FakeAnswerProvider()) as client:
        response = client.post("/v1/medical/ask", json={"query": ""})
    clear_overrides()
    assert response.status_code == 422


def test_ask_without_configured_answer_provider_returns_controlled_error():
    with setup_client([evidence_record()]) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_perplexity_answer_provider_is_grounded_only_in_input_evidence():
    def handler(request):
        body = json.loads(request.content)
        assert request.url == "https://api.perplexity.ai/v1/agent"
        assert "tools" not in body
        assert "only from the supplied evidence" in body["instructions"]
        assert body["input"].find("evidence-1") >= 0
        return httpx.Response(200, json={"output": [{"type": "message", "content": [{
            "type": "output_text",
            "text": '{"answer":"Supported answer [evidence-1]","confidence":0.8,"abstained":false,"citations":[{"evidence_id":"evidence-1"}],"claims":[{"text":"Supported answer","citation_ids":["evidence-1"]}]}',
        }]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PerplexityAnswerProvider(Settings(perplexity_api_key="test-key"), client=client)
    result = await provider.answer("question", [evidence_record()], "en")
    assert result.citations[0].evidence_id == "evidence-1"
    await client.aclose()
