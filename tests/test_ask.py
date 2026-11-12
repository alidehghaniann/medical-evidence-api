import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes import get_answer_provider, get_claim_verifier, get_service
from app.core.config import Settings
from app.main import app
from app.models.answer import AnswerClaim, AnswerResult, ClaimVerification, ProviderCitation
from app.models.evidence import EvidenceRecord
from app.models.medical_query import SearchQueryRepresentation
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
        self.calls = []

    async def answer(self, query, evidence, language):
        self.calls.append((query, evidence, language))
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


class FakeQueryUnderstander:
    async def understand_query(self, query, normalized_query):
        return SearchQueryRepresentation(
            search_query="metformin chronic kidney disease",
            medical_terms=["metformin", "kidney", "disease"],
            intent="general",
        )


def setup_client(records=None, answer_provider=None, search_fail=False, minimum_score=0.0, claim_verifier=None, query_understander=None):
    search_service = MedicalSearchService(
        [FakeSearchProvider(records, fail=search_fail)],
        Settings(minimum_evidence_score=minimum_score),
        query_understander=query_understander,
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
    assert body["diagnostics"]["reason_code"] is None
    assert body["diagnostics"]["stages"]["answer_provider_execution"] == "succeeded"
    assert body["diagnostics"]["stages"]["claim_verification"] == "completed"
    assert body["diagnostics"]["counts"]["claims_extracted"] == 1
    assert body["diagnostics"]["counts"]["supported_claims"] == 1


def test_ask_persian_answer_uses_requested_language():
    result = AnswerResult(answer="پاسخ بر اساس شواهد [evidence-1]", confidence=0.6,
        abstained=False, citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(text="پاسخ بر اساس شواهد", citation_ids=["evidence-1"])])
    evidence = evidence_record().model_copy(update={"title": "Metformin chronic kidney disease"})
    answer_provider = FakeAnswerProvider(result)
    with setup_client([evidence], answer_provider, query_understander=FakeQueryUnderstander()) as client:
        response = client.post("/v1/medical/ask", json={
            "query": "متفورمین در بیماری مزمن کلیه چیست؟",
            "language": "en",
        })
    clear_overrides()
    assert response.status_code == 200
    assert response.json()["language"] == "fa"
    assert answer_provider.calls[0][2] == "fa"
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
    assert response.json()["diagnostics"]["reason_code"] == "no_evidence"
    assert response.json()["diagnostics"]["stages"]["answer_provider_execution"] == "not_started"


def test_retrieval_success_but_search_policy_rejects_answer_provider_is_not_invoked():
    provider = FakeAnswerProvider()
    with setup_client([evidence_record()], provider, minimum_score=1.0) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    body = response.json()
    assert body["abstained"] is True
    assert body["evidence_count"] == 1
    assert provider.calls == []
    assert body["diagnostics"]["reason_code"] == "insufficient_evidence"
    assert body["diagnostics"]["stages"]["answer_provider_execution"] == "not_started"


def test_answer_provider_failure_is_controlled_error(caplog):
    caplog.set_level(logging.INFO, logger="app.services.medical_application")
    provider = FakeAnswerProvider(fail=True)
    with setup_client([evidence_record()], provider) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502
    assert len(provider.calls) == 1
    assert "reason_code=answer_provider_failed" in caplog.text


def test_answer_parse_failure_is_classified_and_fails_closed(caplog):
    caplog.set_level(logging.INFO, logger="app.services.medical_application")
    class InvalidAnswerProvider(FakeAnswerProvider):
        async def answer(self, query, evidence, language):
            self.calls.append((query, evidence, language))
            raise ValueError("invalid structured answer")

    with setup_client([evidence_record()], InvalidAnswerProvider()) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502
    assert "reason_code=answer_parse_failed" in caplog.text


def test_search_provider_failure_is_controlled_error():
    with setup_client([], FakeAnswerProvider(), search_fail=True) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 503


def test_unknown_citation_is_rejected(caplog):
    caplog.set_level(logging.INFO, logger="app.services.medical_application")
    result = AnswerResult(answer="An answer [evidence-999]", confidence=0.8, abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-999")],
        claims=[AnswerClaim(text="An answer", citation_ids=["evidence-999"])])
    with setup_client([evidence_record()], FakeAnswerProvider(result)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502
    assert response.json()["detail"]["reason_code"] == "citation_mapping_failed"
    assert "invalid citation" in response.json()["detail"]["message"]
    assert "reason_code=citation_mapping_failed" in caplog.text


def test_an_extra_answer_sentence_beside_the_claims_never_reaches_the_pipeline():
    """An answer may not carry a sentence its claims do not contain.

    The provider contract now rejects that shape when the answer is built, so the
    extra sentence cannot reach coverage at all. Coverage still refuses it if a
    provider ever manages to construct such a result.
    """
    with pytest.raises(ValidationError):
        AnswerResult(
            answer="Drug X is effective. Drug X also reduces mortality.",
            confidence=0.9,
            abstained=False,
            citations=[ProviderCitation(evidence_id="evidence-1")],
            claims=[AnswerClaim(text="Drug X is effective", citation_ids=["evidence-1"])],
        )

    result = AnswerResult(
        confidence=0.9,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[
            AnswerClaim(text="Drug X is effective", citation_ids=["evidence-1"]),
            AnswerClaim(text="Drug X also reduces mortality", citation_ids=["evidence-1"]),
        ],
    )
    assert result.answer == "Drug X is effective. [evidence-1] Drug X also reduces mortality. [evidence-1]"

    covered_result = AnswerResult(
        confidence=0.9,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[
            AnswerClaim(text="Drug X is effective", citation_ids=["evidence-1"]),
            AnswerClaim(text="A second claim", citation_ids=["evidence-1"]),
        ],
    )
    verdict = FakeClaimVerifier(status="unsupported", confidence=0.9)
    with setup_client([evidence_record()], FakeAnswerProvider(covered_result), claim_verifier=verdict) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.json()["abstained"] is True
    assert verdict.calls == 1
    assert response.json()["diagnostics"]["reason_code"] == "unsupported_claims"


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


def test_an_answer_containing_a_source_url_is_rejected_by_the_provider_contract():
    """A URL cannot ride along in the answer text any more."""
    with pytest.raises(ValidationError):
        AnswerResult(answer="See https://untrusted.example/article [evidence-1]", confidence=0.8,
            abstained=False, citations=[ProviderCitation(evidence_id="evidence-1")],
            claims=[AnswerClaim(text="See a source", citation_ids=["evidence-1"])])

    url_as_claim = AnswerResult(
        confidence=0.8,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(text="See https://untrusted.example/article", citation_ids=["evidence-1"])],
    )
    with setup_client([evidence_record()], FakeAnswerProvider(url_as_claim)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.status_code == 502
    assert response.json()["detail"]["reason_code"] == "citation_mapping_failed"


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
    assert response.json()["diagnostics"]["reason_code"] == f"{status}_claims"
    assert response.json()["diagnostics"]["stages"]["claim_verification"] == "completed"


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
    assert response.json()["diagnostics"]["reason_code"] == "claim_verification_failed"
    assert response.json()["diagnostics"]["stages"]["claim_verification"] == "failed"


def test_low_verifier_confidence_is_downgraded_to_uncertain():
    with setup_client([evidence_record()], FakeAnswerProvider(), claim_verifier=FakeClaimVerifier(confidence=0.4)) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    assert response.json()["abstained"] is True
    assert response.json()["claim_verifications"][0]["status"] == "uncertain"


@pytest.mark.parametrize(("answer", "claim"), [
    # 1 - citation at the end
    ("Drug X reduced mortality [evidence-1].", "Drug X reduced mortality [evidence-1]."),
    # 2 - parenthesised citation
    ("Drug X reduced mortality (evidence-1).", "Drug X reduced mortality [evidence-1]."),
    # 3 - citation in prose
    ("Drug X reduced mortality (see evidence-1).", "Drug X reduced mortality [evidence-1]."),
    ("Drug X reduced mortality (source: evidence-1).", "Drug X reduced mortality [evidence-1]."),
    ("Drug X reduced mortality evidence-1.", "Drug X reduced mortality [evidence-1]."),
    # 5 - several citations in one sentence
    (
        "Drug X reduced mortality [evidence-1] and improved survival [evidence-2].",
        "Drug X reduced mortality and improved survival [evidence-1].",
    ),
    # 6 - harmless whitespace and punctuation differences
    ("  Drug   X reduced mortality,  [evidence-1] .", "Drug X reduced mortality"),
    ("Drug X reduced mortality  ( evidence-1 ),", "Drug X reduced mortality [evidence-1]"),
])
def test_answer_claim_coverage_normalizes_citation_placement(answer, claim):
    from app.services.medical_application import answer_claim_coverage

    assert answer_claim_coverage(answer, [claim]) == (1, 1)


def test_multi_sentence_claim_covers_each_answer_sentence():
    from app.services.medical_application import answer_claim_coverage

    answer = "Drug X reduced mortality. It improved survival [evidence-1]."
    claim = "Drug X reduced mortality. It improved survival [evidence-1]."
    assert answer_claim_coverage(answer, [claim]) == (2, 2)


def test_claims_split_across_entries_cover_a_shared_answer():
    from app.services.medical_application import answer_claim_coverage

    assert answer_claim_coverage("Drug X reduced mortality. It improved survival.", [
        "Drug X reduced mortality",
        "It improved survival",
    ]) == (2, 2)


def test_normalization_does_not_turn_an_unsupported_answer_into_covered_text():
    from app.services.medical_application import answer_claim_coverage

    # Semantically adjacent, textually different: coverage must still fail.
    assert answer_claim_coverage("Drug X completely prevents cancer.", ["Drug X may reduce mortality."]) == (1, 0)
    assert answer_claim_coverage("Drug X reduced mortality.", ["Drug X reduces mortality"]) == (1, 0)


def test_citation_wording_is_not_what_makes_a_claim_coverable():
    from app.services.medical_application import answer_claim_coverage

    # Only citation tokens differ, so the sentence is covered.
    assert answer_claim_coverage("Drug X reduced mortality (evidence-1).", ["Drug X reduced mortality"]) == (1, 1)


def test_ask_accepts_parenthesised_citation_and_multi_sentence_claim():
    result = AnswerResult(
        answer="Drug X reduced mortality (evidence-1). It improved survival (evidence-1).",
        confidence=0.8,
        abstained=False,
        citations=[ProviderCitation(evidence_id="evidence-1")],
        claims=[AnswerClaim(
            text="Drug X reduced mortality. It improved survival [evidence-1].",
            citation_ids=["evidence-1"],
        )],
    )
    verifier = FakeClaimVerifier()
    with setup_client([evidence_record()], FakeAnswerProvider(result), claim_verifier=verifier) as client:
        response = client.post("/v1/medical/ask", json={"query": "treatment evidence"})
    clear_overrides()
    body = response.json()
    assert response.status_code == 200
    assert body["abstained"] is False
    assert verifier.calls == 1
    assert body["diagnostics"]["counts"]["answer_sentences"] == 2
    assert body["diagnostics"]["counts"]["covered_answer_sentences"] == 2
    assert body["diagnostics"]["stages"]["claim_verification"] == "completed"


def test_an_answer_that_contradicts_its_own_claims_is_rejected():
    """A narrative the claims do not contain cannot be constructed at all."""
    with pytest.raises(ValidationError):
        AnswerResult(
            answer="Drug X completely prevents cancer.",
            confidence=0.8,
            abstained=False,
            citations=[ProviderCitation(evidence_id="evidence-1")],
            claims=[AnswerClaim(text="Drug X may reduce mortality.", citation_ids=["evidence-1"])],
        )


def test_repeated_claim_text_still_needs_exact_text_equality():
    from app.services.medical_application import answer_claim_coverage

    # Case and citation placement are normalized; wording is not.
    assert answer_claim_coverage("drug x reduced mortality [evidence-1]", ["Drug X reduced mortality"]) == (1, 1)
    assert answer_claim_coverage("Drug X lowered mortality [evidence-1]", ["Drug X reduced mortality"]) == (1, 0)


def test_medical_numbers_are_not_split_into_sentence_fragments():
    """A decimal point is not a sentence boundary.

    The real provider answers with "74.7% versus 52.7%" and "risk ratio 1.42, 95%
    CI 1.14-1.76". Splitting on the bare dot cut those into fragments, so the
    claim could never match its own sentence and a correct answer abstained.
    """
    from app.services.medical_application import answer_claim_coverage, normalize_answer_sentence

    claim = (
        "In adults with clarithromycin-resistant infection, P-CAB-based regimens had "
        "higher eradication success: 74.7% versus 52.7% across 12 studies "
        "(risk ratio 1.42, 95% CI 1.14-1.76)."
    )
    answer = f"In adults with clarithromycin-resistant infection, P-CAB-based regimens had higher eradication success: 74.7% versus 52.7% across 12 studies (risk ratio 1.42, 95% CI 1.14-1.76). [evidence-1]"
    assert answer_claim_coverage(answer, [claim]) == (1, 1)
    assert normalize_answer_sentence("74.7% versus 52.7%.") == "74.7% versus 52.7%"


def test_a_citation_token_on_a_sentence_boundary_does_not_merge_the_sentences():
    """A citation token sits on the boundary, never inside it.

    "... mortality [evidence-1]. It improved" and "... advantage. [evidence-3] The
    next claim" are two sentences each, and a token that is not followed by a
    sentence still splits normally.
    """
    from app.services.medical_application import _normalized_sentences

    assert _normalized_sentences("Drug X reduced mortality [evidence-1]. It improved survival [evidence-2].") == [
        "drug x reduced mortality",
        "it improved survival",
    ]
    assert _normalized_sentences("The advantage. [evidence-3] The ACG guideline says so.") == [
        "the advantage",
        "the acg guideline says so",
    ]
    assert _normalized_sentences("The advantage.[evidence-3]The ACG guideline says so.") == [
        "the advantage",
        "the acg guideline says so",
    ]


def test_a_leading_separator_left_by_a_stripped_citation_does_not_break_coverage():
    """A citation stripped at a claim boundary leaves ", " behind; it is not wording."""
    from app.services.medical_application import answer_claim_coverage

    assert answer_claim_coverage(
        "It was safe. [evidence-8], [evidence-7], [evidence-2] In a meta-analysis, success was higher.",
        ["It was safe.", "In a meta-analysis, success was higher."],
    ) == (2, 2)


def test_positional_evidence_id_shift_is_detectable_by_the_callers_map():
    """Regression guard for the positional evidence-ID design.

    `map_citations` and the claim-verification map both build `evidence-<n>` from
    the same list position. Reordering the evidence without reordering the
    answer's citation tokens silently repoints a citation at another record, so
    callers must pass one list. This test pins the current, deferred behaviour.
    """
    from app.services.citation_validation import map_citations

    first = evidence_record()
    second = evidence_record().model_copy(update={"source_id": "456", "title": "Second study"})
    result = FakeAnswerProvider().result
    assert map_citations(result, [first, second])[0].source_id == "123"
    assert map_citations(result, [second, first])[0].source_id == "456"


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
    assert response.json()["detail"]["reason_code"] == "answer_provider_not_configured"


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
            "text": '{"confidence":0.8,"abstained":false,"citations":[{"evidence_id":"evidence-1"}],"claims":[{"text":"Supported answer","citation_ids":["evidence-1"]}]}',
        }]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PerplexityAnswerProvider(Settings(perplexity_api_key="test-key"), client=client)
    result = await provider.answer("question", [evidence_record()], "en")
    assert result.citations[0].evidence_id == "evidence-1"
    await client.aclose()


@pytest.mark.asyncio
async def test_perplexity_successful_http_with_malformed_answer_is_rejected():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"output": [{"type": "message", "content": [{
            "type": "output_text", "text": "not valid answer JSON",
        }]}]})
    ))
    provider = PerplexityAnswerProvider(Settings(perplexity_api_key="test-key"), client=client)
    try:
        # Classification for the application is unchanged: a non-JSON body is an
        # unparseable answer and must fail closed.
        with pytest.raises((ValueError, ValidationError)):
            await provider.answer("question", [evidence_record()], "en")
    finally:
        await client.aclose()
