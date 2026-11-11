import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service
from app.core.config import Settings, get_settings
from app.main import app
from app.models.answer import AnswerClaim, AnswerResult, AskResponse, ProviderCitation
from app.models.evidence import EvidenceRecord, SearchResponse
from app.providers.dev_answer import DevAnswerProvider
from app.providers.dev_claim_verifier import DevClaimVerifier
from app.providers.perplexity_answer import PerplexityAnswerProvider
from app.services.medical_application import MedicalApplicationError, medical_ask
from app.services.runtime import answer_provider_context, claim_verifier_context

EXCERPT = "Metformin was studied in patients with chronic kidney disease."


def evidence_record(abstract: str | None = EXCERPT) -> EvidenceRecord:
    return EvidenceRecord(
        source="pubmed",
        source_id="12345",
        title="Metformin and chronic kidney disease",
        abstract=abstract,
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        retrieved_at="2026-01-01T00:00:00Z",
    )


class StaticSearchService:
    def __init__(self, records: list[EvidenceRecord], abstained: bool = False):
        self.records = records
        self.abstained = abstained

    async def search(self, request):
        return SearchResponse(
            query=request.query,
            language=request.language,
            answer="Retrieved sources are listed.",
            abstained=self.abstained,
            uncertainty=None,
            evidence=self.records,
            provider_status={"pubmed": "ok" if self.records else "no_results"},
            disclaimer="Test evidence only.",
        )


class RecordingDevClaimVerifier(DevClaimVerifier):
    def __init__(self):
        self.calls = 0

    async def verify(self, claims):
        self.calls += 1
        return await super().verify(claims)


@pytest.mark.asyncio
async def test_dev_provider_emits_only_an_exact_cited_abstract_sentence():
    result = await DevAnswerProvider().answer(
        "What evidence exists for metformin in chronic kidney disease?",
        [evidence_record()],
        "en",
    )

    assert isinstance(result, AnswerResult)
    assert result.abstained is False
    assert result.answer == EXCERPT
    assert result.claims == [AnswerClaim(text=EXCERPT, citation_ids=["evidence-1"])]
    assert result.citations == [ProviderCitation(evidence_id="evidence-1")]


@pytest.mark.asyncio
async def test_development_answer_and_verifier_return_a_typed_verified_ask_response():
    record = evidence_record()
    verifier = RecordingDevClaimVerifier()
    response = await medical_ask(
        request=_request(),
        service=StaticSearchService([record]),
        answer_provider=DevAnswerProvider(),
        claim_verifier=verifier,
    )

    assert isinstance(response, AskResponse)
    assert response.abstained is False
    assert response.answer == f"{EXCERPT} [evidence-1]"
    assert verifier.calls == 1
    assert [item.status for item in response.claim_verifications] == ["supported"]
    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "evidence-1"
    assert response.citations[0].source_id == record.source_id
    assert str(response.citations[0].url) == str(record.url)


@pytest.mark.asyncio
async def test_unmatched_query_and_english_excerpt_for_persian_language_abstain():
    provider = DevAnswerProvider()
    fictional = await provider.answer(
        "Explain the treatment options for Nimblyx-9182 syndrome.",
        [evidence_record()],
        "en",
    )
    generic_match_only = await provider.answer(
        "Explain the treatment options for Nimblyx-9182 syndrome.",
        [evidence_record("Treatment options are discussed in many studies.")],
        "en",
    )
    wrong_language = await provider.answer(
        "metformin in chronic kidney disease",
        [evidence_record()],
        "fa",
    )

    assert fictional.abstained is True
    assert fictional.citations == []
    assert generic_match_only.abstained is True
    assert generic_match_only.citations == []
    assert wrong_language.abstained is True
    assert wrong_language.claims == []


@pytest.mark.asyncio
async def test_generic_topic_overlap_does_not_answer_a_narrower_question():
    result = await DevAnswerProvider().answer(
        "What evidence exists for metformin in chronic kidney disease?",
        [evidence_record("Metformin is used to manage diabetes. Kidney disease is also studied.")],
        "en",
    )

    assert result.abstained is True
    assert result.citations == []


@pytest.mark.asyncio
async def test_abbreviations_do_not_become_clipped_answer_fragments():
    result = await DevAnswerProvider().answer(
        "What is the treatment for H. pylori?",
        [evidence_record(
            "These guidelines review H. pylori and non-H. pylori treatment options. "
            "Another sentence discusses H. pylori infection.",
        )],
        "en",
    )

    assert result.abstained is True
    assert result.answer == ""
    assert result.citations == []

@pytest.mark.asyncio
async def test_unsupported_claim_fails_closed_through_existing_medical_ask_flow():
    class UnsupportedAnswerProvider:
        async def answer(self, query, evidence, language):
            return AnswerResult(
                answer="Metformin cures chronic kidney disease.",
                confidence=0.9,
                abstained=False,
                citations=[ProviderCitation(evidence_id="evidence-1")],
                claims=[AnswerClaim(
                    text="Metformin cures chronic kidney disease.",
                    citation_ids=["evidence-1"],
                )],
            )

    response = await medical_ask(
        request=_request(),
        service=StaticSearchService([evidence_record()]),
        answer_provider=UnsupportedAnswerProvider(),
        claim_verifier=DevClaimVerifier(),
    )

    assert response.abstained is True
    assert response.answer != "Metformin cures chronic kidney disease."
    assert response.claim_verifications[0].status == "unsupported"


@pytest.mark.asyncio
async def test_empty_evidence_uses_existing_abstention_without_claim_verification():
    verifier = RecordingDevClaimVerifier()
    response = await medical_ask(
        request=_request(),
        service=StaticSearchService([], abstained=True),
        answer_provider=DevAnswerProvider(),
        claim_verifier=verifier,
    )

    assert response.abstained is True
    assert response.evidence_count == 0
    assert response.citations == []
    assert verifier.calls == 0


def test_rest_ask_uses_configured_development_provider_and_verifier():
    settings = Settings(answer_provider="dev")
    service = StaticSearchService([evidence_record()])
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as client:
            result = client.post(
                "/v1/medical/ask",
                json={"query": "What evidence exists for metformin in chronic kidney disease?", "language": "en"},
            )
    finally:
        app.dependency_overrides.clear()

    assert result.status_code == 200
    response = AskResponse.model_validate(result.json())
    assert response.abstained is False
    assert response.claim_verifications[0].status == "supported"


@pytest.mark.asyncio
async def test_dev_provider_is_explicit_and_blocked_outside_development_and_test():
    async with answer_provider_context(Settings(answer_provider="dev")) as provider:
        assert isinstance(provider, DevAnswerProvider)
    async with claim_verifier_context(Settings(answer_provider="dev")) as verifier:
        assert isinstance(verifier, DevClaimVerifier)

    for environment in ("production", "prod", "staging"):
        settings = Settings(answer_provider="dev", app_env=environment)
        with pytest.raises(MedicalApplicationError, match="only when APP_ENV is development or test"):
            async with answer_provider_context(settings):
                pytest.fail("development provider entered outside development/test")


@pytest.mark.asyncio
async def test_perplexity_remains_default_and_missing_key_stays_controlled():
    assert Settings().answer_provider == "perplexity"
    with pytest.raises(MedicalApplicationError, match="No answer provider is configured"):
        async with answer_provider_context(Settings(answer_provider="perplexity")):
            pytest.fail("Perplexity provider entered without a key")

    async with answer_provider_context(Settings(answer_provider="perplexity", perplexity_api_key="test-key")) as provider:
        assert isinstance(provider, PerplexityAnswerProvider)


def test_answer_provider_can_be_selected_through_environment(monkeypatch):
    monkeypatch.setenv("ANSWER_PROVIDER", "dev")
    assert Settings(_env_file=None).answer_provider == "dev"


def _request():
    from app.models.evidence import SearchRequest

    return SearchRequest(query="What evidence exists for metformin in chronic kidney disease?", language="en")
