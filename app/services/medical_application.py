import re

from app.models.answer import AnswerCitation, AnswerResult, AskResponse, ClaimToVerify, ClaimVerification
from app.models.evidence import SearchRequest, SearchResponse
from app.providers.answer_base import AnswerProvider
from app.providers.claim_verifier_base import ClaimVerifier
from app.providers.europepmc import EuropePMCProvider
from app.providers.pubmed import PubMedProvider
from app.services.citation_validation import InvalidCitationError, map_citations
from app.services.search import MedicalSearchService


class MedicalApplicationError(Exception):
    """Controlled application failure shared by the REST and MCP interfaces."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def medical_search(request: SearchRequest, service: MedicalSearchService) -> SearchResponse:
    return await service.search(request)


async def pubmed_search(request: SearchRequest, provider: PubMedProvider) -> SearchResponse:
    """Return PubMed records through the existing normalized evidence schema."""
    try:
        records = await provider.search(request.query, request.limit)
    except Exception as exc:
        raise MedicalApplicationError(502, "PubMed provider failed.") from exc

    found = bool(records)
    return SearchResponse(
        query=request.query,
        language=request.language,
        answer=(
            "Retrieved PubMed records are listed below."
            if found
            else "No PubMed records matched the query."
        ),
        abstained=not found,
        uncertainty=None if found else "PubMed returned no matching records.",
        evidence=records,
        provider_status={"pubmed": "ok" if found else "no_results"},
        disclaimer=(
            "PubMed metadata and abstracts are provided as research information; "
            "this response does not provide diagnosis or treatment advice."
        ),
    )


async def europepmc_search(request: SearchRequest, provider: EuropePMCProvider) -> SearchResponse:
    """Return Europe PMC records through the shared normalized evidence schema."""
    try:
        records = await provider.search(request.query, request.limit)
    except Exception as exc:
        raise MedicalApplicationError(502, "Europe PMC provider failed.") from exc

    found = bool(records)
    return SearchResponse(
        query=request.query,
        language=request.language,
        answer=(
            "Retrieved Europe PMC records are listed below."
            if found
            else "No Europe PMC records matched the query."
        ),
        abstained=not found,
        uncertainty=None if found else "Europe PMC returned no matching records.",
        evidence=records,
        provider_status={"europepmc": "ok" if found else "no_results"},
        disclaimer=(
            "Europe PMC metadata and abstracts are provided as research information; "
            "this response does not provide diagnosis or treatment advice."
        ),
    )


async def medical_ask(
    request: SearchRequest,
    service: MedicalSearchService,
    answer_provider: AnswerProvider,
    claim_verifier: ClaimVerifier,
) -> AskResponse:
    search_result = await service.search(request)
    statuses = search_result.provider_status
    if not statuses or all(status == "unavailable" for status in statuses.values()):
        raise MedicalApplicationError(503, "Evidence search providers are unavailable.")
    if search_result.abstained or not search_result.evidence:
        return abstention(request.language, len(search_result.evidence))

    try:
        result: AnswerResult = await answer_provider.answer(
            request.query, search_result.evidence, request.language
        )
    except Exception as exc:
        raise MedicalApplicationError(502, "Answer provider failed.") from exc

    try:
        citations = map_citations(result, search_result.evidence)
    except InvalidCitationError as exc:
        raise MedicalApplicationError(502, "Answer provider returned an invalid citation.") from exc

    if result.abstained or not result.answer.strip() or not result.claims or not citations:
        return abstention(request.language, len(search_result.evidence))
    if not answer_is_covered_by_claims(result.answer, [claim.text for claim in result.claims]):
        return abstention(request.language, len(search_result.evidence))

    evidence_by_id = {
        f"evidence-{index}": record for index, record in enumerate(search_result.evidence, start=1)
    }
    claim_requests = [
        ClaimToVerify(
            claim_id=f"claim-{index}",
            text=claim.text,
            citation_ids=claim.citation_ids,
            cited_evidence=[evidence_by_id[item] for item in claim.citation_ids],
        )
        for index, claim in enumerate(result.claims, start=1)
    ]
    cited_claims = [claim for claim in claim_requests if claim.citation_ids]
    uncited_claims = [claim for claim in claim_requests if not claim.citation_ids]
    verifications = [
        ClaimVerification(claim_id=claim.claim_id, claim_text=claim.text, citation_ids=[], status="uncertain", confidence=0.0)
        for claim in uncited_claims
    ]
    if cited_claims:
        try:
            verified = await claim_verifier.verify(cited_claims)
            verified = validate_verifications(cited_claims, verified)
            verifications.extend([
                item.model_copy(update={"status": "uncertain"})
                if item.status == "supported" and item.confidence < 0.7 else item
                for item in verified
            ])
        except Exception:
            verifications.extend(uncertain_verifications(cited_claims))

    if any(item.status != "supported" for item in verifications):
        return abstention(
            request.language,
            len(search_result.evidence),
            citations=citations,
            claim_verifications=verifications,
        )
    confidence = min(result.confidence, *(item.confidence for item in verifications))
    return AskResponse(
        answer=render_verified_claims(claim_requests),
        language=request.language,
        citations=citations,
        confidence=confidence,
        abstained=False,
        evidence_count=len(search_result.evidence),
        claim_verifications=verifications,
    )


def answer_is_covered_by_claims(answer: str, claims: list[str]) -> bool:
    answer_without_citations = re.sub(r"\[?evidence-\d+\]?", "", answer, flags=re.IGNORECASE)
    sentences = [part.strip() for part in re.split(r"[.!?؟\n]+", answer_without_citations) if part.strip()]
    normalized_claims = {normalize_answer_sentence(claim) for claim in claims}
    return bool(sentences) and all(
        normalize_answer_sentence(sentence) in normalized_claims
        for sentence in sentences
    )


def normalize_answer_sentence(text: str) -> str:
    text = re.sub(r"\[?evidence-\d+\]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[.!?؟؛:،,]+$", "", text.strip())
    return " ".join(text.casefold().split())


def render_verified_claims(claims: list[ClaimToVerify]) -> str:
    return " ".join(
        f"{claim.text.rstrip()} " + " ".join(f"[{evidence_id}]" for evidence_id in claim.citation_ids)
        for claim in claims
    )


def uncertain_verifications(claims: list[ClaimToVerify]) -> list[ClaimVerification]:
    return [
        ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            citation_ids=claim.citation_ids,
            status="uncertain",
            confidence=0.0,
        )
        for claim in claims
    ]


def validate_verifications(
    claims: list[ClaimToVerify],
    verifications: list[ClaimVerification],
) -> list[ClaimVerification]:
    expected = {claim.claim_id: claim for claim in claims}
    received = {item.claim_id: item for item in verifications}
    if len(received) != len(verifications) or set(received) != set(expected):
        raise ValueError("Claim verifier returned missing, duplicate, or unknown claim IDs")
    if any(
        received[claim.claim_id].claim_text != claim.text
        or received[claim.claim_id].citation_ids != claim.citation_ids
        for claim in claims
    ):
        raise ValueError("Claim verifier changed a claim or its citation IDs")
    return [received[claim.claim_id] for claim in claims]


def abstention(
    language: str,
    evidence_count: int = 0,
    citations: list[AnswerCitation] | None = None,
    claim_verifications: list[ClaimVerification] | None = None,
) -> AskResponse:
    answer = (
        "شواهد کافی برای ارائه پاسخ قابل اتکا بازیابی نشد."
        if language == "fa"
        else "Insufficient evidence was retrieved to provide a reliable answer."
    )
    return AskResponse(
        answer=answer,
        language=language,
        citations=citations or [],
        confidence=0.0,
        abstained=True,
        evidence_count=evidence_count,
        claim_verifications=claim_verifications or [],
    )
