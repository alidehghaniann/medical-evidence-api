import logging
import re

from app.models.answer import AnswerCitation, AnswerResult, AskResponse, ClaimToVerify, ClaimVerification
from app.models.evidence import SearchRequest, SearchResponse
from app.models.medical_query import MedicalQuery
from app.providers.answer_base import AnswerProvider
from app.providers.claim_verifier_base import ClaimVerifier
from app.providers.europepmc import EuropePMCProvider
from app.providers.pubmed import PubMedProvider
from app.services.citation_validation import InvalidCitationError, map_citations
from app.services.search import MedicalSearchService
from app.services.query_understanding import QueryUnderstander, understand_medical_query

logger = logging.getLogger(__name__)


class MedicalApplicationError(Exception):
    """Controlled application failure shared by the REST and MCP interfaces."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def medical_search(request: SearchRequest, service: MedicalSearchService) -> SearchResponse:
    return await service.search(request)


async def pubmed_search(
    request: SearchRequest,
    provider: PubMedProvider,
    query_understander: QueryUnderstander | None = None,
) -> SearchResponse:
    """Return PubMed records through the existing normalized evidence schema."""
    return await _source_search(request, provider, "pubmed", "PubMed", query_understander)


async def europepmc_search(
    request: SearchRequest,
    provider: EuropePMCProvider,
    query_understander: QueryUnderstander | None = None,
) -> SearchResponse:
    """Return Europe PMC records through the shared normalized evidence schema."""
    return await _source_search(request, provider, "europepmc", "Europe PMC", query_understander)


async def _source_search(
    request: SearchRequest,
    provider: PubMedProvider | EuropePMCProvider,
    source: str,
    display_name: str,
    query_understander: QueryUnderstander | None,
) -> SearchResponse:
    query_understanding_failed = False
    try:
        medical_query: MedicalQuery = await understand_medical_query(request.query, query_understander)
    except Exception as exc:
        logger.warning("medical query understanding failed error_type=%s", type(exc).__name__)
        query_understanding_failed = True
        medical_query = None

    if medical_query is None or medical_query.search_query is None:
        query_status = "unavailable" if query_understanding_failed else "not_configured"
        return SearchResponse(
            query=request.query,
            language=request.language,
            answer=("عبارت جست‌وجوی قابل اتکایی آماده نشد؛ جست‌وجو انجام نشد."
                    if request.language == "fa" else "A reliable search query could not be prepared; no search was performed."),
            abstained=True,
            uncertainty="No safe search-ready representation was available.",
            evidence=[],
            provider_status={source: "skipped_query_understanding", "query_understanding": query_status},
            abstention_reason="A reliable search-ready representation could not be prepared for this Persian query.",
            disclaimer=(
                f"{display_name} metadata and abstracts are provided as research information; "
                "this response does not provide diagnosis or treatment advice."
            ),
        )

    try:
        records = await provider.search(medical_query.search_query, request.limit)
    except Exception as exc:
        logger.warning("medical evidence provider failed provider=%s error_type=%s", source, type(exc).__name__)
        raise MedicalApplicationError(502, f"{display_name} provider failed.") from exc

    found = bool(records)
    return SearchResponse(
        query=request.query,
        language=medical_query.language,
        answer=(
            f"Retrieved {display_name} records are listed below."
            if found else f"No {display_name} records matched the query."
        ),
        abstained=not found,
        uncertainty=None if found else f"{display_name} returned no matching records.",
        evidence=records,
        provider_status={source: "ok" if found else "no_results"},
        disclaimer=(
            f"{display_name} metadata and abstracts are provided as research information; "
            "this response does not provide diagnosis or treatment advice."
        ),
    )


async def medical_ask(
    request: SearchRequest,
    service: MedicalSearchService,
    answer_provider: AnswerProvider,
    claim_verifier: ClaimVerifier,
) -> AskResponse:
    logger.info("medical ask started")
    search_result = await service.search(request)
    statuses = search_result.provider_status
    if not statuses or all(status == "unavailable" for status in statuses.values()):
        logger.warning("medical ask failed because all evidence providers are unavailable")
        raise MedicalApplicationError(503, "Evidence search providers are unavailable.")
    if search_result.abstained or not search_result.evidence:
        logger.info("medical ask abstained stage=evidence")
        return abstention(request.language, len(search_result.evidence))

    try:
        result: AnswerResult = await answer_provider.answer(
            request.query, search_result.evidence, request.language
        )
    except Exception as exc:
        logger.warning("medical answer provider failed error_type=%s", type(exc).__name__)
        raise MedicalApplicationError(502, "Answer provider failed.") from exc

    logger.info(
        "medical answer generated abstained=%s claim_count=%d",
        result.abstained,
        len(result.claims),
    )

    try:
        citations = map_citations(result, search_result.evidence)
    except InvalidCitationError as exc:
        logger.warning("medical answer citation validation failed")
        raise MedicalApplicationError(502, "Answer provider returned an invalid citation.") from exc

    if result.abstained or not result.answer.strip() or not result.claims or not citations:
        logger.info("medical ask abstained stage=answer_validation")
        return abstention(request.language, len(search_result.evidence))
    if not answer_is_covered_by_claims(result.answer, [claim.text for claim in result.claims]):
        logger.info("medical ask abstained stage=claim_coverage")
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
        except Exception as exc:
            logger.warning("medical claim verifier failed error_type=%s", type(exc).__name__)
            verifications.extend(uncertain_verifications(cited_claims))

    if any(item.status != "supported" for item in verifications):
        logger.info(
            "medical claim verification completed supported=%d unsupported=%d uncertain=%d",
            sum(item.status == "supported" for item in verifications),
            sum(item.status == "unsupported" for item in verifications),
            sum(item.status == "uncertain" for item in verifications),
        )
        logger.info("medical ask abstained stage=claim_verification")
        return abstention(
            request.language,
            len(search_result.evidence),
            citations=citations,
            claim_verifications=verifications,
        )
    logger.info(
        "medical claim verification completed supported=%d unsupported=0 uncertain=0",
        len(verifications),
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
