import logging
import json
import re

from app.models.answer import AnswerCitation, AnswerResult, AskDiagnostics, AskResponse, ClaimToVerify, ClaimVerification
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


def _provider_name(provider) -> str:
    return getattr(provider, "name", type(provider).__name__.replace("Provider", "").casefold())


def _ask_diagnostics(search_result, answer_provider, claim_verifier) -> AskDiagnostics:
    search = search_result.pipeline_diagnostics
    providers = {
        "answer": _provider_name(answer_provider),
        "claim_verifier": _provider_name(claim_verifier),
    }
    if search is not None:
        providers["query_understanding"] = search.query_understanding_provider
        providers.update({name: status for name, status in search.provider_status.items() if name != "query_understanding"})
        counts = {
            "medical_terms": search.medical_terms_count,
            "retrieved_records": search.retrieved_count,
            "normalized_records": search.normalized_count,
            "deduplicated_records": search.deduplicated_count,
            "ranked_records": search.ranked_count,
            "assessed_records": search.evidence_assessed_count,
            "returned_evidence": search_result.evidence_count,
        }
        query_status = search.query_understanding_status
        source_statuses = [
            status for name, status in search.provider_status.items() if name != "query_understanding"
        ]
        if source_statuses and all(status == "skipped_query_understanding" for status in source_statuses):
            retrieval_status = "skipped"
        elif not source_statuses or all(status == "unavailable" for status in source_statuses):
            retrieval_status = "failed"
        else:
            retrieval_status = "completed"
    else:
        providers["query_understanding"] = "unknown"
        counts = {"returned_evidence": search_result.evidence_count}
        query_status = "unknown"
        retrieval_status = "unknown"
    stages = {
        "query_understanding": query_status,
        "retrieval": retrieval_status,
        "normalization": "completed",
        "ranking": "completed",
        "evidence_assessment": "completed",
        "answer_provider_selection": "selected",
        "answer_provider_execution": "not_started",
        "answer_parsing": "not_started",
        "claim_extraction": "not_started",
        "citation_mapping": "not_started",
        "claim_verification": "not_started",
        "abstention": "pending",
    }
    return AskDiagnostics(
        query_language=search_result.language,
        query_intent=search.intent if search is not None else None,
        stages=stages,
        providers=providers,
        counts=counts,
    )


def _log_ask_diagnostics(diagnostics: AskDiagnostics, *, error_type: str | None = None) -> None:
    log = logger.warning if error_type else logger.info
    log(
        "medical ask diagnostic reason_code=%s providers=%s stages=%s counts=%s error_type=%s",
        diagnostics.reason_code or "none",
        diagnostics.providers,
        diagnostics.stages,
        diagnostics.counts,
        error_type or "none",
    )


class MedicalApplicationError(Exception):
    """Controlled application failure shared by the REST and MCP interfaces."""

    def __init__(self, status_code: int, detail: str, reason_code: str | None = None):
        self.status_code = status_code
        self.detail = detail
        self.reason_code = reason_code
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
    diagnostics = _ask_diagnostics(search_result, answer_provider, claim_verifier)
    statuses = search_result.provider_status
    if not statuses or all(status == "unavailable" for status in statuses.values()):
        logger.warning("medical ask failed because all evidence providers are unavailable")
        raise MedicalApplicationError(503, "Evidence search providers are unavailable.", "retrieval_failed")
    if search_result.abstained or not search_result.evidence:
        query_status = diagnostics.stages["query_understanding"]
        if query_status in {"unavailable", "not_configured"}:
            reason = "query_understanding_failed"
        else:
            reason = "no_evidence" if not search_result.evidence else "insufficient_evidence"
        diagnostics.reason_code = reason
        diagnostics.stages["abstention"] = "abstained"
        _log_ask_diagnostics(diagnostics)
        return abstention(request.language, len(search_result.evidence), diagnostics=diagnostics)

    diagnostics.stages["answer_provider_execution"] = "started"
    try:
        result: AnswerResult = await answer_provider.answer(
            request.query, search_result.evidence, request.language
        )
    except Exception as exc:
        diagnostics.stages["answer_provider_execution"] = "failed"
        diagnostics.reason_code = "answer_parse_failed" if isinstance(exc, (ValueError, json.JSONDecodeError)) else "answer_provider_failed"
        diagnostics.stages["abstention"] = "error"
        _log_ask_diagnostics(diagnostics, error_type=type(exc).__name__)
        raise MedicalApplicationError(502, "Answer provider failed.", diagnostics.reason_code) from exc

    diagnostics.stages["answer_provider_execution"] = "succeeded"
    diagnostics.stages["answer_parsing"] = "succeeded"
    diagnostics.stages["claim_extraction"] = "completed"
    diagnostics.counts["claims_extracted"] = len(result.claims)
    logger.info(
        "medical answer generated abstained=%s claim_count=%d",
        result.abstained,
        len(result.claims),
    )

    try:
        citations = map_citations(result, search_result.evidence)
    except InvalidCitationError as exc:
        diagnostics.stages["citation_mapping"] = "failed"
        diagnostics.reason_code = "citation_mapping_failed"
        diagnostics.stages["abstention"] = "error"
        _log_ask_diagnostics(diagnostics, error_type=type(exc).__name__)
        raise MedicalApplicationError(502, "Answer provider returned an invalid citation.", "citation_mapping_failed") from exc

    diagnostics.stages["citation_mapping"] = "succeeded"
    diagnostics.counts["citations_mapped"] = len(citations)
    if result.abstained:
        diagnostics.reason_code = "insufficient_evidence"
        diagnostics.stages["abstention"] = "abstained"
        _log_ask_diagnostics(diagnostics)
        return abstention(request.language, len(search_result.evidence), diagnostics=diagnostics)
    if not result.answer.strip():
        diagnostics.reason_code = "answer_parse_failed"
        diagnostics.stages["abstention"] = "abstained"
        _log_ask_diagnostics(diagnostics)
        return abstention(request.language, len(search_result.evidence), diagnostics=diagnostics)
    if not result.claims:
        diagnostics.reason_code = "no_claims"
        diagnostics.stages["abstention"] = "abstained"
        _log_ask_diagnostics(diagnostics)
        return abstention(request.language, len(search_result.evidence), diagnostics=diagnostics)
    if not citations:
        diagnostics.reason_code = "citation_mapping_failed"
        diagnostics.stages["abstention"] = "abstained"
        _log_ask_diagnostics(diagnostics)
        return abstention(request.language, len(search_result.evidence), diagnostics=diagnostics)
    answer_sentence_count, covered_answer_sentence_count = answer_claim_coverage(
        result.answer, [claim.text for claim in result.claims]
    )
    diagnostics.counts["answer_sentences"] = answer_sentence_count
    diagnostics.counts["covered_answer_sentences"] = covered_answer_sentence_count
    if answer_sentence_count == 0 or answer_sentence_count != covered_answer_sentence_count:
        diagnostics.reason_code = "claim_coverage_failed"
        diagnostics.stages["claim_extraction"] = "incomplete_coverage"
        diagnostics.stages["abstention"] = "abstained"
        _log_ask_diagnostics(diagnostics)
        return abstention(request.language, len(search_result.evidence), diagnostics=diagnostics)

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
        diagnostics.stages["claim_verification"] = "started"
        try:
            verified = await claim_verifier.verify(cited_claims)
            verified = validate_verifications(cited_claims, verified)
            diagnostics.stages["claim_verification"] = "completed"
            verifications.extend([
                item.model_copy(update={"status": "uncertain"})
                if item.status == "supported" and item.confidence < 0.7 else item
                for item in verified
            ])
        except Exception as exc:
            logger.warning("medical claim verifier failed error_type=%s", type(exc).__name__)
            diagnostics.stages["claim_verification"] = "failed"
            diagnostics.reason_code = "claim_verification_failed"
            verifications.extend(uncertain_verifications(cited_claims))

    diagnostics.counts["supported_claims"] = sum(item.status == "supported" for item in verifications)
    diagnostics.counts["unsupported_claims"] = sum(item.status == "unsupported" for item in verifications)
    diagnostics.counts["uncertain_claims"] = sum(item.status == "uncertain" for item in verifications)
    if any(item.status != "supported" for item in verifications):
        if diagnostics.reason_code is None:
            diagnostics.reason_code = "unsupported_claims" if diagnostics.counts["unsupported_claims"] else "uncertain_claims"
        diagnostics.stages["abstention"] = "abstained"
        logger.info(
            "medical claim verification completed supported=%d unsupported=%d uncertain=%d",
            sum(item.status == "supported" for item in verifications),
            sum(item.status == "unsupported" for item in verifications),
            sum(item.status == "uncertain" for item in verifications),
        )
        logger.info("medical ask abstained stage=claim_verification")
        _log_ask_diagnostics(diagnostics)
        return abstention(
            request.language,
            len(search_result.evidence),
            citations=citations,
            claim_verifications=verifications,
            diagnostics=diagnostics,
        )
    logger.info(
        "medical claim verification completed supported=%d unsupported=0 uncertain=0",
        len(verifications),
    )
    confidence = min(result.confidence, *(item.confidence for item in verifications))
    diagnostics.stages["abstention"] = "not_abstained"
    diagnostics.reason_code = None
    _log_ask_diagnostics(diagnostics)
    return AskResponse(
        answer=render_verified_claims(claim_requests),
        language=request.language,
        citations=citations,
        confidence=confidence,
        abstained=False,
        evidence_count=len(search_result.evidence),
        claim_verifications=verifications,
        diagnostics=diagnostics,
    )


def answer_is_covered_by_claims(answer: str, claims: list[str]) -> bool:
    sentence_count, covered_count = answer_claim_coverage(answer, claims)
    return bool(sentence_count) and sentence_count == covered_count


def answer_claim_coverage(answer: str, claims: list[str]) -> tuple[int, int]:
    """Count answer sentences and the ones whose text a claim covers.

    Both sides use the same normalization and sentence segmentation, so a claim
    that repeats several answer sentences covers each of them. Coverage remains
    exact text equality after normalization: it never becomes a similarity test.
    """
    sentences = _normalized_sentences(answer)
    claim_sentences = {
        normalize_answer_sentence(sentence)
        for claim in claims
        for sentence in _normalized_sentences(claim)
    }
    covered = sum(normalize_answer_sentence(sentence) in claim_sentences for sentence in sentences)
    return len(sentences), covered


_SENTINEL = "\x00"
# Leading separators come from citations stripped at the start of a sentence. A
# single leading dot is kept: it is either a disfluent "..." or a decimal like ".5".
_LEADING_SEPARATORS = re.compile(r"^[ ,;:\u060c\u061b]*\.{0,2}")
# A boundary is sentence punctuation followed by whitespace or the end of the text,
# or sentence punctuation followed directly by a capital letter. A bare split on the
# punctuation would also cut "74.7%", "1.14-1.76" and "1.42" into fragments, so a
# claim could never match its own sentence.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?؟])\s+|\n+|(?<=[.!?؟])$|(?<=[.!?])(?=[A-Z][a-z])"
)


def normalize_answer_sentence(text: str) -> str:
    """Reduce a sentence to its comparable text without changing its wording."""
    # Citation tokens carry no claim text, so they are removed before comparison.
    # The parenthesised/bracketed forms are dropped as a whole, which also removes
    # any lead-in such as "(see ...)" or "(source: ...)". Only the token itself is
    # replaced; the surrounding prose is left exactly as written.
    text = re.sub(r"[\[(][^\])]*evidence-\d+[^\])]*[\])]", _SENTINEL, text, flags=re.IGNORECASE)
    text = re.sub(r"evidence-\d+", _SENTINEL, text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    text = _LEADING_SEPARATORS.sub("", text).lstrip(" ,;:\u060c\u061b")
    # A separator left where a citation stood is dropped with it.
    text = re.sub(_SENTINEL + r"([\s,;:\u060c\u061b.]*)" + _SENTINEL, _SENTINEL, text)
    text = re.sub(_SENTINEL + r"[\s,;:\u060c\u061b.]+$", "", text)
    text = re.sub(_SENTINEL + r"(?=[.,;:\u060c\u061b])", "", text)
    text = text.replace(_SENTINEL, " ")
    text = re.sub(r"[\s,;:\u060c\u061b.]+$", "", text)
    return " ".join(text.casefold().split())


# A citation token with a sentence boundary in front of it is moved behind that
# boundary, so what follows the token starts the next sentence instead of reading
# as a full stop inside one claim. Only spaces are moved: a token that is followed
# directly by a capital is left alone, because no boundary can be told apart there.
_CITATION_TOKEN_AT_BOUNDARY = re.compile(
    r"(?<=[.!?\u061f])[ ]*(\[?evidence-\d\]?)[ ]*(?=[A-Z\u0600-\u06ff])"
)


def _split_citation_runs(text: str) -> str:
    """Move a citation token that starts a sentence back onto the sentence it cites.

    "... advantage. [evidence-3] The next claim" and "... advantage. [evidence-3]The
    next claim" are two sentences, but a token sitting on the boundary makes the
    splitter treat the boundary as a full stop inside one claim. Moving the token
    behind the boundary leaves what follows it to start the next sentence.
    """
    return _CITATION_TOKEN_AT_BOUNDARY.sub(" ", text)


def _normalized_sentences(text: str) -> list[str]:
    return [
        normalized
        for part in _SENTENCE_BOUNDARY.split(_split_citation_runs(text))
        if (normalized := normalize_answer_sentence(part))
    ]


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
    diagnostics: AskDiagnostics | None = None,
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
        diagnostics=diagnostics or AskDiagnostics(
            reason_code="no_evidence",
            stages={"abstention": "abstained"},
            counts={"returned_evidence": evidence_count},
        ),
    )
