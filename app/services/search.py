import asyncio
import logging
import re

from app.core.config import Settings
from app.evidence.processing import deduplicate, rank_records, validate_citations
from app.models.evidence import SearchRequest, SearchResponse
from app.models.medical_query import detect_query_language
from app.providers.base import EvidenceProvider
from app.services.query_understanding import QueryUnderstander, understand_medical_query
from app.services.source_router import SourceRouter

DISCLAIMER = "This service provides research information, not diagnosis or treatment. It is not a validated clinical decision support system. Consult a qualified clinician."
logger = logging.getLogger(__name__)

class MedicalSearchService:
    def __init__(
        self,
        providers: list[EvidenceProvider],
        settings: Settings,
        query_understander: QueryUnderstander | None = None,
        source_router: SourceRouter | None = None,
    ):
        self.providers = providers
        self.settings = settings
        self.query_understander = query_understander
        self.source_router = source_router or SourceRouter()

    async def search(self, request: SearchRequest) -> SearchResponse:
        limit = min(request.limit, self.settings.max_search_results)
        language = detect_query_language(request.query)
        logger.info("medical evidence search started provider_count=%d", len(self.providers))
        query_error = False
        try:
            medical_query = await understand_medical_query(request.query, self.query_understander)
        except Exception as exc:
            logger.warning("medical query understanding failed error_type=%s", type(exc).__name__)
            query_error = True
            medical_query = None
        route = self.source_router.route(self.providers) if medical_query is not None else None
        searchable = medical_query is not None and medical_query.search_query is not None
        if searchable:
            provider_query = medical_query.search_query
            ranking_query = provider_query
            for provider in route.providers:
                logger.debug("medical evidence provider called provider=%s", provider.name)
            calls = [provider.search(provider_query, limit) for provider in route.providers]
            results = await asyncio.gather(*calls, return_exceptions=True)
        else:
            results = [[] for _ in self.providers]
        statuses: dict[str, str] = {}
        if language == "fa":
            statuses["query_understanding"] = "unavailable" if query_error else (
                "ok" if searchable else "not_configured"
            )
        records = []
        routed_providers = route.providers if route else tuple(self.providers)
        for provider, result in zip(routed_providers, results):
            if not searchable:
                statuses[provider.name] = "skipped_query_understanding"
                continue
            if isinstance(result, Exception):
                statuses[provider.name] = "unavailable"
                logger.warning(
                    "medical evidence provider failed provider=%s error_type=%s",
                    provider.name,
                    type(result).__name__,
                )
            else:
                statuses[provider.name] = "ok" if result else "no_results"
                records.extend(result)
        ranking_query = ranking_query if searchable else ""
        normalized_candidates = deduplicate(records)
        ranked = rank_records(normalized_candidates, ranking_query)[:limit]
        topic_match = searchable and has_topic_match(ranking_query, ranked)
        if ranked and not topic_match:
            # Providers can return broad fallback hits for unusual terms. Do
            # not present those as evidence when none match the query topic.
            ranked = []
        valid = validate_citations(ranked)
        sufficient = (
            bool(ranked)
            and valid
            and topic_match
            and ranked[0].rank_score >= self.settings.minimum_evidence_score
        )
        if sufficient:
            if language == "fa":
                answer = "منابع بازیابی‌شده در فهرست شواهد آمده‌اند. این جست‌وجو توصیه یا نتیجه‌گیری بالینی ارائه نمی‌کند."
            else:
                answer = "Retrieved sources are listed below. This MVP does not generate a clinical recommendation; review the cited evidence directly."
            uncertainty = "Evidence ranking is heuristic and has not been clinically validated."
        else:
            answer = "برای این پرسش عبارت جست‌وجوی انگلیسی قابل اتکایی آماده نشد؛ شواهدی جست‌وجو نشد." if language == "fa" and not searchable else (
                "شواهد کافی و قابل استناد بازیابی نشد؛ پاسخ بالینی ارائه نمی‌شود." if language == "fa" else "Insufficient attributable evidence was retrieved; no clinical answer is provided."
            )
            uncertainty = "No evidence passed the minimum evidence and citation checks."
        if sufficient:
            abstention_reason = None
        elif not searchable and language == "fa":
            abstention_reason = "A reliable search-ready representation could not be prepared for this Persian query."
        elif not ranked:
            abstention_reason = "No matching evidence was found."
        elif not valid:
            abstention_reason = "Retrieved records did not pass citation validation."
        else:
            abstention_reason = "Retrieved evidence did not meet the minimum quality threshold."
        response = SearchResponse(query=request.query, language=language, answer=answer,
            abstained=not sufficient, uncertainty=uncertainty, evidence=ranked,
            abstention_reason=abstention_reason, provider_status=statuses, disclaimer=DISCLAIMER)
        if response.abstained:
            logger.info("medical evidence search abstained")
        logger.info(
            "medical evidence search completed evidence_count=%d abstained=%s",
            response.evidence_count,
            response.abstained,
        )
        return response


def has_topic_match(query: str, records) -> bool:
    """Require at least one non-generic query term in a title or abstract before calling evidence sufficient."""
    all_terms = {
        term.casefold()
        for term in re.findall(r"[^\W\d_]+", query, flags=re.UNICODE)
        if len(term) >= 4
    }
    generic = {
        "about", "clinical", "condition", "disease", "evidence", "first", "for", "from",
        "in", "infection", "latest", "medical", "of", "on", "recent", "research",
        "study", "studies", "the", "treatment", "therapy", "what", "which", "with",
        "disorder", "syndrome", "line", "use", "using",
    }
    terms = all_terms - generic
    if not terms:
        # Preserve relevance gating for short, broad API queries such as
        # "treatment evidence"; the rank threshold remains the second check.
        terms = all_terms
    if not terms:
        return False
    for record in records:
        evidence_terms = set(
            re.findall(r"[^\W\d_]+", f"{record.title} {record.abstract or ''}".casefold(), flags=re.UNICODE)
        )
        if terms & evidence_terms:
            return True
    return False
