import asyncio

from app.core.config import Settings
from app.evidence.processing import deduplicate, rank_records, validate_citations
from app.models.evidence import SearchRequest, SearchResponse
from app.providers.base import EvidenceProvider

DISCLAIMER = "This service provides research information, not diagnosis or treatment. It is not a validated clinical decision support system. Consult a qualified clinician."


class MedicalSearchService:
    def __init__(self, providers: list[EvidenceProvider], settings: Settings):
        self.providers = providers
        self.settings = settings

    async def search(self, request: SearchRequest) -> SearchResponse:
        limit = min(request.limit, self.settings.max_search_results)
        calls = [provider.search(request.query, limit) for provider in self.providers]
        results = await asyncio.gather(*calls, return_exceptions=True)
        statuses: dict[str, str] = {}
        records = []
        for provider, result in zip(self.providers, results):
            if isinstance(result, Exception):
                statuses[provider.name] = "unavailable"
            else:
                statuses[provider.name] = "ok" if result else "no_results"
                records.extend(result)
        ranked = rank_records(deduplicate(records), request.query)
        valid = validate_citations(ranked)
        sufficient = bool(ranked) and valid and ranked[0].rank_score >= self.settings.minimum_evidence_score
        if sufficient:
            if request.language == "fa":
                answer = "منابع بازیابی‌شده در فهرست شواهد آمده‌اند. برای جلوگیری از ساختن توصیه پزشکی، این نسخه خلاصه بالینی تولید نمی‌کند."
            else:
                answer = "Retrieved sources are listed below. This MVP does not generate a clinical recommendation; review the cited evidence directly."
            uncertainty = "Evidence ranking is heuristic and has not been clinically validated."
        else:
            answer = "شواهد کافی و قابل استناد بازیابی نشد؛ پاسخ بالینی ارائه نمی‌شود." if request.language == "fa" else "Insufficient attributable evidence was retrieved; no clinical answer is provided."
            uncertainty = "No evidence passed the minimum evidence and citation checks."
        return SearchResponse(query=request.query, language=request.language, answer=answer,
            abstained=not sufficient, uncertainty=uncertainty, evidence=ranked,
            provider_status=statuses, disclaimer=DISCLAIMER)
