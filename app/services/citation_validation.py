import re

from app.models.answer import AnswerCitation, AnswerResult
from app.models.evidence import EvidenceRecord


class InvalidCitationError(ValueError):
    """The answer provider referenced evidence outside this search result."""


def map_citations(result: AnswerResult, evidence: list[EvidenceRecord]) -> list[AnswerCitation]:
    evidence_by_id = {f"evidence-{index}": item for index, item in enumerate(evidence, start=1)}
    known_urls = {str(item.url).rstrip("/") for item in evidence}
    citations = []
    seen = set()
    citation_ids = [citation.evidence_id for citation in result.citations]
    for claim in result.claims:
        for evidence_id in claim.citation_ids:
            if evidence_id not in evidence_by_id:
                raise InvalidCitationError(f"Unknown evidence ID: {evidence_id}")
            if evidence_id not in citation_ids:
                citation_ids.append(evidence_id)
    for evidence_id in citation_ids:
        record = evidence_by_id.get(evidence_id)
        if record is None:
            raise InvalidCitationError(f"Unknown evidence ID: {evidence_id}")
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        citations.append(AnswerCitation(
            evidence_id=evidence_id,
            title=record.title,
            source=record.source,
            source_id=record.source_id,
            url=record.url,
        ))
    for referenced_id in re.findall(r"\[?(evidence-\d+)\]?", result.answer):
        if referenced_id not in evidence_by_id or referenced_id not in seen:
            raise InvalidCitationError(f"Unvalidated citation in answer: {referenced_id}")
    for referenced_url in re.findall(r"https?://[^\s\])}>\"']+", result.answer):
        if referenced_url.rstrip(".,;:").rstrip("/") not in known_urls:
            raise InvalidCitationError("Unvalidated source URL in answer")
    return citations
