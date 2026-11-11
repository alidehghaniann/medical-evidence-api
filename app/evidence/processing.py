import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from app.models.evidence import EvidenceAssessment, EvidenceRecord


def normalize_record(record: EvidenceRecord) -> EvidenceRecord:
    record.title = " ".join(record.title.split())
    record.abstract = " ".join(record.abstract.split()) if record.abstract else None
    record.authors = list(dict.fromkeys(" ".join(a.split()) for a in record.authors if a.strip()))
    record.doi = record.doi.removeprefix("https://doi.org/").strip() if record.doi else None
    record.evidence_type = record.evidence_type or classify_evidence(record.title, record.abstract or "")
    record.assessment = assess_evidence(record)
    return record


def assess_evidence(record: EvidenceRecord) -> EvidenceAssessment:
    """Assign explicit baseline metadata without implying clinical appraisal."""
    current_year = datetime.now(timezone.utc).year
    authority = {"pubmed": 0.55, "europepmc": 0.55, "perplexity": 0.35}.get(record.source.lower())
    recency = max(0.0, 1 - (current_year - record.publication_year) / 20) if record.publication_year else None
    strength = {
        "guideline": "high",
        "systematic_review": "high",
        "randomized_trial": "moderate",
        "observational_study": "low",
    }.get(record.evidence_type, "unknown")
    known_type = record.evidence_type is not None
    reasons = []
    if known_type:
        reasons.append(f"Evidence type classified as {record.evidence_type} by title/abstract pattern.")
    else:
        reasons.append("Evidence type could not be classified from the available title and abstract.")
    reasons.append("Source authority is a fixed source-level baseline." if authority is not None else "Source authority is unknown.")
    reasons.append("Recency is estimated from publication year." if recency is not None else "Publication year is unavailable; recency is unknown.")
    return EvidenceAssessment(
        evidence_type=record.evidence_type,
        evidence_strength=strength,
        source_authority=authority,
        recency=recency,
        guideline_priority=record.evidence_type == "guideline" if known_type else None,
        assessment_confidence=0.5 if known_type else 0.2,
        assessment_reasons=reasons,
    )


def classify_evidence(title: str, abstract: str) -> str | None:
    text = f"{title} {abstract}".lower()
    for label, pattern in (
        ("systematic_review", r"systematic review|meta-analysis|meta analysis"),
        ("randomized_trial", r"randomi[sz]ed (controlled )?trial"),
        ("guideline", r"guideline|practice recommendation|consensus statement"),
        ("observational_study", r"cohort study|case-control|cross-sectional"),
    ):
        if re.search(pattern, text):
            return label
    return None


def deduplicate(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    unique: list[EvidenceRecord] = []
    seen_ids: set[str] = set()
    seen_dois: set[str] = set()
    for raw in records:
        item = normalize_record(raw)
        keys = [key.lower() for key in (item.source_id, item.doi) if key]
        title_key = re.sub(r"\W+", " ", item.title.lower()).strip()
        if any(key in seen_ids or key in seen_dois for key in keys):
            continue
        if any(SequenceMatcher(None, title_key, re.sub(r"\W+", " ", old.title.lower()).strip()).ratio() > 0.94 for old in unique):
            continue
        unique.append(item)
        if item.source_id:
            seen_ids.add(item.source_id.lower())
        if item.doi:
            seen_dois.add(item.doi.lower())
    return unique


def rank_records(records: list[EvidenceRecord], query: str) -> list[EvidenceRecord]:
    terms = set(re.findall(r"[\w]+", query.lower()))
    current_year = datetime.now(timezone.utc).year
    for item in records:
        text_terms = set(re.findall(r"[\w]+", f"{item.title} {item.abstract or ''}".lower()))
        item.relevance_score = len(terms & text_terms) / max(1, len(terms))
        assessment = (
            item.assessment
            if item.assessment.assessment_reasons and item.assessment.evidence_type == item.evidence_type
            else assess_evidence(item)
        )
        item.assessment = assessment
        item.authority_score = assessment.source_authority if assessment.source_authority is not None else 0.25
        item.recency_score = assessment.recency if assessment.recency is not None else 0.2
        evidence_boost = {"systematic_review": 0.12, "guideline": 0.12, "randomized_trial": 0.08}.get(assessment.evidence_type, 0.0)
        item.rank_score = min(1.0, 0.55 * item.relevance_score + 0.25 * item.authority_score + 0.20 * item.recency_score + evidence_boost)
    return sorted(records, key=lambda item: item.rank_score, reverse=True)


def validate_citations(records: list[EvidenceRecord]) -> bool:
    return all(item.url and item.source for item in records)
