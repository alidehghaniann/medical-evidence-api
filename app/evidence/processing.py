import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from app.models.evidence import EvidenceRecord


def normalize_record(record: EvidenceRecord) -> EvidenceRecord:
    record.title = " ".join(record.title.split())
    record.abstract = " ".join(record.abstract.split()) if record.abstract else None
    record.authors = list(dict.fromkeys(" ".join(a.split()) for a in record.authors if a.strip()))
    record.doi = record.doi.removeprefix("https://doi.org/").strip() if record.doi else None
    record.evidence_type = record.evidence_type or classify_evidence(record.title, record.abstract or "")
    return record


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
        item.authority_score = {
            "pubmed": 0.55,
            "europepmc": 0.55,
            "perplexity": 0.35,
        }.get(item.source.lower(), 0.25)
        item.recency_score = max(0.0, 1 - (current_year - item.publication_year) / 20) if item.publication_year else 0.2
        evidence_boost = {"systematic_review": 0.12, "guideline": 0.12, "randomized_trial": 0.08}.get(item.evidence_type, 0.0)
        item.rank_score = min(1.0, 0.55 * item.relevance_score + 0.25 * item.authority_score + 0.20 * item.recency_score + evidence_boost)
    return sorted(records, key=lambda item: item.rank_score, reverse=True)


def validate_citations(records: list[EvidenceRecord]) -> bool:
    return all(item.url and item.source for item in records)
