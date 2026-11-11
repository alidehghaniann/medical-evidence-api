import re
from typing import Protocol

from app.models.medical_query import (
    MedicalIntent,
    MedicalQuery,
    SearchQueryRepresentation,
    detect_query_language,
    normalize_query_text,
)


class QueryUnderstander(Protocol):
    async def understand_query(
        self, query: str, normalized_query: str
    ) -> SearchQueryRepresentation:
        """Prepare a Persian query for biomedical literature search."""


_INTENT_PATTERNS: tuple[tuple[MedicalIntent, re.Pattern[str]], ...] = (
    ("clinical_trial", re.compile(r"\b(trial|randomi[sz]ed)\b", re.I)),
    ("guideline", re.compile(r"\b(guideline|recommendation)\b", re.I)),
    ("adverse_effects", re.compile(r"\b(adverse effect|side effect|toxicity)\b", re.I)),
    ("drug_interaction", re.compile(r"\b(interaction|interact)\b", re.I)),
    ("safety", re.compile(r"\b(safe|safety|risk)\b", re.I)),
    ("efficacy", re.compile(r"\b(efficacy|effective|effectiveness)\b", re.I)),
    ("prevention", re.compile(r"\b(prevent|prevention|prophylaxis)\b", re.I)),
    ("prognosis", re.compile(r"\b(prognosis|outcome|survival)\b", re.I)),
    ("screening", re.compile(r"\b(screen|screening)\b", re.I)),
    ("diagnosis", re.compile(r"\b(diagnos|evaluation|test)\w*\b", re.I)),
    ("treatment", re.compile(r"\b(treat|therapy|management)\w*\b", re.I)),
    ("epidemiology", re.compile(r"\b(prevalence|incidence|epidemiology)\b", re.I)),
    ("mechanism", re.compile(r"\b(mechanism|pathophysiology)\b", re.I)),
    ("evidence", re.compile(r"\b(evidence|research|studies|literature)\b", re.I)),
)


def classify_intent(query: str) -> MedicalIntent:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return "general"


async def understand_medical_query(
    query: str,
    query_understander: QueryUnderstander | None,
) -> MedicalQuery:
    normalized = normalize_query_text(query)
    language = detect_query_language(query)
    if language == "en":
        terms = re.findall(r"[A-Za-z][A-Za-z0-9-]*", normalized)
        return MedicalQuery(
            original_query=query,
            language=language,
            normalized_query=normalized,
            search_query=query,
            medical_terms=terms,
            intent=classify_intent(normalized),
        )

    if query_understander is None:
        return MedicalQuery(
            original_query=query,
            language=language,
            normalized_query=normalized,
            search_query=None,
            medical_terms=[],
            intent="general",
        )

    representation = await query_understander.understand_query(query, normalized)
    if not isinstance(representation, SearchQueryRepresentation):
        representation = SearchQueryRepresentation.model_validate(representation)
    return MedicalQuery(
        original_query=query,
        language=language,
        normalized_query=normalized,
        search_query=representation.search_query,
        medical_terms=representation.medical_terms,
        intent=representation.intent,
    )
