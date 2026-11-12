"""Deterministic reduction of an English medical question to index search terms."""

import re

# Question scaffolding and hedging that carry no biomedical meaning for an index.
_QUESTION_TERMS = frozenset({
    "about", "an", "and", "any", "are", "can", "could", "did", "do", "does",
    "explain", "for", "from", "give", "has", "have", "how", "is", "latest", "me", "of",
    "on", "please", "recent", "should", "show", "tell", "than", "that",
    "the", "there", "these", "this", "to", "was", "were", "what", "when", "which", "who",
    "why", "will", "with", "would", "you",
})

# Short but meaningful biomedical abbreviations the length filter must not drop.
_SHORT_KEPT_TERMS = frozenset({"ace", "adhd", "aki", "b12", "bmi", "cbd", "ckd", "copd",
                               "crp", "dvt", "gerd", "hiv", "hp", "ibd", "ihd", "iv", "mrsa",
                               "nsaid", "pcos", "pe", "ppi", "tb", "tsh", "uti"})

_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*[A-Za-z0-9]|[A-Za-z0-9]")


class SearchTermError(ValueError):
    """The question could not be reduced to safe, index-ready search terms."""


def build_search_query(question: str) -> tuple[str, list[str]]:
    """Return a concise search query and the medical terms it was built from.

    Fails closed: a question whose medical content is entirely scaffolding or
    single letters raises instead of becoming a broad, low-precision search.
    """
    terms = _terms(question)
    if not terms:
        raise SearchTermError("No searchable medical terms were found in the question")
    return " ".join(terms), terms


def _terms(question: str) -> list[str]:
    terms: list[str] = []
    for raw in _TERM.findall(question):
        term = raw.strip("-'")
        if not term:
            continue
        key = term.casefold()
        if key in _QUESTION_TERMS:
            continue
        if len(key) < 3 and key not in _SHORT_KEPT_TERMS:
            continue
        if key.isdigit():
            continue
        if key not in {existing.casefold() for existing in terms}:
            terms.append(term)
    return terms
