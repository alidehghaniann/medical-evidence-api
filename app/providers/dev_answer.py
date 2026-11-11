"""Deterministic extractive answer provider for development and tests only."""

import re

from app.models.answer import AnswerClaim, AnswerResult, ProviderCitation
from app.models.evidence import EvidenceRecord
from app.providers.answer_base import AnswerProvider

_WORD = re.compile(r"[\w]+", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z0-9])|\r?\n+")
_PERSIAN_SCRIPT = re.compile(r"[\u0600-\u06ff]")
_NON_SPECIFIC_QUERY_TERMS = frozenset({
    "a", "about", "adverse", "and", "are", "chronic", "clinical", "condition", "conditions",
    "disease", "diseases", "does", "drug", "drugs", "effect", "effectiveness",
    "evidence", "explain", "for", "from", "health", "how", "in", "is", "latest",
    "line", "medical", "medicine", "medicines", "new", "of", "on", "paper",
    "option", "options", "papers", "patient", "patients", "recent", "research",
    "risk", "side", "study", "studies", "symptom", "symptoms", "syndrome", "the",
    "therapy", "treatment", "what", "which", "with", "infection", "infections",
})


class DevAnswerProvider(AnswerProvider):
    """Return one exact abstract sentence; never synthesize or translate medical content.

    This provider is DEVELOPMENT/TEST ONLY. It requires query-term overlap and an
    abstract sentence in the requested language. Missing or unsuitable evidence
    produces an explicit abstention.
    """

    async def close(self) -> None:
        """Match the lifecycle used by network-backed providers."""

    async def answer(
        self,
        query: str,
        evidence: list[EvidenceRecord],
        language: str,
    ) -> AnswerResult:
        query_terms = _specific_terms(query)
        if not evidence or not query_terms:
            return _abstained_result()

        for index, record in enumerate(evidence, start=1):
            if not record.abstract:
                continue
            evidence_id = f"evidence-{index}"
            for sentence in _SENTENCE_BOUNDARY.split(record.abstract.strip()):
                sentence = sentence.strip()
                overlap = query_terms.intersection(_tokens(sentence))
                required_overlap = 1 if len(query_terms) == 1 else 2
                if len(sentence) < 20 or len(overlap) < required_overlap:
                    continue
                if language == "fa" and not _PERSIAN_SCRIPT.search(sentence):
                    continue
                if language == "en" and _PERSIAN_SCRIPT.search(sentence):
                    continue
                # The application validates one claim per sentence. Skip
                # abbreviations/initials that its sentence splitter would split.
                if re.search(r"[.!?]", sentence.rstrip(".!?")):
                    continue
                claim = AnswerClaim(text=sentence, citation_ids=[evidence_id])
                return AnswerResult(
                    answer=sentence,
                    confidence=0.7,
                    abstained=False,
                    citations=[ProviderCitation(evidence_id=evidence_id)],
                    claims=[claim],
                )
        return _abstained_result()


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _WORD.findall(text)}


def _specific_terms(query: str) -> set[str]:
    return {
        token for token in _tokens(query)
        if len(token) >= 3 and token.isalpha() and token not in _NON_SPECIFIC_QUERY_TERMS
    }


def _abstained_result() -> AnswerResult:
    return AnswerResult(
        answer="",
        confidence=0.0,
        abstained=True,
        citations=[],
        claims=[],
    )
