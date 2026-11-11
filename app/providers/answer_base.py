from typing import Protocol

from app.models.answer import AnswerResult
from app.models.evidence import EvidenceRecord


class AnswerProvider(Protocol):
    async def answer(self, query: str, evidence: list[EvidenceRecord], language: str) -> AnswerResult:
        """Synthesize an answer only from the supplied evidence records."""
