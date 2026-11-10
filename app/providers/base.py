from abc import ABC, abstractmethod

from app.models.evidence import EvidenceRecord


class EvidenceProvider(ABC):
    name: str

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[EvidenceRecord]:
        """Return evidence records or raise a provider-specific exception."""
