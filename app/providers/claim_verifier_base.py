from typing import Protocol

from app.models.answer import ClaimToVerify, ClaimVerification


class ClaimVerifier(Protocol):
    async def verify(
        self,
        claims: list[ClaimToVerify],
    ) -> list[ClaimVerification]:
        """Judge support using only the evidence explicitly cited by each claim."""
