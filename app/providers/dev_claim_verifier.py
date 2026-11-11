"""Exact-text claim verifier for development and tests only."""

from app.models.answer import ClaimToVerify, ClaimVerification
from app.providers.claim_verifier_base import ClaimVerifier


class DevClaimVerifier(ClaimVerifier):
    """Support only claims copied verbatim from their cited title or abstract.

    This DEVELOPMENT/TEST ONLY verifier does no semantic or clinical reasoning.
    Claims it cannot match exactly are marked unsupported so the existing
    application abstains.
    """

    async def close(self) -> None:
        """Match the lifecycle used by network-backed verifiers."""

    async def verify(
        self,
        claims: list[ClaimToVerify],
    ) -> list[ClaimVerification]:
        verifications = []
        for claim in claims:
            normalized_claim = _normalize(claim.text)
            has_citations = bool(claim.citation_ids) and len(claim.citation_ids) == len(claim.cited_evidence)
            supported = has_citations and any(
                normalized_claim in _normalize(text)
                for record in claim.cited_evidence
                for text in (record.title, record.abstract or "")
            )
            verifications.append(ClaimVerification(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                citation_ids=claim.citation_ids,
                status="supported" if supported else "unsupported",
                confidence=0.7 if supported else 0.0,
            ))
        return verifications


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
