import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.models.evidence import EvidenceRecord


class ProviderCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


class ClaimToVerify(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str]
    cited_evidence: list[EvidenceRecord]


CITATION_TOKEN = re.compile(r"^evidence-\d+$")
_CITATION_IN_TEXT = re.compile(r"[\[(]?\s*evidence-\d+\s*[\])]?", re.IGNORECASE)
_ENDS_SENTENCE = re.compile(r"[.!?\u061f]$")


def classify_citation_style(text: str) -> str:
    """Describe how an answer renders its citation IDs, without judging correctness."""
    styles = set()
    for match in _CITATION_IN_TEXT.finditer(text):
        token = match.group()
        if token.startswith("["):
            styles.add("bracketed")
        elif token.startswith("("):
            styles.add("parenthesised")
        else:
            styles.add("bare")
    return "+".join(sorted(styles)) or "none"


class AnswerResult(BaseModel):
    """A provider answer whose text is defined by its claims.

    A provider may send `answer`, or `citation_style` plus `claims`, but never a
    free-form narrative beside a separate claim list: the answer always covers
    exactly the claims, so no claim can be missing from it and no sentence can
    appear in it without a claim.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    citations: list[ProviderCitation]
    claims: list[AnswerClaim] = Field(default_factory=list)
    citation_style: Literal["none", "bare", "bracketed", "parenthesised"] = "bracketed"

    @model_validator(mode="after")
    def reconcile_answer_with_claims(self):
        if not self.claims:
            return self
        expected = render_claim_answer(self.claims, self.citation_style)
        if not self.answer.strip():
            self.answer = expected
            return self
        supplied = _comparable_answer_text(self.answer)
        if supplied != _comparable_answer_text(expected):
            raise ValueError(
                "Answer text does not match its claims. Send the claims, or the claims with a "
                "citation_style, instead of a separate narrative answer."
            )
        self.answer = expected
        return self


def _comparable_answer_text(text: str) -> str:
    """Reduce answer text to what must match: wording and order, not citation layout."""
    text = _CITATION_IN_TEXT.sub(" ", text)
    text = re.sub(r"\s+([.,;:!?\u061f])", r"\1", text)
    return " ".join(text.casefold().split()).strip(" .,;:!?\u061f")


def render_claim_answer(claims: list[AnswerClaim], citation_style: str = "bracketed") -> str:
    """Join claims into answer text, citing each claim with its own evidence IDs."""
    parts = []
    for claim in claims:
        text = claim.text.strip()
        if claim.citation_ids:
            sources = ", ".join(_render_citation(item, citation_style) for item in claim.citation_ids)
            if not _ENDS_SENTENCE.search(text):
                text = f"{text}."
            text = f"{text} {sources}"
        parts.append(text)
    return " ".join(parts)


def _render_citation(evidence_id: str, citation_style: str) -> str:
    if not CITATION_TOKEN.match(evidence_id):
        raise ValueError(f"Citation IDs must look like evidence-<n>, got {evidence_id!r}")
    if citation_style == "bare":
        return evidence_id
    if citation_style == "parenthesised":
        return f"({evidence_id})"
    return f"[{evidence_id}]"


class ClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    citation_ids: list[str]
    status: Literal["supported", "unsupported", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)


class AnswerCitation(BaseModel):
    evidence_id: str
    claim: str | None = None
    title: str
    source: str
    source_id: str | None = None
    url: HttpUrl


class AskResponse(BaseModel):
    answer: str
    language: str
    citations: list[AnswerCitation]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    evidence_count: int = Field(ge=0)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    diagnostics: "AskDiagnostics" = Field(default_factory=lambda: AskDiagnostics())


class AskDiagnostics(BaseModel):
    """Safe, request-scoped explanation of the ask pipeline."""

    reason_code: str | None = None
    query_language: str | None = None
    query_intent: str | None = None
    stages: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
