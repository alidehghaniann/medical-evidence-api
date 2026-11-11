from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

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


class AnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool
    citations: list[ProviderCitation]
    claims: list[AnswerClaim] = Field(default_factory=list)


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
