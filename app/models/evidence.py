from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.models.medical_query import detect_query_language


class EvidenceRecord(BaseModel):
    """A normalized record with source provenance retained."""

    model_config = ConfigDict(str_strip_whitespace=True)

    source: str
    source_id: str | None = None
    citation_id: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    url: HttpUrl
    evidence_type: str | None = None
    retrieved_at: str
    relevance_score: float = 0.0
    authority_score: float = 0.0
    recency_score: float = 0.0
    rank_score: float = 0.0

    @model_validator(mode="after")
    def set_citation_id(self):
        if self.citation_id is None and self.source_id:
            self.citation_id = f"{self.source}:{self.source_id}"
        return self


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    language: str = Field(default="en", pattern="^(en|fa)$")
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def detect_language_from_query(self):
        self.language = detect_query_language(self.query)
        return self


class SearchResponse(BaseModel):
    query: str
    language: str
    answer: str
    abstained: bool
    uncertainty: str | None = None
    evidence: list[EvidenceRecord]
    evidence_found: bool = False
    evidence_count: int = 0
    abstention_reason: str | None = None
    provider_status: dict[str, str]
    disclaimer: str

    @model_validator(mode="after")
    def set_evidence_summary(self):
        self.evidence_count = len(self.evidence)
        self.evidence_found = bool(self.evidence)
        return self
