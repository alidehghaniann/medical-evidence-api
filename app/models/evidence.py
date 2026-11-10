from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class EvidenceRecord(BaseModel):
    """A normalized record with source provenance retained."""

    model_config = ConfigDict(str_strip_whitespace=True)

    source: str
    source_id: str | None = None
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


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    language: str = Field(default="en", pattern="^(en|fa)$")
    limit: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    language: str
    answer: str
    abstained: bool
    uncertainty: str | None = None
    evidence: list[EvidenceRecord]
    provider_status: dict[str, str]
    disclaimer: str
