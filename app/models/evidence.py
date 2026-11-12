from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.models.medical_query import detect_query_language


class EvidenceAssessment(BaseModel):
    """Transparent baseline assessment; values are not clinically validated."""

    evidence_type: str | None = None
    evidence_strength: str = "unknown"
    source_authority: float | None = None
    recency: float | None = None
    guideline_priority: bool | None = None
    assessment_confidence: float = 0.0
    assessment_reasons: list[str] = Field(default_factory=list)


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
    assessment: EvidenceAssessment = Field(default_factory=EvidenceAssessment)

    @model_validator(mode="after")
    def set_citation_id(self):
        if self.citation_id is None and self.source_id:
            self.citation_id = f"{self.source}:{self.source_id}"
        return self


class SearchPipelineDiagnostics(BaseModel):
    """Secret-free processing counts and provider states for one search."""

    query_understanding_provider: str = "unknown"
    query_understanding_status: str = "unknown"
    intent: str | None = None
    medical_terms_count: int = 0
    provider_status: dict[str, str] = Field(default_factory=dict)
    retrieved_by_provider: dict[str, int] = Field(default_factory=dict)
    retrieved_count: int = 0
    normalized_count: int = 0
    deduplicated_count: int = 0
    ranked_count: int = 0
    evidence_assessed_count: int = 0


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
    pipeline_diagnostics: SearchPipelineDiagnostics | None = None

    @model_validator(mode="after")
    def set_evidence_summary(self):
        self.evidence_count = len(self.evidence)
        self.evidence_found = bool(self.evidence)
        return self
