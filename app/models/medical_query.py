import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MedicalIntent = Literal[
    "treatment",
    "diagnosis",
    "adverse_effects",
    "drug_interaction",
    "safety",
    "efficacy",
    "prevention",
    "prognosis",
    "epidemiology",
    "screening",
    "mechanism",
    "evidence",
    "clinical_trial",
    "guideline",
    "general",
]


class SearchQueryRepresentation(BaseModel):
    """Validated, concise English query representation for biomedical indexes."""

    model_config = ConfigDict(extra="forbid")

    search_query: str = Field(min_length=1, max_length=500)
    medical_terms: list[str] = Field(min_length=1, max_length=30)
    intent: MedicalIntent = "general"

    @field_validator("search_query", "medical_terms")
    @classmethod
    def require_latin_search_terms(cls, value):
        values = [value] if isinstance(value, str) else value
        if any(not item.isascii() for item in values):
            raise ValueError("search representation must use Latin biomedical terms")
        return value


class MedicalQuery(BaseModel):
    """Language-aware query representation shared by search and answer flows."""

    original_query: str
    language: Literal["en", "fa"]
    normalized_query: str
    search_query: str | None = None
    medical_terms: list[str] = Field(default_factory=list)
    intent: MedicalIntent = "general"


def detect_query_language(query: str) -> Literal["en", "fa"]:
    """Detect Persian from Arabic-script characters; default to English."""
    return "fa" if re.search(r"[\u0600-\u06ff]", query) else "en"


def normalize_query_text(query: str) -> str:
    """Apply Unicode and common Arabic/Persian character normalization only."""
    normalized = unicodedata.normalize("NFKC", query).translate(str.maketrans({
        "\u064a": "\u06cc",  # Arabic yeh -> Persian yeh
        "\u0649": "\u06cc",
        "\u0643": "\u06a9",  # Arabic kaf -> Persian kaf
    }))
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", normalized)
    normalized = normalized.replace("\u200c", " ")
    return " ".join(normalized.split())
