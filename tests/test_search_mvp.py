import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service
from app.core.config import Settings
from app.main import app
from app.models.evidence import EvidenceRecord
from app.models.medical_query import SearchQueryRepresentation, normalize_query_text
from app.services.search import MedicalSearchService


class FakeEvidenceProvider:
    def __init__(self, name, records=None, fail=False):
        self.name = name
        self.records = records or []
        self.fail = fail
        self.queries = []

    async def search(self, query, limit=10):
        self.queries.append(query)
        if self.fail:
            raise httpx.ConnectError("offline")
        return self.records[:limit]


def record(source, source_id, title, *, doi=None, abstract="Study abstract", year=2024):
    return EvidenceRecord(
        source=source,
        source_id=source_id,
        title=title,
        abstract=abstract,
        journal="Medical Journal",
        publication_year=year,
        doi=doi,
        url=f"https://example.org/{source}/{source_id}",
        retrieved_at="2026-01-01T00:00:00Z",
    )


class FakeQueryUnderstander:
    def __init__(self, search_query="", medical_terms=None, fail=False):
        self.search_query = search_query
        self.medical_terms = medical_terms or search_query.split()
        self.fail = fail
        self.calls = []

    async def understand_query(self, query, normalized_query):
        self.calls.append((query, normalized_query))
        if self.fail:
            raise httpx.ConnectError("offline")
        return SearchQueryRepresentation(search_query=self.search_query, medical_terms=self.medical_terms, intent="general")


def with_providers(providers, minimum_score=0.0, query_understander=None):
    service = MedicalSearchService(providers, Settings(minimum_evidence_score=minimum_score), query_understander=query_understander)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def clear_overrides():
    app.dependency_overrides.clear()


def test_search_returns_combined_normalized_source_metadata_and_schema():
    pubmed = FakeEvidenceProvider("pubmed", [record("pubmed", "101", "Metformin chronic kidney evidence")])
    europepmc = FakeEvidenceProvider("europepmc", [record("europepmc", "PMC202", "Metformin kidney outcomes")])
    try:
        with with_providers([pubmed, europepmc]) as client:
            response = client.post("/v1/medical/search", json={"query": "metformin kidney evidence", "language": "en"})
        assert response.status_code == 200
        payload = response.json()
        assert pubmed.queries == ["metformin kidney evidence"]
        assert europepmc.queries == ["metformin kidney evidence"]
        assert payload["query"] == "metformin kidney evidence"
        assert payload["evidence_found"] is True
        assert payload["evidence_count"] == 2
        assert payload["abstained"] is False
        assert payload["abstention_reason"] is None
        assert {item["source"] for item in payload["evidence"]} == {"pubmed", "europepmc"}
        assert payload["evidence"][0]["rank_score"] >= payload["evidence"][1]["rank_score"]
        item = payload["evidence"][0]
        assert item["title"] and item["abstract"] and item["publication_year"]
        assert item["source_id"] and item["citation_id"] == f"{item['source']}:{item['source_id']}"
        assert item["url"] and item["journal"]
        assert payload["provider_status"] == {"pubmed": "ok", "europepmc": "ok"}
    finally:
        clear_overrides()


def test_persian_search_uses_general_query_understanding_and_preserves_original_question():
    query = "\u0634\u0648\u0627\u0647\u062f \u0645\u062a\u0641\u0648\u0631\u0645\u06cc\u0646 \u062f\u0631 \u0628\u06cc\u0645\u0627\u0631\u06cc \u0645\u0632\u0645\u0646 \u06a9\u0644\u06cc\u0647 \u0686\u06cc\u0633\u062a\u061f"
    providers = [
        FakeEvidenceProvider("pubmed", [record("pubmed", "201", "Metformin kidney study")]),
        FakeEvidenceProvider("europepmc"),
    ]
    understander = FakeQueryUnderstander("metformin safety chronic kidney disease", ["metformin", "kidney", "disease"])
    try:
        with with_providers(providers, query_understander=understander) as client:
            payload = client.post("/v1/medical/search", json={"query": query, "language": "en"}).json()
        assert payload["query"] == query
        assert payload["language"] == "fa"
        assert payload["evidence_count"] == 1
        assert providers[0].queries == ["metformin safety chronic kidney disease"]
        assert providers[1].queries == ["metformin safety chronic kidney disease"]
        assert understander.calls[0][0] == query
        arabic_variant = "\u0645\u062a\u0641\u0648\u0631\u0645\u064a\u0646 \u0628\u064a\u0645\u0627\u0631\u064a \u0645\u0632\u0645\u0646 \u0643\u0644\u064a\u0647"
        persian_variant = "\u0645\u062a\u0641\u0648\u0631\u0645\u06cc\u0646 \u0628\u06cc\u0645\u0627\u0631\u06cc \u0645\u0632\u0645\u0646 \u06a9\u0644\u06cc\u0647"
        assert normalize_query_text(arabic_variant) == normalize_query_text(persian_variant)
    finally:
        clear_overrides()


@pytest.mark.parametrize(("query", "search_query", "terms"), [
    ("\u0641\u0634\u0627\u0631 \u062e\u0648\u0646 \u0648 \u0646\u0627\u0631\u0633\u0627\u06cc\u06cc \u0642\u0644\u0628\u06cc", "hypertension heart failure", ["hypertension", "heart", "failure"]),
    ("\u062f\u06cc\u0627\u0628\u062a \u0648 \u06a9\u0645\u200c\u06a9\u0627\u0631\u06cc \u062a\u06cc\u0631\u0648\u0626\u06cc\u062f", "diabetes hypothyroidism", ["diabetes", "hypothyroidism"]),
    ("\u062f\u0631\u0645\u0627\u0646 \u0639\u0641\u0648\u0646\u062a \u0628\u0627\u06a9\u062a\u0631\u06cc\u0627\u06cc\u06cc", "bacterial infection treatment", ["bacterial", "infection"]),
    ("\u0639\u0648\u0627\u0631\u0636 \u062f\u0627\u0631\u0648\u06cc \u0622\u062a\u0648\u0631\u0648\u0627\u0633\u062a\u0627\u062a\u06cc\u0646", "atorvastatin adverse effects", ["atorvastatin", "adverse", "effects"]),
    ("\u0645\u0635\u0631\u0641 metformin \u062f\u0631 \u0628\u06cc\u0645\u0627\u0631\u06cc \u06a9\u0644\u06cc\u0648\u06cc", "metformin chronic kidney disease", ["metformin", "kidney", "disease"]),
])
def test_general_persian_domains_use_supplied_search_representation(query, search_query, terms):
    understander = FakeQueryUnderstander(search_query, terms)
    providers = [FakeEvidenceProvider("pubmed"), FakeEvidenceProvider("europepmc")]
    try:
        with with_providers(providers, query_understander=understander) as client:
            response = client.post("/v1/medical/search", json={"query": query})
        assert response.status_code == 200
        assert response.json()["language"] == "fa"
        assert providers[0].queries == [search_query]
        assert providers[1].queries == [search_query]
    finally:
        clear_overrides()


def test_persian_query_without_configured_understander_abstains_without_translation():
    query = "\u0622\u06cc\u0627 \u062f\u0631\u0645\u0627\u0646 \u0627\u06cc\u0646 \u0628\u06cc\u0645\u0627\u0631\u06cc \u0646\u0627\u0634\u0646\u0627\u062e\u062a\u0647 \u0645\u0624\u062b\u0631 \u0627\u0633\u062a\u061f"
    providers = [FakeEvidenceProvider("pubmed"), FakeEvidenceProvider("europepmc")]
    try:
        with with_providers(providers) as client:
            payload = client.post("/v1/medical/search", json={"query": query, "language": "en"}).json()
        assert payload["language"] == "fa"
        assert payload["evidence"] == []
        assert payload["abstained"] is True
        assert payload["provider_status"]["query_understanding"] == "not_configured"
        assert providers[0].queries == providers[1].queries == []
    finally:
        clear_overrides()


def test_query_understanding_failure_is_controlled_and_does_not_search_original_persian():
    providers = [FakeEvidenceProvider("pubmed"), FakeEvidenceProvider("europepmc")]
    try:
        with with_providers(providers, query_understander=FakeQueryUnderstander(fail=True)) as client:
            payload = client.post("/v1/medical/search", json={"query": "\u067e\u0631\u0633\u0634 \u067e\u0632\u0634\u06a9\u06cc \u062f\u0631\u0628\u0627\u0631\u0647 \u0628\u06cc\u0645\u0627\u0631\u06cc"}).json()
        assert payload["abstained"] is True
        assert payload["provider_status"]["query_understanding"] == "unavailable"
        assert providers[0].queries == providers[1].queries == []
    finally:
        clear_overrides()


def test_deduplication_across_providers_and_rank_order():
    duplicate = record("pubmed", "301", "Metformin and chronic kidney disease", doi="10.1/example")
    near_duplicate = record("europepmc", "301", "Metformin and chronic kidney disease", doi="10.1/example")
    weaker = record("europepmc", "302", "Unrelated study topic", abstract="No shared topic")
    try:
        with with_providers([
            FakeEvidenceProvider("pubmed", [duplicate]),
            FakeEvidenceProvider("europepmc", [near_duplicate, weaker]),
        ]) as client:
            payload = client.post("/v1/medical/search", json={"query": "metformin chronic kidney disease"}).json()
        assert payload["evidence_count"] == 2
        assert len({item["source_id"] for item in payload["evidence"]}) == 2
        assert payload["evidence"][0]["rank_score"] >= payload["evidence"][1]["rank_score"]
    finally:
        clear_overrides()


def test_empty_results_return_explicit_abstention():
    try:
        with with_providers([FakeEvidenceProvider("pubmed"), FakeEvidenceProvider("europepmc")]) as client:
            response = client.post("/v1/medical/search", json={"query": "very rare topic"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["evidence_found"] is False
        assert payload["evidence_count"] == 0
        assert payload["abstained"] is True
        assert payload["abstention_reason"]
    finally:
        clear_overrides()


def test_one_provider_failure_keeps_other_provider_results():
    available = FakeEvidenceProvider("pubmed", [record("pubmed", "401", "Kidney evidence study")])
    try:
        with with_providers([FakeEvidenceProvider("europepmc", fail=True), available]) as client:
            payload = client.post("/v1/medical/search", json={"query": "kidney evidence study"}).json()
        assert payload["provider_status"] == {"europepmc": "unavailable", "pubmed": "ok"}
        assert payload["evidence_count"] == 1
        assert payload["evidence"][0]["source_id"] == "401"
    finally:
        clear_overrides()


def test_malformed_query_is_rejected_with_validation_error():
    with with_providers([]) as client:
        response = client.post("/v1/medical/search", json={"query": "x"})
    clear_overrides()
    assert response.status_code == 422


def test_evidence_below_quality_threshold_has_abstention_reason():
    try:
        with with_providers([
            FakeEvidenceProvider("pubmed", [record("pubmed", "501", "Metformin kidney research")]),
            FakeEvidenceProvider("europepmc"),
        ], minimum_score=1.0) as client:
            payload = client.post("/v1/medical/search", json={"query": "metformin kidney"}).json()
        assert payload["evidence_found"] is True
        assert payload["evidence_count"] == 1
        assert payload["abstained"] is True
        assert payload["abstention_reason"]
    finally:
        clear_overrides()


def test_unmatched_main_query_term_forces_abstention_despite_broad_search_hits():
    broad_hit = record("pubmed", "601", "Disease treatment evidence in clinical research")
    try:
        with with_providers([
            FakeEvidenceProvider("pubmed", [broad_hit]),
            FakeEvidenceProvider("europepmc"),
        ]) as client:
            payload = client.post("/v1/medical/search", json={
                "query": "zxqvplmokylop imaginary disorder treatment evidence",
            }).json()
        assert payload["evidence_found"] is False
        assert payload["evidence_count"] == 0
        assert payload["abstained"] is True
        assert payload["abstention_reason"] == "No matching evidence was found."
    finally:
        clear_overrides()


def test_demo_page_is_served_by_fastapi():
    with TestClient(app) as client:
        response = client.get("/demo")
    assert response.status_code == 200
    assert "Medical Evidence Search" in response.text
    assert 'dir="rtl"' in response.text
    assert 'dir="auto"' in response.text
    assert "unicode-bidi: isolate" in response.text
    assert "detectedLanguage(query.value)" in response.text
    assert "فارسی" in response.text
