from app.evidence.processing import deduplicate, normalize_record, rank_records, validate_citations
from app.models.evidence import EvidenceRecord


def record(**overrides):
    data = dict(source="pubmed", source_id="123", title="  Randomized trial of treatment ",
                abstract="  Abstract   text ", authors=["A Smith", "A Smith"], journal="Journal",
                publication_year=2024, doi="https://doi.org/10.1000/test",
                url="https://pubmed.ncbi.nlm.nih.gov/123/", retrieved_at="2026-01-01T00:00:00Z")
    data.update(overrides)
    return EvidenceRecord(**data)


def test_normalize_and_classify_evidence():
    item = normalize_record(record())
    assert item.title == "Randomized trial of treatment"
    assert item.abstract == "Abstract text"
    assert item.doi == "10.1000/test"
    assert item.authors == ["A Smith"]
    assert item.evidence_type == "randomized_trial"


def test_duplicate_removal_by_pmid_doi_and_near_title():
    original = record()
    duplicate_id = record(source_id="123", doi="10.1000/other", title="Different title")
    duplicate_title = record(source_id="456", doi="10.1000/different", title="Randomized trial of treatment!")
    assert len(deduplicate([original, duplicate_id, duplicate_title])) == 1


def test_ranking_is_relevance_sensitive_and_citations_require_provenance():
    relevant = record(source_id="1", title="Helicobacter pylori treatment guideline")
    irrelevant = record(source_id="2", title="Cardiac imaging methods")
    ranked = rank_records([irrelevant, relevant], "Helicobacter pylori treatment")
    assert ranked[0].source_id == "1"
    assert ranked[0].rank_score > ranked[1].rank_score
    assert validate_citations(ranked)
