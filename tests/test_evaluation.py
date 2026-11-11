from collections import Counter
import sys

import httpx
import pytest

from app.core.config import Settings
from app.evaluation.metrics import calculate_metrics
from app.evaluation.benchmark import (
    BenchmarkDataset,
    build_report,
    evaluate_service,
    load_dataset,
    render_markdown,
)
from app.models.evidence import EvidenceRecord
from app.services.search import MedicalSearchService
from app.evaluation.__main__ import main as evaluation_main


def test_metrics_calculate_precision_recall_f1_mrr_and_ndcg_and_remove_duplicates():
    result = calculate_metrics(
        ["pmid:1", "pmid:2", "pmid:1", "pmid:3"],
        {"pmid:1": 2, "pmid:2": 0, "pmid:3": 1},
        3,
    )

    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.recall_at_k == 1.0
    assert result.f1_at_k == pytest.approx(0.8)
    assert result.reciprocal_rank == 1.0
    assert result.ndcg_at_k == pytest.approx(3.5 / (3.0 + 1.0 / 1.5849625007))
    assert result.judgment_coverage_at_k == 1.0
    assert result.duplicate_identifiers_removed == 1


def test_unjudged_and_missing_identifier_records_are_not_counted_as_irrelevant():
    result = calculate_metrics([None, "pmid:1", None], {"pmid:1": 1}, 3)

    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.judgment_coverage_at_k == pytest.approx(1 / 3)
    assert result.unique_retrieved_at_k == 3

    missed_known_relevant = calculate_metrics([None, "pmid:unknown"], {"pmid:relevant": 1}, 3)
    assert missed_known_relevant.precision_at_k is None
    assert missed_known_relevant.recall_at_k == 0.0
    assert missed_known_relevant.reciprocal_rank == 0.0
    assert missed_known_relevant.ndcg_at_k == 0.0


def test_metric_edge_cases_for_empty_results_no_relevant_ids_and_unjudged_case():
    no_results = calculate_metrics([], {"pmid:1": 1}, 5)
    assert no_results.precision_at_k == 0.0
    assert no_results.recall_at_k == 0.0
    assert no_results.f1_at_k == 0.0
    assert no_results.reciprocal_rank == 0.0
    assert no_results.ndcg_at_k == 0.0

    no_known_relevant = calculate_metrics(["pmid:2"], {"pmid:2": 0}, 5)
    assert no_known_relevant.precision_at_k == 0.0
    assert no_known_relevant.recall_at_k is None
    assert no_known_relevant.f1_at_k is None
    assert no_known_relevant.reciprocal_rank is None
    assert no_known_relevant.ndcg_at_k is None

    unjudged = calculate_metrics(["pmid:1"], None, 5)
    assert all(getattr(unjudged, name) is None for name in (
        "precision_at_k", "recall_at_k", "f1_at_k", "reciprocal_rank", "ndcg_at_k", "judgment_coverage_at_k"
    ))


def test_metrics_reject_invalid_k_and_judgments():
    with pytest.raises(ValueError, match="k must"):
        calculate_metrics([], {}, 0)
    with pytest.raises(ValueError, match="non-empty identifiers"):
        calculate_metrics([], {" ": 1}, 1)
    with pytest.raises(ValueError, match="grades 0 through 3"):
        calculate_metrics([], {"pmid:1": -1}, 1)
    with pytest.raises(ValueError, match="duplicate identifiers"):
        calculate_metrics([], {"PMID:1": 1, "pmid:1": 2}, 1)


def test_live_evaluation_cli_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.evaluation"])
    with pytest.raises(SystemExit) as exit_info:
        evaluation_main()
    assert exit_info.value.code == 2


def test_benchmark_dataset_loads_versioned_balanced_cases_and_marks_unjudged():
    dataset = load_dataset()
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 22
    assert {case.language for case in dataset.cases} == {"en", "fa"}
    assert len({case.category for case in dataset.cases}) == 11
    assert sum(case.judgment_status == "partial" for case in dataset.cases) == 4
    assert sum(case.judgment_status == "unjudged" for case in dataset.cases) == 18
    assert all(case.expected_search_query for case in dataset.cases if case.language == "fa")

    serialized = dataset.model_dump(mode="json")
    assert BenchmarkDataset.model_validate(serialized).model_dump(mode="json") == serialized


def test_canonical_identifiers_prefer_pmid_then_doi_and_preserve_source_namespace():
    from app.evaluation.benchmark import evidence_identifier

    assert evidence_identifier(_record("pubmed", "123", doi="10.1000/xyz")) == "pmid:123"
    assert evidence_identifier(_record("europepmc", "PMC123", doi="10.1000/xyz")) == "pmcid:pmc123"
    assert evidence_identifier(_record("other", None, doi="https://doi.org/10.1000/xyz")) == "doi:10.1000/xyz"
    assert evidence_identifier(_record("other", None)) is None


@pytest.mark.asyncio
async def test_benchmark_calls_shared_search_service_and_produces_reproducible_metrics():
    dataset = load_dataset()
    case = next(item for item in dataset.cases if item.case_id == "en-drug-safety-metformin-ckd")
    record = _record("pubmed", "34009711", "Safety outcomes metformin reduced renal function")

    class FakeProvider:
        def __init__(self, name, records):
            self.name = name
            self.records = records
            self.queries = []

        async def search(self, query, limit=10):
            self.queries.append(query)
            return self.records[:limit]

    pubmed = FakeProvider("pubmed", [record])
    europepmc = FakeProvider("europepmc", [
        _record("europepmc", "PMC987654", "Metformin kidney function unrelated secondary record")
    ])
    service = MedicalSearchService([pubmed, europepmc], Settings(minimum_evidence_score=0.0))
    single_case_dataset = dataset.model_copy(update={"cases": [case]})
    first = await evaluate_service(single_case_dataset, service, k=5, evaluation_mode="mocked")
    second = await evaluate_service(single_case_dataset, service, k=5, evaluation_mode="mocked")

    assert pubmed.queries == europepmc.queries == [case.query, case.query]
    assert first["overall_metrics"]["precision_at_k"] == 1.0
    assert first["overall_metrics"]["recall_at_k"] == 1.0
    assert first["overall_metrics"]["reciprocal_rank"] == 1.0
    assert set(first["cases"][0]["retrieved_identifiers"]) == {"pmid:34009711", "pmcid:pmc987654"}
    assert first["cases"][0]["metrics"]["judgment_coverage_at_k"] == 0.5
    diagnostics = {item["identifier"]: item for item in first["cases"][0]["ranking_diagnostics"]}
    assert diagnostics["pmid:34009711"]["relevance_judgment"] == 2
    assert diagnostics["pmcid:pmc987654"]["relevance_judgment"] == "unjudged"
    assert {"relevance_score", "authority_score", "recency_score", "evidence_type", "rank_score"} <= set(
        diagnostics["pmid:34009711"]
    )
    assert first["cases"][0]["metrics"] == second["cases"][0]["metrics"]
    assert render_markdown(first).startswith("# Medical Retrieval Evaluation")


@pytest.mark.asyncio
async def test_benchmark_reports_provider_failures_without_claiming_unjudged_success():
    dataset = load_dataset()
    case = next(item for item in dataset.cases if item.case_id == "fa-treatment-hypertension")

    class FailingProvider:
        name = "pubmed"

        async def search(self, query, limit=10):
            raise httpx.ConnectError("offline")

    service = MedicalSearchService([FailingProvider()], Settings(minimum_evidence_score=0.0))
    report = await evaluate_service(dataset.model_copy(update={"cases": [case]}), service, evaluation_mode="mocked")

    assert report["unjudged_case_count"] == 1
    assert report["judged_case_count"] == 0
    assert report["overall_metrics"]["precision_at_k"] is None
    assert report["failed_cases"] == [{
        "case_id": "fa-treatment-hypertension",
        "unavailable_sources": [],
        "query_understanding": "not_configured",
    }]
    assert report["cases"][0]["judgment_status"] == "unjudged"


def test_report_builder_does_not_emit_metrics_for_unjudged_cases():
    dataset = load_dataset()
    case = next(item for item in dataset.cases if item.judgment_status == "unjudged")
    report = build_report(
        dataset.model_copy(update={"cases": [case]}),
        [{"case_id": case.case_id, "status": "unjudged"}],
        source_record_counts=Counter(),
        source_case_counts=Counter(),
        missing_relevant=[],
        ranking_errors=[],
        failed_cases=[],
        k=5,
        evaluation_mode="mocked",
    )
    assert report["overall_metrics"]["precision_at_k"] is None
    assert report["per_language_metrics"][case.language]["precision_at_k"] is None


def _record(source, source_id, title="Metformin safety reduced renal function", doi=None):
    return EvidenceRecord(
        source=source,
        source_id=source_id,
        title=title,
        abstract="Metformin and kidney function safety outcomes.",
        doi=doi,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{source_id or 'unknown'}/",
        retrieved_at="2026-01-01T00:00:00Z",
    )
