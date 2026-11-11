import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.evidence import SearchRequest
from app.models.medical_query import MedicalIntent
from app.services.search import MedicalSearchService

DATASET_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "medical_retrieval_v1.json"
REPOSITORY_ROOT = DATASET_PATH.parent.parent
METRIC_NAMES = ("precision_at_k", "recall_at_k", "f1_at_k", "reciprocal_rank", "ndcg_at_k")
METRIC_DEFINITIONS = {
    "precision_at_k": "Positive judged retrieved IDs divided by judged retrieved IDs in unique top K; null if top K contains results but none are judged; zero when no results are returned.",
    "recall_at_k": "Positive judged retrieved IDs in top K divided by all known positive judgments; null when no known positive judgments exist.",
    "f1_at_k": "Harmonic mean of precision_at_k and recall_at_k when both are defined.",
    "reciprocal_rank": "Reciprocal rank of the first positive among explicitly judged top-K results; unjudged items are skipped; null if no positive judgments exist.",
    "ndcg_at_k": "Graded gain (2^grade - 1) over explicitly judged top-K results, with unjudged items skipped and judged ranks compressed, divided by ideal DCG over all known positive judgments; null if no positive judgments exist.",
    "judgment_coverage_at_k": "Number of judged unique retrieved results divided by unique returned results in top K; zero when no results are returned.",
}


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    query: str = Field(min_length=3)
    language: str
    category: str
    intent: MedicalIntent
    expected_medical_concepts: list[str]
    expected_search_query: str | None = None
    relevance_judgments: dict[str, int] | None = None
    expected_source_types: list[str] = Field(default_factory=list)
    notes: str

    @model_validator(mode="after")
    def validate_relevance(self):
        if self.language not in {"en", "fa"}:
            raise ValueError("language must be en or fa")
        if self.relevance_judgments is not None and any(
            not identifier.strip() or grade not in {0, 1, 2, 3}
            for identifier, grade in self.relevance_judgments.items()
        ):
            raise ValueError("relevance judgments must use identifiers and grades 0 through 3")
        return self

    @property
    def judgment_status(self) -> str:
        return "unjudged" if self.relevance_judgments is None else "partial"


class BenchmarkDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    cases: list[BenchmarkCase]

    @model_validator(mode="after")
    def validate_unique_cases(self):
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark case IDs must be unique")
        return self


def load_dataset(path: Path = DATASET_PATH) -> BenchmarkDataset:
    return BenchmarkDataset.model_validate_json(path.read_text(encoding="utf-8"))


def evidence_identifier(record) -> str | None:
    source_id = record.source_id
    if record.source.casefold() in {"pubmed", "europepmc"} and source_id and source_id.isdigit():
        return f"pmid:{source_id}"
    if source_id and source_id.upper().startswith("PMC"):
        return f"pmcid:{source_id.casefold()}"
    if record.doi:
        return f"doi:{record.doi.removeprefix('https://doi.org/').casefold()}"
    if not source_id:
        return None
    return f"{record.source.casefold()}:{source_id.casefold()}"


async def evaluate_service(
    dataset: BenchmarkDataset,
    service: MedicalSearchService,
    *,
    k: int = 10,
    evaluation_mode: str = "live",
) -> dict[str, Any]:
    from app.evaluation.metrics import calculate_metrics

    results = []
    source_record_counts: Counter[str] = Counter()
    source_case_counts: Counter[str] = Counter()
    provider_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    query_understanding_status_counts: Counter[str] = Counter()
    missing_relevant = []
    ranking_errors = []
    failed_cases = []

    for case in dataset.cases:
        try:
            response = await service.search(SearchRequest(query=case.query, language=case.language, limit=k))
        except Exception as exc:
            failed_cases.append({"case_id": case.case_id, "error_type": type(exc).__name__})
            results.append({"case_id": case.case_id, "status": "failed", "error_type": type(exc).__name__})
            continue

        identifiers = [evidence_identifier(record) for record in response.evidence]
        metrics = calculate_metrics(identifiers, case.relevance_judgments, k)
        found_ids = {identifier for identifier in identifiers if identifier is not None}
        known_relevant = [
            identifier for identifier, grade in (case.relevance_judgments or {}).items() if grade > 0
        ]
        absent = sorted(set(known_relevant) - found_ids)
        if absent:
            missing_relevant.append({"case_id": case.case_id, "identifiers": absent})

        graded = case.relevance_judgments or {}
        known_ranked = [
            (rank, identifier)
            for rank, identifier in enumerate(identifiers, start=1)
            if identifier in graded
        ]
        for index, (rank, identifier) in enumerate(known_ranked):
            if graded[identifier] > 0 and any(graded[earlier] == 0 for _, earlier in known_ranked[:index]):
                ranking_errors.append({"case_id": case.case_id, "relevant_identifier": identifier, "rank": rank})

        case_sources = sorted({record.source for record in response.evidence})
        source_case_counts.update(case_sources)
        source_record_counts.update(record.source for record in response.evidence)
        for source, status in response.provider_status.items():
            if source == "query_understanding":
                query_understanding_status_counts[status] += 1
            else:
                provider_status_counts[source][status] += 1
        unavailable = sorted(
            source for source, status in response.provider_status.items()
            if source != "query_understanding" and status == "unavailable"
        )
        query_unavailable = response.provider_status.get("query_understanding") in {
            "unavailable", "not_configured"
        } and case.language == "fa"
        if unavailable or query_unavailable:
            failure = {"case_id": case.case_id, "unavailable_sources": unavailable}
            if query_unavailable:
                failure["query_understanding"] = response.provider_status.get("query_understanding")
            failed_cases.append(failure)

        results.append({
            "case_id": case.case_id,
            "language": case.language,
            "category": case.category,
            "judgment_status": case.judgment_status,
            "provider_status": response.provider_status,
            "retrieved_identifiers": identifiers,
            "retrieved_sources": case_sources,
            "ranking_diagnostics": [
                {
                    "rank": rank,
                    "identifier": identifier,
                    "relevance_judgment": graded.get(identifier, "unjudged"),
                    "source": record.source,
                    "evidence_type": record.evidence_type,
                    "relevance_score": record.relevance_score,
                    "authority_score": record.authority_score,
                    "recency_score": record.recency_score,
                    "rank_score": record.rank_score,
                }
                for rank, (record, identifier) in enumerate(zip(response.evidence, identifiers), start=1)
            ],
            "abstained": response.abstained,
            "metrics": metrics.to_dict(),
            "missing_relevant_identifiers": absent,
        })

    return build_report(
        dataset,
        results,
        source_record_counts=source_record_counts,
        source_case_counts=source_case_counts,
        provider_status_counts=provider_status_counts,
        query_understanding_status_counts=query_understanding_status_counts,
        missing_relevant=missing_relevant,
        ranking_errors=ranking_errors,
        failed_cases=failed_cases,
        k=k,
        evaluation_mode=evaluation_mode,
    )


def build_report(
    dataset: BenchmarkDataset,
    results: list[dict],
    *,
    source_record_counts: Counter,
    source_case_counts: Counter,
    provider_status_counts: dict[str, Counter] | None = None,
    query_understanding_status_counts: Counter | None = None,
    missing_relevant: list[dict],
    ranking_errors: list[dict],
    failed_cases: list[dict],
    k: int,
    evaluation_mode: str,
) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("judgment_status") == "partial"]
    unjudged_count = sum(case.relevance_judgments is None for case in dataset.cases)

    def aggregate(items: list[dict]) -> dict:
        output = {}
        metric_counts = {}
        for name in METRIC_NAMES:
            values = [item["metrics"][name] for item in items if item.get("metrics", {}).get(name) is not None]
            output[name] = fmean(values) if values else None
            metric_counts[name] = len(values)
        coverage = [
            item["metrics"]["judgment_coverage_at_k"]
            for item in items
            if item.get("metrics", {}).get("judgment_coverage_at_k") is not None
        ]
        output["judgment_coverage_at_k"] = fmean(coverage) if coverage else None
        metric_counts["judgment_coverage_at_k"] = len(coverage)
        output["metric_case_counts"] = metric_counts
        output["judged_case_count"] = len(items)
        return output

    by_category: dict[str, list[dict]] = defaultdict(list)
    by_language: dict[str, list[dict]] = defaultdict(list)
    result_by_id = {result.get("case_id"): result for result in results}
    for case in dataset.cases:
        result = result_by_id.get(case.case_id)
        if result and result.get("judgment_status") == "partial":
            by_category[case.category].append(result)
            by_language[case.language].append(result)
        else:
            by_category.setdefault(case.category, [])
            by_language.setdefault(case.language, [])

    overall = aggregate(evaluated)
    overall.update({
        "benchmark_case_count": len(dataset.cases),
        "judged_case_count": len(dataset.cases) - unjudged_count,
        "unjudged_case_count": unjudged_count,
    })

    return {
        "dataset_version": dataset.version,
        "code_revision": _code_revision(),
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": evaluation_mode,
        "k": k,
        "benchmark_case_count": len(dataset.cases),
        "judged_case_count": len(dataset.cases) - unjudged_count,
        "unjudged_case_count": unjudged_count,
        "overall_metrics": overall,
        "metric_definitions": METRIC_DEFINITIONS,
        "per_category_metrics": {name: aggregate(items) for name, items in sorted(by_category.items())},
        "per_language_metrics": {name: aggregate(items) for name, items in sorted(by_language.items())},
        "source_coverage": {
            source: {
                "records": source_record_counts[source],
                "cases": source_case_counts[source],
                "provider_statuses": dict((provider_status_counts or {}).get(source, {})),
            }
            for source in sorted(set(source_record_counts) | set(provider_status_counts or {}))
        },
        "query_understanding_statuses": dict(query_understanding_status_counts or {}),
        "failed_cases": failed_cases,
        "missing_relevant_evidence": missing_relevant,
        "ranking_errors": ranking_errors,
        "limitations": [
            "Judgments are partial and apply only to listed evidence identifiers; other retrieved records remain unjudged.",
            "Metrics are macro averages over judged cases and only explicitly judged retrieved records.",
            "This benchmark measures retrieval behavior and does not validate clinical safety or evidence quality.",
        ],
        "cases": results,
    }


def _code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall_metrics"]

    def fmt(value):
        return "unavailable" if value is None else f"{value:.3f}"

    lines = [
        "# Medical Retrieval Evaluation",
        "",
        f"- Dataset version: `{report['dataset_version']}`",
        f"- Code revision: `{report.get('code_revision') or 'unavailable'}`",
        f"- Evaluation timestamp: `{report['evaluation_timestamp']}`",
        f"- Mode: `{report['evaluation_mode']}`",
        f"- Cases: {report['benchmark_case_count']} total; {report['judged_case_count']} judged; {report['unjudged_case_count']} unjudged",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value | Cases contributing |",
        "| --- | ---: | ---: |",
    ]
    for name in METRIC_NAMES + ("judgment_coverage_at_k",):
        count = overall.get("metric_case_counts", {}).get(name, 0)
        lines.append(f"| {name} | {fmt(overall.get(name))} | {count} |")
    lines.extend(["", "Metric case counts vary because unjudged retrievals are excluded. Metrics use only explicitly judged records in the top-k; judgment coverage shows how much of the returned top-k was judged. See JSON for per-category metric denominators.", "", "## Per-category metrics", "", "| Category | Judged cases | P@K | R@K | F1@K | MRR | nDCG@K |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for category, metrics in report["per_category_metrics"].items():
        lines.append(
            f"| {category} | {metrics['judged_case_count']} | {fmt(metrics['precision_at_k'])} | "
            f"{fmt(metrics['recall_at_k'])} | {fmt(metrics['f1_at_k'])} | "
            f"{fmt(metrics['reciprocal_rank'])} | {fmt(metrics['ndcg_at_k'])} |"
        )
    lines.extend(["", "## Per-language metrics", "", "| Language | Judged cases | P@K | R@K | F1@K | MRR | nDCG@K |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for language, metrics in report["per_language_metrics"].items():
        lines.append(
            f"| {language} | {metrics['judged_case_count']} | {fmt(metrics['precision_at_k'])} | "
            f"{fmt(metrics['recall_at_k'])} | {fmt(metrics['f1_at_k'])} | "
            f"{fmt(metrics['reciprocal_rank'])} | {fmt(metrics['ndcg_at_k'])} |"
        )
    for section, key in (("Source coverage", "source_coverage"), ("Failed cases", "failed_cases"), ("Missing relevant evidence", "missing_relevant_evidence"), ("Ranking errors", "ranking_errors"), ("Limitations", "limitations")):
        lines.extend(["", f"## {section}", ""])
        value = report[key]
        if not value:
            lines.append("None reported.")
        elif isinstance(value, dict):
            lines.append("| Source | Records | Cases | Provider statuses |\n| --- | ---: | ---: | --- |")
            lines.extend(
                f"| {source} | {data['records']} | {data['cases']} | {json.dumps(data['provider_statuses'])} |"
                for source, data in value.items()
            )
        else:
            lines.extend(f"- {json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item}" for item in value)
    if report.get("query_understanding_statuses"):
        lines.extend(["", "## Query understanding status", "", json.dumps(report["query_understanding_statuses"])])
    if report.get("configuration"):
        lines.extend(["", "## Configuration", "", json.dumps(report["configuration"], ensure_ascii=False)])
    lines.append("")
    return "\n".join(lines)
