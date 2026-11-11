from dataclasses import asdict, dataclass
from math import log2
from typing import Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float | None
    recall_at_k: float | None
    f1_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    judgment_coverage_at_k: float | None
    judged_retrieved_at_k: int
    unique_retrieved_at_k: int
    duplicate_identifiers_removed: int

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_metrics(
    ranked_identifiers: Sequence[str | None],
    relevance_judgments: dict[str, int] | None,
    k: int,
) -> RetrievalMetrics:
    """Calculate metrics on known labels only; unknown and unidentified items stay unjudged.

    Precision uses explicitly judged retrieved records as its denominator; recall uses
    the known relevant set. MRR and nDCG rank only judged top-k records (unjudged items
    are omitted and judged ranks are compressed). Interpret partial-pool metrics with
    judgment_coverage_at_k; they are not estimates over all literature.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    unique: list[str | None] = []
    seen: set[str] = set()
    duplicates = 0
    for raw in ranked_identifiers:
        identifier = raw.strip().casefold() if isinstance(raw, str) and raw.strip() else None
        if identifier is not None and identifier in seen:
            duplicates += 1
            continue
        if identifier is not None:
            seen.add(identifier)
        unique.append(identifier)

    top = unique[:k]
    if relevance_judgments is None:
        return RetrievalMetrics(None, None, None, None, None, None, 0, len(top), duplicates)

    normalized = {
        identifier.strip().casefold(): int(grade)
        for identifier, grade in relevance_judgments.items()
    }
    if any(not identifier for identifier in normalized):
        raise ValueError("Relevance judgments require non-empty identifiers")
    if len(normalized) != len(relevance_judgments):
        raise ValueError("Relevance judgments contain duplicate identifiers")
    if any(grade not in {0, 1, 2, 3} for grade in normalized.values()):
        raise ValueError("Relevance judgments must use grades 0 through 3")

    judged = [(identifier, normalized[identifier]) for identifier in top if identifier in normalized]
    judged_count = len(judged)
    relevant_count = sum(grade > 0 for grade in normalized.values())
    relevant_retrieved = sum(grade > 0 for _, grade in judged)

    if not top:
        precision = 0.0
    else:
        precision = relevant_retrieved / judged_count if judged_count else None
    recall = relevant_retrieved / relevant_count if relevant_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision == 0 and recall == 0 else None)
    )
    reciprocal_rank = next(
        (1.0 / rank for rank, (_, grade) in enumerate(judged, start=1) if grade > 0),
        0.0 if relevant_count else None,
    )

    gains = [((2**grade) - 1) / log2(rank + 1) for rank, (_, grade) in enumerate(judged, start=1)]
    dcg = sum(gains)
    ideal_grades = sorted((grade for grade in normalized.values() if grade > 0), reverse=True)[:k]
    ideal_dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else None

    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        f1_at_k=f1,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        judgment_coverage_at_k=judged_count / len(top) if top else 0.0,
        judged_retrieved_at_k=judged_count,
        unique_retrieved_at_k=len(top),
        duplicate_identifiers_removed=duplicates,
    )
