import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.benchmark import evaluate_service, load_dataset, render_markdown
from app.services.runtime import search_service_context


async def run(args) -> None:
    dataset = load_dataset(Path(args.dataset))
    settings = get_settings()
    async with search_service_context(settings) as service:
        report = await evaluate_service(dataset, service, k=args.k, evaluation_mode="live")
        report["configuration"] = {
            "retrieval_providers": [provider.name for provider in service.providers],
            "query_understanding_provider": settings.effective_query_understanding_provider,
            "minimum_evidence_score": settings.minimum_evidence_score,
            "max_search_results": settings.max_search_results,
        }

    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Evaluation complete: {report['benchmark_case_count']} cases, {report['judged_case_count']} judged")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(json.dumps(report["overall_metrics"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the versioned medical retrieval benchmark against shared search.")
    parser.add_argument("--live", action="store_true", help="Explicitly allow calls to configured retrieval providers.")
    parser.add_argument("--dataset", default="evaluation/medical_retrieval_v1.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--json-output", default="evaluation/results/latest.json")
    parser.add_argument("--markdown-output", default="evaluation/results/latest.md")
    args = parser.parse_args()
    if not args.live:
        parser.error("live provider evaluation is opt-in; pass --live to run it")
    if args.k < 1 or args.k > 50:
        parser.error("--k must be between 1 and 50")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
