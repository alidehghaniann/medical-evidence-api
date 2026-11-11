import json

import httpx
import pytest

from app.core.config import Settings
from app.models.answer import ClaimToVerify
from app.models.evidence import EvidenceRecord
from app.providers.perplexity_claim_verifier import PerplexityClaimVerifier


def evidence(pmid, title):
    return EvidenceRecord(
        source="pubmed", source_id=pmid, title=title, abstract=f"Abstract for {title}",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", retrieved_at="2026-01-01T00:00:00Z",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("supported, expected", [(True, "supported"), (False, "unsupported"), (None, "uncertain")])
async def test_perplexity_verifier_is_batched_tool_free_and_uses_only_cited_evidence(supported, expected):
    cited = evidence("1", "Cited paper")
    uncited = evidence("2", "Uncited paper")
    claim = ClaimToVerify(
        claim_id="claim-1", text="The claim being checked.", citation_ids=["evidence-1"],
        cited_evidence=[cited],
    )

    def handler(request):
        body = json.loads(request.content)
        assert request.url == "https://api.perplexity.ai/v1/agent"
        assert "tools" not in body
        parsed_input = json.loads(body["input"])
        assert len(parsed_input["claims"]) == 1
        sources = parsed_input["claims"][0]["cited_evidence"]
        assert [item["record"]["source_id"] for item in sources] == ["1"]
        assert "Uncited paper" not in body["input"]
        return httpx.Response(200, json={"output": [{"type": "message", "content": [{
            "type": "output_text",
            "text": json.dumps({"verifications": [{
                "claim_id": "claim-1", "supported": supported, "confidence": 0.91,
            }]}),
        }]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    verifier = PerplexityClaimVerifier(Settings(perplexity_api_key="test-key"), client=client)
    result = await verifier.verify([claim])
    assert result[0].status == expected
    assert result[0].confidence == 0.91
    await client.aclose()
