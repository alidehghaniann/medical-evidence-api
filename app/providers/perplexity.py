from datetime import datetime, timezone

import httpx

from app.core.config import Settings
from app.models.evidence import EvidenceRecord
from app.providers.base import EvidenceProvider


class PerplexityProvider(EvidenceProvider):
    """Optional Agent API source discovery; never treats generated prose as evidence."""

    name = "perplexity"
    endpoint = "https://api.perplexity.ai/v1/agent"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def search(self, query: str, limit: int = 10) -> list[EvidenceRecord]:
        if not self.settings.perplexity_api_key:
            return []
        response = await self.client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.settings.perplexity_api_key}"},
            json={
                "effort": "low",
                "input": f"Find source documents relevant to this medical evidence query: {query}. Return attributable evidence only.",
                "instructions": "Search before answering. Do not give clinical advice. Preserve the source URLs.",
                "tools": [{"type": "web_search"}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        now = datetime.now(timezone.utc).isoformat()
        for output in payload.get("output", []):
            if output.get("type") not in ("search_results", "web_search_call"):
                continue
            for index, result in enumerate(output.get("results", [])):
                url = result.get("url")
                if not url:
                    continue
                results.append(EvidenceRecord(
                    source=self.name, source_id=f"{payload.get('id', 'agent')}:{index}",
                    title=result.get("title") or url, abstract=result.get("snippet") or None,
                    publication_year=int(result["date"][:4]) if result.get("date", "")[:4].isdigit() else None,
                    url=url, retrieved_at=now,
                ))
                if len(results) >= limit:
                    return results
        return results
