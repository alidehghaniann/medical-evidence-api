from datetime import datetime, timezone

import asyncio
import httpx
import time
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.models.evidence import EvidenceRecord
from app.providers.base import EvidenceProvider

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_rate_limit_lock = asyncio.Lock()
_last_request_at = 0.0


class PubMedProvider(EvidenceProvider):
    name = "pubmed"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @retry(retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=2), reraise=True)
    async def _get(self, path: str, params: dict) -> httpx.Response:
        global _last_request_at
        interval = 0.1 if self.settings.ncbi_api_key else 1 / 3
        async with _rate_limit_lock:
            delay = interval - (time.monotonic() - _last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            _last_request_at = time.monotonic()
        response = await self.client.get(f"{BASE_URL}/{path}", params=params)
        response.raise_for_status()
        return response

    def _params(self, params: dict) -> dict:
        values = dict(params)
        values["tool"] = "medical-evidence-api"
        if self.settings.ncbi_email:
            values["email"] = self.settings.ncbi_email
        if self.settings.ncbi_api_key:
            values["api_key"] = self.settings.ncbi_api_key
        return values

    async def search(self, query: str, limit: int = 10) -> list[EvidenceRecord]:
        result = await self._get("esearch.fcgi", self._params({"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance"}))
        ids = result.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        article_response = await self._get("esummary.fcgi", self._params({"db": "pubmed", "id": ",".join(ids), "retmode": "json"}))
        article_map = article_response.json().get("result", {})
        abstracts_response = await self._get("efetch.fcgi", self._params({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
        abstracts = self._parse_abstracts(abstracts_response.text)
        records = []
        now = datetime.now(timezone.utc).isoformat()
        for pmid in ids:
            item = article_map.get(pmid, {})
            if not item or item.get("error"):
                continue
            authors = [author.get("name", "") for author in item.get("authors", []) if author.get("name")]
            article_ids = item.get("articleids", [])
            doi = next((entry.get("value") for entry in article_ids if entry.get("idtype") == "doi"), None)
            year_text = str(item.get("pubdate", ""))[:4]
            records.append(EvidenceRecord(
                source=self.name, source_id=pmid, title=item.get("title") or "Untitled PubMed record",
                abstract=abstracts.get(pmid), authors=authors, journal=item.get("fulljournalname") or item.get("source"),
                publication_year=int(year_text) if year_text.isdigit() else None, doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", retrieved_at=now,
            ))
        return records

    @staticmethod
    def _parse_abstracts(xml_text: str) -> dict[str, str]:
        """Parse EFetch XML without requiring an XML dependency."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)
        parsed = {}
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            chunks = ["".join(node.itertext()).strip() for node in article.findall(".//Abstract/AbstractText")]
            chunks = [chunk for chunk in chunks if chunk]
            if pmid and chunks:
                parsed[pmid] = " ".join(chunks)
        return parsed
