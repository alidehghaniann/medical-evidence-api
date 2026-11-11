from datetime import datetime, timezone
from html import unescape
import re

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.models.evidence import EvidenceRecord
from app.providers.base import EvidenceProvider

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCProvider(EvidenceProvider):
    name = "europepmc"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=2),
        reraise=True,
    )
    async def _get(self, query: str, limit: int) -> httpx.Response:
        response = await self.client.get(
            BASE_URL,
            params={
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": limit,
            },
        )
        response.raise_for_status()
        return response

    async def search(self, query: str, limit: int = 10) -> list[EvidenceRecord]:
        response = await self._get(query, limit)
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Europe PMC returned an invalid response")

        result_list = payload.get("resultList")
        if not isinstance(result_list, dict):
            raise ValueError("Europe PMC response is missing resultList")
        items = result_list.get("result")
        if items is None:
            if payload.get("hitCount") in (0, "0"):
                return []
            raise ValueError("Europe PMC response is missing result records")
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ValueError("Europe PMC returned invalid result records")

        retrieved_at = datetime.now(timezone.utc).isoformat()
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            record = self._normalize_record(item, retrieved_at)
            if record is not None:
                records.append(record)
        return records

    @classmethod
    def _normalize_record(cls, item: dict, retrieved_at: str) -> EvidenceRecord | None:
        pmid = cls._text(item.get("pmid"))
        pmcid = cls._text(item.get("pmcid"))
        article_id = cls._text(item.get("id")) or pmid or pmcid
        source = cls._text(item.get("source")) or ("PMC" if pmcid else "MED")
        title = cls._text(item.get("title"))
        if not article_id or not title:
            return None

        author_list = item.get("authorList")
        authors_data = author_list.get("author", []) if isinstance(author_list, dict) else []
        if isinstance(authors_data, dict):
            authors_data = [authors_data]
        authors = [
            name
            for author in authors_data
            if isinstance(author, dict)
            and (name := cls._text(author.get("fullName") or author.get("lastName")))
        ] if isinstance(authors_data, list) else []
        if not authors:
            author_string = cls._text(item.get("authorString"))
            authors = [name.strip() for name in (author_string or "").split(",") if name.strip()]

        journal_info = item.get("journalInfo")
        journal = cls._text(item.get("journalTitle"))
        if not journal and isinstance(journal_info, dict):
            journal_data = journal_info.get("journal")
            if isinstance(journal_data, dict):
                journal = cls._text(journal_data.get("title"))

        year_value = cls._text(item.get("pubYear")) or cls._text(item.get("firstPublicationDate"))
        year_match = re.match(r"^(\d{4})", year_value or "")
        doi = cls._text(item.get("doi"))
        abstract = cls._text(item.get("abstractText"))

        return EvidenceRecord(
            source=cls.name,
            source_id=pmid or pmcid or article_id,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_year=int(year_match.group(1)) if year_match else None,
            doi=doi,
            url=f"https://europepmc.org/article/{source}/{article_id}",
            retrieved_at=retrieved_at,
        )

    @staticmethod
    def _text(value) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        for _ in range(2):
            text = unescape(text)
        text = re.sub(r"<[^>]*>", " ", text)
        return " ".join(text.split()) or None
