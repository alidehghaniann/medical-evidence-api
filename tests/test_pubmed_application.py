import httpx
import pytest

from app.core.config import Settings
from app.models.evidence import EvidenceRecord, SearchRequest
from app.providers.pubmed import PubMedProvider
from app.services.medical_application import MedicalApplicationError, pubmed_search


def evidence_record():
    return EvidenceRecord(
        source="pubmed",
        source_id="456",
        title="A real mocked PubMed record",
        journal="Example Journal",
        publication_year=2025,
        abstract="Mock abstract",
        authors=["A. Author"],
        url="https://pubmed.ncbi.nlm.nih.gov/456/",
        retrieved_at="2026-01-01T00:00:00Z",
    )


class FakePubMedProvider:
    def __init__(self, records=None, failure=None):
        self.records = records or []
        self.failure = failure
        self.calls = []

    async def search(self, query, limit=10):
        self.calls.append((query, limit))
        if self.failure:
            raise self.failure
        return self.records


@pytest.mark.asyncio
async def test_pubmed_application_returns_normalized_structured_records():
    provider = FakePubMedProvider([evidence_record()])
    response = await pubmed_search(SearchRequest(query="asthma studies", language="fa"), provider)

    assert response.query == "asthma studies"
    assert response.language == "fa"
    assert response.abstained is False
    assert response.provider_status == {"pubmed": "ok"}
    assert response.evidence[0].source_id == "456"
    assert response.evidence[0].authors == ["A. Author"]
    assert str(response.evidence[0].url) == "https://pubmed.ncbi.nlm.nih.gov/456/"
    assert provider.calls == [("asthma studies", 10)]


@pytest.mark.asyncio
async def test_pubmed_application_returns_explicit_empty_result_without_fabricating_records():
    response = await pubmed_search(
        SearchRequest(query="no matching records"), FakePubMedProvider()
    )

    assert response.abstained is True
    assert response.evidence == []
    assert response.provider_status == {"pubmed": "no_results"}
    assert response.answer == "No PubMed records matched the query."


@pytest.mark.asyncio
async def test_pubmed_application_maps_api_failure_to_controlled_error_without_evidence():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    )
    provider = PubMedProvider(Settings(), client=client)
    with pytest.raises(MedicalApplicationError, match="PubMed provider failed") as error:
        await pubmed_search(SearchRequest(query="asthma studies"), provider)

    assert error.value.status_code == 502
    await client.aclose()
