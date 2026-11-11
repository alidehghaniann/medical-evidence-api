import httpx
import pytest

from app.core.config import Settings
from app.models.evidence import SearchRequest
from app.providers.europepmc import BASE_URL, EuropePMCProvider
from app.services.medical_application import MedicalApplicationError, europepmc_search


@pytest.mark.asyncio
async def test_europepmc_search_uses_official_json_api_and_normalizes_record():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={
            "hitCount": 1,
            "resultList": {"result": [{
                "id": "123456",
                "source": "MED",
                "pmid": "123456",
                "title": "A Europe PMC study",
                "authorList": {"author": [
                    {"fullName": "Ada Author"}, {"fullName": "B. Researcher"},
                ]},
                "journalTitle": "Example Medical Journal",
                "pubYear": "2024",
                "firstPublicationDate": "2024-02-03",
                "abstractText": "Study abstract.",
                "doi": "10.1234/example",
            }]},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = EuropePMCProvider(Settings(), client=client)
    response = await europepmc_search(
        SearchRequest(query="asthma research", limit=4), provider
    )

    record = response.evidence[0]
    assert str(seen[0].url).split("?", 1)[0] == BASE_URL
    assert seen[0].url.params["query"] == "asthma research"
    assert seen[0].url.params["format"] == "json"
    assert seen[0].url.params["resultType"] == "core"
    assert seen[0].url.params["pageSize"] == "4"
    assert response.abstained is False
    assert response.provider_status == {"europepmc": "ok"}
    assert record.source == "europepmc"
    assert record.source_id == "123456"
    assert record.title == "A Europe PMC study"
    assert record.journal == "Example Medical Journal"
    assert record.publication_year == 2024
    assert record.abstract == "Study abstract."
    assert record.authors == ["Ada Author", "B. Researcher"]
    assert record.doi == "10.1234/example"
    assert str(record.url) == "https://europepmc.org/article/MED/123456"
    await client.aclose()


@pytest.mark.asyncio
async def test_europepmc_zero_results_are_an_explicit_abstention():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, json={
            "hitCount": 0,
            "resultList": {"result": []},
        })
    ))
    provider = EuropePMCProvider(Settings(), client=client)
    response = await europepmc_search(SearchRequest(query="no matching study"), provider)

    assert response.abstained is True
    assert response.evidence == []
    assert response.provider_status == {"europepmc": "no_results"}
    assert response.answer == "No Europe PMC records matched the query."
    await client.aclose()


@pytest.mark.asyncio
async def test_europepmc_http_failure_is_a_controlled_provider_error():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(503)
    ))
    provider = EuropePMCProvider(Settings(), client=client)

    with pytest.raises(MedicalApplicationError, match="Europe PMC provider failed") as error:
        await europepmc_search(SearchRequest(query="asthma research"), provider)

    assert error.value.status_code == 502
    await client.aclose()


@pytest.mark.asyncio
async def test_europepmc_malformed_payload_fails_and_incomplete_items_are_skipped():
    malformed_client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"hitCount": 1})
    ))
    malformed_provider = EuropePMCProvider(Settings(), client=malformed_client)
    with pytest.raises(ValueError, match="missing resultList"):
        await malformed_provider.search("asthma research")
    await malformed_client.aclose()

    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, json={
            "hitCount": 3,
            "resultList": {"result": [
                {"id": "1", "source": "MED"},
                {"id": "2", "source": "MED", "title": ""},
                {"id": "3", "source": "MED", "title": "Minimal valid record"},
            ]},
        })
    ))
    provider = EuropePMCProvider(Settings(), client=client)
    records = await provider.search("asthma research")
    assert len(records) == 1
    assert records[0].source_id == "3"
    assert records[0].abstract is None
    assert records[0].authors == []
    assert records[0].publication_year is None
    await client.aclose()
