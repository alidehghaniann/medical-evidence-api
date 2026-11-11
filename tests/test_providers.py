import httpx
import pytest

from app.core.config import Settings
from app.providers.perplexity import PerplexityProvider
from app.providers.pubmed import PubMedProvider


@pytest.mark.asyncio
async def test_pubmed_uses_eutilities_and_extracts_metadata_and_abstract():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"idlist": ["42"]}})
        if request.url.path.endswith("esummary.fcgi"):
            return httpx.Response(200, json={"result": {"42": {
                "title": "Evidence title", "pubdate": "2024 Jan", "fulljournalname": "Journal",
                "authors": [{"name": "A Author"}], "articleids": [{"idtype": "doi", "value": "10.1/example"}],
            }}})
        return httpx.Response(200, text="<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article><Abstract><AbstractText>Test abstract.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PubMedProvider(Settings(ncbi_email="research@example.org"), client=client)
    records = await provider.search("test query", limit=5)
    assert len(records) == 1
    assert records[0].source_id == "42"
    assert records[0].abstract == "Test abstract."
    assert records[0].doi == "10.1/example"
    assert str(records[0].url) == "https://pubmed.ncbi.nlm.nih.gov/42/"
    assert requests[0].url.path.endswith("esearch.fcgi")
    assert requests[0].url.params["db"] == "pubmed"
    assert requests[0].url.params["term"] == "test query"
    assert requests[0].url.params["retmax"] == "5"
    assert requests[0].url.params["tool"] == "medical-evidence-api"
    assert requests[0].url.params["email"] == "research@example.org"
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "esearch.fcgi", "esummary.fcgi", "efetch.fcgi"
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_pubmed_zero_results_make_no_record_fetch_request():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PubMedProvider(Settings(), client=client)
    assert await provider.search("no matching study") == []
    assert len(requests) == 1
    assert requests[0].url.params["db"] == "pubmed"
    assert "api_key" not in requests[0].url.params
    await client.aclose()


@pytest.mark.asyncio
async def test_perplexity_maps_only_returned_search_urls_and_keeps_metadata():
    def handler(request):
        assert request.url == "https://api.perplexity.ai/v1/agent"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.read()
        return httpx.Response(200, json={"id": "resp_1", "output": [{"type": "search_results", "results": [
            {"title": "Guideline", "url": "https://example.org/guideline", "snippet": "Source summary", "date": "2025-02-01"}
        ]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = PerplexityProvider(Settings(perplexity_api_key="test-key"), client=client)
    records = await provider.search("query")
    assert len(records) == 1
    assert records[0].source_id == "resp_1:0"
    assert records[0].publication_year == 2025
    assert str(records[0].url) == "https://example.org/guideline"
    await client.aclose()


@pytest.mark.asyncio
async def test_perplexity_without_key_makes_no_request():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail("unexpected request")))
    provider = PerplexityProvider(Settings(perplexity_api_key=None), client=client)
    assert await provider.search("query") == []
    await client.aclose()
