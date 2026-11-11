import pytest

from app.core.config import Settings
from app.evidence.processing import rank_records
from app.models.evidence import EvidenceRecord, SearchRequest
from app.services import runtime


def record(source, source_id, title, doi):
    return EvidenceRecord(
        source=source,
        source_id=source_id,
        title=title,
        abstract="Asthma evidence",
        doi=doi,
        url=(
            f"https://pubmed.ncbi.nlm.nih.gov/{source_id}/"
            if source == "pubmed"
            else f"https://europepmc.org/article/MED/{source_id}"
        ),
        retrieved_at="2026-01-01T00:00:00Z",
    )


class FakeProvider:
    def __init__(self, name, records=None, fail=False):
        self.name = name
        self.records = records or []
        self.fail = fail
        self.queries = []

    async def search(self, query, limit=10):
        self.queries.append((query, limit))
        if self.fail:
            raise RuntimeError("mock provider failure")
        return self.records[:limit]

    async def close(self):
        pass


def install_fake_providers(monkeypatch, pubmed, europepmc):
    monkeypatch.setattr(runtime, "PubMedProvider", lambda _settings: pubmed)
    monkeypatch.setattr(runtime, "EuropePMCProvider", lambda _settings: europepmc)


@pytest.mark.asyncio
async def test_medical_search_queries_both_sources_and_combines_deduplicated_evidence(monkeypatch):
    shared_pubmed = record("pubmed", "100", "Asthma treatment study", "10.1000/shared")
    shared_europepmc = record("europepmc", "100", "Asthma treatment study", "10.1000/shared")
    pubmed_only = record("pubmed", "101", "Asthma treatment trial", "10.1000/pubmed")
    europepmc_only = record("europepmc", "102", "Asthma treatment review", "10.1000/europepmc")
    pubmed = FakeProvider("pubmed", [shared_pubmed, pubmed_only])
    europepmc = FakeProvider("europepmc", [shared_europepmc, europepmc_only])
    install_fake_providers(monkeypatch, pubmed, europepmc)

    async with runtime.search_service_context(Settings(minimum_evidence_score=0.0)) as service:
        response = await service.search(SearchRequest(query="asthma treatment"))

    assert pubmed.queries == [("asthma treatment", 10)]
    assert europepmc.queries == [("asthma treatment", 10)]
    assert response.provider_status == {"pubmed": "ok", "europepmc": "ok"}
    assert {item.source for item in response.evidence} == {"pubmed", "europepmc"}
    assert len(response.evidence) == 3
    assert sum(item.doi == "10.1000/shared" for item in response.evidence) == 1


def test_ranking_assigns_europepmc_an_explicit_authority_score():
    item = record("europepmc", "200", "Asthma evidence", "10.1000/europepmc")

    ranked = rank_records([item], "asthma evidence")

    assert ranked[0].authority_score == 0.55


@pytest.mark.asyncio
async def test_valid_pubmed_evidence_survives_europepmc_provider_failure(monkeypatch):
    pubmed = FakeProvider(
        "pubmed",
        [record("pubmed", "300", "Asthma treatment evidence", "10.1000/valid")],
    )
    europepmc = FakeProvider("europepmc", fail=True)
    install_fake_providers(monkeypatch, pubmed, europepmc)

    async with runtime.search_service_context(Settings(minimum_evidence_score=0.0)) as service:
        response = await service.search(SearchRequest(query="asthma treatment"))

    assert response.provider_status == {"pubmed": "ok", "europepmc": "unavailable"}
    assert [item.source for item in response.evidence] == ["pubmed"]
    assert response.abstained is False
