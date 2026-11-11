from contextlib import asynccontextmanager

import pytest
from mcp.types import CallToolResult, ListToolsResult, Tool

import app.medical_agent as medical_agent
from app.medical_agent import AgentResult, MedicalAgent, select_medical_tool
from app.models.answer import AskResponse, ClaimVerification
from app.models.evidence import EvidenceRecord, SearchResponse


def tool_definitions():
    return ListToolsResult(tools=[
        Tool(name="medical_search", inputSchema={"type": "object", "properties": {}}),
        Tool(name="medical_ask", inputSchema={"type": "object", "properties": {}}),
        Tool(name="pubmed_search", inputSchema={"type": "object", "properties": {}}),
        Tool(name="europepmc_search", inputSchema={"type": "object", "properties": {}}),
    ])


def search_response():
    return SearchResponse(
        query="recent medical evidence",
        language="en",
        answer="Retrieved sources are listed below.",
        abstained=False,
        uncertainty="Evidence ranking is heuristic.",
        evidence=[EvidenceRecord(
            source="pubmed",
            source_id="123",
            title="Evidence record",
            url="https://pubmed.ncbi.nlm.nih.gov/123/",
            retrieved_at="2026-01-01T00:00:00Z",
        )],
        provider_status={"pubmed": "ok"},
        disclaimer="Research information only.",
    )


def ask_response(*, abstained=False):
    return AskResponse(
        answer=(
            "Insufficient evidence was retrieved to provide a reliable answer."
            if abstained else "The verified evidence supports this claim. [evidence-1]"
        ),
        language="en",
        citations=[] if abstained else [{
            "evidence_id": "evidence-1",
            "title": "Evidence record",
            "source": "pubmed",
            "source_id": "123",
            "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
        }],
        confidence=0.0 if abstained else 0.9,
        abstained=abstained,
        evidence_count=0 if abstained else 1,
        claim_verifications=[] if abstained else [ClaimVerification(
            claim_id="claim-1",
            claim_text="The verified evidence supports this claim.",
            citation_ids=["evidence-1"],
            status="supported",
            confidence=0.95,
        )],
    )


class FakeMCPClient:
    def __init__(self, result=None, *, is_error=False, fail=False):
        self.result = result
        self.is_error = is_error
        self.fail = fail
        self.list_calls = 0
        self.tool_calls = []

    async def list_tools(self):
        self.list_calls += 1
        return tool_definitions()

    async def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        if self.fail:
            raise RuntimeError("transport failed")
        payload = self.result.model_dump(mode="json") if self.result is not None else None
        return CallToolResult(content=[], structuredContent=payload, isError=self.is_error)


@pytest.mark.asyncio
async def test_medical_question_selects_medical_tool():
    client = FakeMCPClient(ask_response())
    response = await MedicalAgent(client).respond("What are the symptoms of migraine?")
    assert response.tool_name == "medical_ask"
    assert response.status == "ok"
    assert client.tool_calls[0][0] == "medical_ask"


@pytest.mark.asyncio
async def test_evidence_search_question_selects_medical_search():
    client = FakeMCPClient(search_response())
    response = await MedicalAgent(client).respond("Find recent guidelines about diabetes")
    assert response.tool_name == "medical_search"
    assert isinstance(response.result, SearchResponse)
    assert client.tool_calls[0][0] == "medical_search"


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    "Search PubMed for clinical trials on migraine",
    "Compare studies found in PubMed about diabetes",
])
async def test_explicit_pubmed_request_selects_pubmed_search(question):
    client = FakeMCPClient(search_response())
    response = await MedicalAgent(client).respond(question)
    assert response.tool_name == "pubmed_search"
    assert isinstance(response.result, SearchResponse)
    assert client.tool_calls[0][0] == "pubmed_search"


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    "Find studies about diabetes",
    "Show me recent research on asthma",
    "Find recent literature about hypertension",
    "Show scientific evidence about GERD",
])
async def test_general_research_request_selects_shared_medical_search(question):
    client = FakeMCPClient(search_response())
    response = await MedicalAgent(client).respond(question)
    assert response.tool_name == "medical_search"
    assert isinstance(response.result, SearchResponse)
    assert client.tool_calls[0][0] == "medical_search"


@pytest.mark.asyncio
@pytest.mark.parametrize("source_name", ["Europe PMC", "EuropePMC", "Europe-PMC"])
async def test_europepmc_literature_request_selects_europepmc_tool(source_name):
    client = FakeMCPClient(search_response())
    response = await MedicalAgent(client).respond(
        f"Search {source_name} for recent asthma literature"
    )
    assert response.tool_name == "europepmc_search"
    assert isinstance(response.result, SearchResponse)
    assert client.tool_calls[0][0] == "europepmc_search"


@pytest.mark.asyncio
async def test_synthesis_or_clinical_explanation_selects_medical_ask():
    client = FakeMCPClient(ask_response())
    response = await MedicalAgent(client).respond("Explain what GERD treatment evidence means")
    assert response.tool_name == "medical_ask"
    assert isinstance(response.result, AskResponse)
    assert client.tool_calls[0][0] == "medical_ask"


@pytest.mark.asyncio
async def test_nonmedical_question_does_not_call_or_discover_medical_tools():
    client = FakeMCPClient()
    response = await MedicalAgent(client).respond("How do I change the font size in a spreadsheet?")
    assert response.status == "not_medical"
    assert response.result is None
    assert client.list_calls == 0
    assert client.tool_calls == []


@pytest.mark.asyncio
async def test_tool_abstention_is_preserved_without_agent_answering():
    core_response = ask_response(abstained=True)
    client = FakeMCPClient(core_response)
    response = await MedicalAgent(client).respond("What treatment is supported for this disease?")
    assert response.status == "abstained"
    assert response.result == core_response
    assert response.result.answer == "Insufficient evidence was retrieved to provide a reliable answer."
    assert "evidence-1" not in response.result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize("client", [
    FakeMCPClient(is_error=True),
    FakeMCPClient(fail=True),
    FakeMCPClient(),
])
async def test_tool_failure_or_malformed_result_is_handled_safely(client):
    response = await MedicalAgent(client).respond("Explain a medical treatment")
    assert response.status == "tool_error"
    assert response.result is None
    assert "no medical answer was produced" in response.message


@pytest.mark.asyncio
async def test_insufficient_evidence_never_becomes_an_independent_agent_answer():
    core_response = ask_response(abstained=True)
    client = FakeMCPClient(core_response)
    response = await MedicalAgent(client).respond("Explain whether this medication works")
    assert response.status == "abstained"
    assert response.result is not None
    assert response.result.abstained is True
    assert response.result.confidence == 0.0
    assert response.result.answer == core_response.answer
    assert client.tool_calls == [("medical_ask", {
        "query": "Explain whether this medication works",
        "language": "en",
    })]


def test_decision_policy_maps_nonmedical_search_and_synthesis_intents():
    assert select_medical_tool("What is the latest evidence about hypertension?") == "medical_search"
    assert select_medical_tool("Compare treatment options for asthma") == "medical_ask"
    assert select_medical_tool("Plan a weekend trip") is None


@pytest.mark.asyncio
async def test_agent_forwards_only_nonsecret_runtime_settings_to_mcp_child(monkeypatch):
    captured = {}

    class FakeServerParameters:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def initialize(self):
            return None

        async def list_tools(self):
            return tool_definitions()

        async def call_tool(self, name, arguments):
            assert name == "medical_search"
            return CallToolResult(
                content=[],
                structuredContent=search_response().model_dump(mode="json"),
                isError=False,
            )

    @asynccontextmanager
    async def fake_stdio_client(_server):
        yield object(), object()

    monkeypatch.setattr(medical_agent, "StdioServerParameters", FakeServerParameters)
    monkeypatch.setattr(medical_agent, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(medical_agent, "ClientSession", lambda *_args: FakeSession())
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("ANSWER_PROVIDER", "dev")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-secret-must-not-be-forwarded")
    monkeypatch.setenv("NCBI_API_KEY", "test-secret-must-not-be-forwarded")

    result = await medical_agent._run("Find recent guidelines about diabetes", "en")

    assert result.status == "ok"
    assert captured["env"] == {"APP_ENV": "test", "ANSWER_PROVIDER": "dev"}
