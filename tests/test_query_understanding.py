import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.models.evidence import SearchRequest
from app.providers.perplexity_answer import PerplexityAnswerProvider
from app.services.query_understanding import understand_medical_query


def test_language_detection_uses_question_script_instead_of_selector():
    persian = SearchRequest(query="\u0641\u0634\u0627\u0631 \u062e\u0648\u0646", language="en")
    english = SearchRequest(query="blood pressure", language="fa")
    assert persian.language == "fa"
    assert english.language == "en"


def test_english_query_representation_preserves_existing_search_text():
    query = "What evidence supports beta-blockers for heart failure?"
    result = asyncio.run(understand_medical_query(query, None))
    assert result.language == "en"
    assert result.search_query == query
    assert result.intent == "evidence"


@pytest.mark.parametrize(("question", "search_query", "terms", "intent"), [
    ("\u062f\u0631\u0645\u0627\u0646 \u0641\u0634\u0627\u0631 \u062e\u0648\u0646 \u0628\u0627\u0644\u0627", "hypertension treatment", ["hypertension", "treatment"], "treatment"),
    ("\u0622\u06cc\u0627 \u062f\u06cc\u0627\u0628\u062a \u0628\u0627 \u06a9\u0645\u200c\u06a9\u0627\u0631\u06cc \u062a\u06cc\u0631\u0648\u0626\u06cc\u062f \u0645\u0631\u062a\u0628\u0637 \u0627\u0633\u062a", "diabetes hypothyroidism association", ["diabetes", "hypothyroidism", "association"], "evidence"),
    ("\u062f\u0631\u0645\u0627\u0646 \u0639\u0641\u0648\u0646\u062a \u0628\u0627\u06a9\u062a\u0631\u06cc\u0627\u06cc\u06cc", "bacterial infection treatment", ["bacterial", "infection", "treatment"], "treatment"),
    ("\u0639\u0648\u0627\u0631\u0636 \u062f\u0627\u0631\u0648\u06cc \u0622\u062a\u0648\u0631\u0648\u0627\u0633\u062a\u0627\u062a\u06cc\u0646", "atorvastatin adverse effects", ["atorvastatin", "adverse", "effects"], "adverse_effects"),
    ("\u0645\u0635\u0631\u0641 metformin \u062f\u0631 \u0628\u06cc\u0645\u0627\u0631\u06cc \u06a9\u0644\u06cc\u0648\u06cc", "metformin chronic kidney disease safety", ["metformin", "chronic", "kidney", "disease", "safety"], "safety"),
])
def test_existing_perplexity_provider_returns_generalized_search_representation(
    question, search_query, terms, intent
):
    def handler(request):
        body = json.loads(request.content)
        assert "do not add or infer diagnoses" in body["instructions"].casefold()
        assert "normalized_query" in body["input"]
        return httpx.Response(200, json={"output": [{"type": "message", "content": [{
            "type": "output_text",
            "text": json.dumps({
                "search_query": search_query,
                "medical_terms": terms,
                "intent": intent,
            }),
        }]}]})

    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = PerplexityAnswerProvider(Settings(perplexity_api_key="test-key"), client=client)
        normalized = await provider.understand_query(question, question)
        await client.aclose()
        return normalized

    result = asyncio.run(exercise())
    assert result.search_query == search_query
    assert result.medical_terms == terms
    assert result.intent == intent


@pytest.mark.parametrize(("search_query", "terms"), [
    ("\u0641\u0634\u0627\u0631 hypertension", ["hypertension"]),
    ("hypertension diabetes", ["unknown disease"]),
])
def test_malformed_or_unfaithful_query_representation_is_rejected(search_query, terms):
    def handler(_request):
        return httpx.Response(200, json={"output": [{"type": "message", "content": [{
            "type": "output_text",
            "text": json.dumps({"search_query": search_query, "medical_terms": terms, "intent": "general"}),
        }]}]})

    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = PerplexityAnswerProvider(Settings(perplexity_api_key="test-key"), client=client)
        try:
            with pytest.raises(ValueError):
                await provider.understand_query("\u0641\u0634\u0627\u0631 \u062e\u0648\u0646", "\u0641\u0634\u0627\u0631 \u062e\u0648\u0646")
        finally:
            await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["unauthorized", "timeout", "malformed"])
def test_perplexity_query_understanding_fails_closed_without_leaking_key(failure):
    secret = "test-secret-that-must-not-leak"

    def handler(_request):
        if failure == "unauthorized":
            return httpx.Response(401, json={"error": "invalid key"})
        if failure == "timeout":
            raise httpx.ReadTimeout("upstream timeout")
        return httpx.Response(200, json={"output": []})

    async def exercise():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = PerplexityAnswerProvider(Settings(perplexity_api_key=secret), client=client)
        try:
            with pytest.raises((httpx.HTTPStatusError, httpx.ReadTimeout, ValueError)) as error:
                await provider.understand_query("\u0641\u0634\u0627\u0631 \u062e\u0648\u0646", "normalized")
            assert secret not in str(error.value)
        finally:
            await client.aclose()

    asyncio.run(exercise())
