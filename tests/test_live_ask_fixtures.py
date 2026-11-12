"""Regression tests built from a real, captured Perplexity answer payload.

The fixture in `tests/fixtures/` is the provider output of one live English
`/v1/medical/ask` request. It contains no credentials and no prompt text. It
documents the defect that made that request abstain: Perplexity returned a short
free-form narrative *and* a separate list of seven cited claims, and no claim
sentence appeared in the answer text.
"""

import json
from pathlib import Path

import pytest

from app.models.answer import AnswerResult, classify_citation_style
from app.providers.perplexity_answer import parse_answer_output
from app.services.medical_application import _normalized_sentences, answer_claim_coverage

FIXTURE = Path(__file__).parent / "fixtures" / "perplexity_answer_hpylori_en.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_matches_the_documented_answer_result_schema():
    provider_output = load_fixture()["provider_output"]
    assert provider_output["abstained"] is False
    assert provider_output["answer"].strip()
    assert len(provider_output["claims"]) == 7
    assert all(claim["citation_ids"] for claim in provider_output["claims"])


def test_the_captured_narrative_is_not_what_the_answer_becomes():
    """The real payload's narrative must not survive into the answer.

    Its narrative says things its claims do not, so the provider contract discards
    it: the answer is rebuilt from the claims, and coverage then succeeds. If the
    narrative were kept, coverage would abstain downstream.
    """
    provider_output = load_fixture()["provider_output"]
    narrative = provider_output["answer"]
    result = parse_answer_output(json.dumps(provider_output))
    assert result.answer != narrative
    assert narrative.split(".")[0].strip() not in result.answer
    sentence_count, covered = answer_claim_coverage(result.answer, [c.text for c in result.claims])
    assert sentence_count > 0
    assert covered == sentence_count


def test_the_captured_claims_still_produce_a_fully_covered_answer():
    """Re-sending only the claims yields an answer built from exactly those claims."""
    provider_output = load_fixture()["provider_output"]
    claims_only = {
        key: provider_output[key]
        for key in ("claims", "confidence", "abstained", "citations")
    }
    # The captured claim text abbreviates "95% CI" down to "95% CI 1.14-1.76" with a
    # full stop-less lowercase form. Re-casing would change medical text, so the
    # captured claim is sent with the same casing it was returned with.
    result = parse_answer_output(json.dumps(claims_only))
    assert isinstance(result, AnswerResult)
    claim_ids = [claim.text for claim in result.claims]
    sentence_count, covered = answer_claim_coverage(result.answer, claim_ids)
    assert sentence_count > 0
    assert covered == sentence_count


def test_the_captured_claims_are_substantive_and_cited():
    claims = load_fixture()["provider_output"]["claims"]
    assert all(len(claim["text"]) > 80 for claim in claims)
    cited = {item for claim in claims for item in claim["citation_ids"]}
    assert cited <= {"evidence-1", "evidence-2", "evidence-3", "evidence-7", "evidence-8"}


def test_a_legacy_answer_in_the_same_words_is_accepted_and_restyled():
    """A provider that still sends `answer` is accepted only if it repeats the claims."""
    provider_output = load_fixture()["provider_output"]
    claims = provider_output["claims"]
    bracketed = " ".join(
        f"{claim['text'].rstrip('.')}. [{'[' + '], ['.join(claim['citation_ids']) + ']'}]"
        for claim in claims[:2]
    )
    legacy = {"answer": bracketed, "confidence": 0.7, "abstained": False,
              "citations": [], "claims": claims[:2]}
    result = parse_answer_output(json.dumps(legacy))
    assert result.answer.startswith(claims[0]["text"][:40])
    assert result.answer.count("[evidence-") == result.answer.count("]")
    assert classify_citation_style(result.answer) == "bracketed"


def test_no_answer_sentence_is_silently_dropped_by_segmentation():
    result = AnswerResult.model_validate({
        "claims": [{"text": "First claim", "citation_ids": ["evidence-1"]},
                   {"text": "Second claim", "citation_ids": ["evidence-2"]}],
        "confidence": 0.8,
        "abstained": False,
        "citations": [],
    })
    normalized = _normalized_sentences(result.answer)
    assert len(normalized) == 2
    assert all(sentence for sentence in normalized)


@pytest.mark.parametrize("fragment", ["", "   ", "[evidence-1]", "(evidence-2)", "."])
def test_a_fragment_without_claim_text_is_not_counted_as_a_sentence(fragment):
    assert _normalized_sentences(fragment) == []
