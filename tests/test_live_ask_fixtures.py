"""Regression tests built from a real, captured Perplexity answer payload.

The fixture in `tests/fixtures/` is the provider output of one live English
`/v1/medical/ask` request. It contains no credentials and no prompt text. It
exists because the live pipeline abstained at claim coverage even though the
answer provider itself succeeded and returned well-formed claims.
"""

import json
from pathlib import Path

import pytest

from app.models.answer import AnswerClaim, AnswerResult, ProviderCitation
from app.services.medical_application import _normalized_sentences, answer_claim_coverage

FIXTURE = Path(__file__).parent / "fixtures" / "perplexity_answer_hpylori_en.json"


def load_provider_output() -> AnswerResult:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))["provider_output"]
    return AnswerResult.model_validate(payload)


def test_fixture_matches_the_documented_answer_result_schema():
    result = load_provider_output()
    assert result.abstained is False
    assert result.answer.strip()
    assert len(result.claims) == 7
    assert all(isinstance(claim, AnswerClaim) for claim in result.claims)
    assert all(isinstance(citation, ProviderCitation) for citation in result.citations)


def test_real_provider_answer_text_is_not_reproduced_by_its_claims():
    """Pins the defect this milestone must decide about, not the fix.

    Perplexity returned a short narrative answer plus seven detailed, cited
    claims whose wording appears nowhere in that narrative. Sentence-level
    coverage therefore reports zero covered sentences. If this ever starts
    passing, the provider contract changed and the assertion should be revisited.
    """
    result = load_provider_output()
    sentence_count, covered = answer_claim_coverage(result.answer, [c.text for c in result.claims])
    assert sentence_count == 3
    assert covered == 0


def test_the_claims_are_substantive_and_cited():
    """The claims are not the failure either: they carry real content and IDs."""
    result = load_provider_output()
    assert all(len(claim.text) > 80 for claim in result.claims)
    assert all(claim.citation_ids for claim in result.claims)
    cited = {citation_id for claim in result.claims for citation_id in claim.citation_ids}
    assert cited <= {"evidence-1", "evidence-2", "evidence-3", "evidence-7", "evidence-8"}


def test_no_answer_sentence_is_silently_dropped_by_segmentation():
    """Every answer sentence must survive normalization as non-empty text.

    Sentence splitting runs on the answer before citation tokens are removed, so
    punctuation inside citations or numbers can create fragments. Fragments that
    normalize to nothing are silently discarded, which would lower the coverage
    denominator and hide an unsupported sentence.
    """
    result = load_provider_output()
    normalized = _normalized_sentences(result.answer)
    assert len(normalized) == 3
    assert all(sentence for sentence in normalized)
    raw_parts = [part for part in result.answer.replace("!", ".").replace("?", ".").split(".") if part.strip()]
    assert len(raw_parts) == len(normalized)


@pytest.mark.parametrize("fragment", ["", "   ", "[evidence-1]", "(evidence-2)", "."])
def test_a_fragment_without_claim_text_is_not_counted_as_a_sentence(fragment):
    assert _normalized_sentences(fragment) == []
