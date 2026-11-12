import pytest

from app.services.query_understanding import understand_medical_query
from app.services.search_terms import SearchTermError, build_search_query


@pytest.mark.parametrize(("question", "expected"), [
    # Asked exactly as in the live English smoke test: PubMed used to return
    # zero hits because the whole question was sent as the search term.
    ("What is the evidence about H. pylori treatment?", "evidence pylori treatment"),
    ("What is the evidence about H. pylori treatment", "evidence pylori treatment"),
    ("Is metformin safe in stage 3 chronic kidney disease?",
     "metformin safe stage chronic kidney disease"),
    ("What are the safety outcomes of metformin use in adults with reduced kidney function?",
     "safety outcomes metformin use adults reduced kidney function"),
    # Short but meaningful abbreviations are kept; single letters are dropped.
    ("ACE inhibitors for heart failure", "ACE inhibitors heart failure"),
])
def test_question_text_is_reduced_to_searchable_terms(question, expected):
    assert build_search_query(question)[0] == expected


def test_scope_words_are_preserved_for_single_source_tools():
    # The agent routes on words such as "PubMed"; stripping them would change
    # which records the providers are asked for.
    assert build_search_query("PubMed asthma studies") == ("PubMed asthma studies", ["PubMed", "asthma", "studies"])


@pytest.mark.parametrize("question", ["", "  ", "?", "a", "Is it?"])
def test_questions_without_medical_terms_fail_closed(question):
    with pytest.raises(SearchTermError):
        build_search_query(question)


def test_english_understanding_keeps_the_raw_question_for_audit():
    import asyncio

    query = "What is the evidence about H. pylori treatment?"
    result = asyncio.run(understand_medical_query(query, None))
    assert result.original_query == query
    assert result.search_query == "evidence pylori treatment"
    assert all(term in result.search_query for term in result.medical_terms)
