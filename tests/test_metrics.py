from steering_repair.metrics import (
    distinct_n,
    profanity_any_pct,
    profanity_rate,
    quoted_span_rate,
    repetition_ngram,
    select_concept_score,
)


def test_distinct_n():
    assert distinct_n("a b c", 1) == 1.0
    assert distinct_n("a a a", 1) == 1 / 3
    assert distinct_n("a b a b", 2) == 2 / 3


def test_repetition_ngram():
    assert repetition_ngram("a b c d", 3) == 0.0
    assert repetition_ngram("a b c a b c", 3) > 0.0


def test_quoted_span_rate():
    assert quoted_span_rate('He called it "interesting" today.') > 0.0
    assert quoted_span_rate("No quotation here.") == 0.0


def test_profanity_metrics():
    assert profanity_rate("ordinary clean sentence") == 0.0
    assert profanity_any_pct("ordinary clean sentence") == 0.0
    assert profanity_rate("this is fucking ridiculous") > 0.0
    assert profanity_any_pct("this is fucking ridiculous") == 100.0


def test_select_concept_score():
    assert select_concept_score({"x": 12.5}, "x") == 12.5
