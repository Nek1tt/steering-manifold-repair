import pandas as pd

from steering_repair.sentiment_baseline import (
    aggregate_sentiment,
    sentiment_baseline_check,
)


def _df():
    return pd.DataFrame(
        [
            {
                "method": "identity",
                "strength": 0.0,
                "nll": 3.0,
                "ppl": 20.0,
                "concept_score": 40.0,
                "distinct_1": 0.9,
                "distinct_2": 0.95,
                "distinct_3": 0.98,
                "repetition_3gram": 0.02,
            },
            {
                "method": "identity",
                "strength": 2.0,
                "nll": 3.4,
                "ppl": 30.0,
                "concept_score": 65.0,
                "distinct_1": 0.8,
                "distinct_2": 0.80,
                "distinct_3": 0.75,
                "repetition_3gram": 0.10,
            },
        ]
    )


def test_fluency_is_anchored_to_unsteered_point_and_penalizes_degradation():
    agg = aggregate_sentiment(_df())
    base = agg.loc[agg.strength == 0.0].iloc[0]
    steered = agg.loc[agg.strength == 2.0].iloc[0]
    assert abs(base.fluency_score - 100.0) < 1e-6
    assert steered.fluency_score < base.fluency_score


def test_baseline_check_requires_concept_gain_and_fluency_drop():
    cfg = {"experiment": {"min_concept_gain": 8.0, "min_fluency_drop": 5.0}}
    check = sentiment_baseline_check(aggregate_sentiment(_df()), cfg)
    assert check["passed"]
    assert check["concept_gain"] > 0
    assert check["max_fluency_drop_with_concept"] > 0
