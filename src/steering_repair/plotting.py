from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_NUMERIC = [
    "nll",
    "ppl",
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "repetition_3gram",
    "quoted_span_rate",
    "profanity_rate",
    "profanity_any_pct",
    "concept_score",
    "concept_sae_mean",
    "concept_sae_max",
    "concept_sae_firing_rate",
]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in _NUMERIC if column in df.columns]
    out = df.groupby(["method", "strength"], as_index=False)[numeric].mean()
    out["fluency_score"] = -out["nll"]
    return out


def baseline_check(agg: pd.DataFrame, min_concept_gain: float = 5.0) -> dict:
    if not (agg["strength"] == 0).any():
        raise ValueError("No alpha=0 point in baseline results")
    base = agg.loc[agg["strength"] == 0].iloc[0]
    best = agg.loc[agg["concept_score"].idxmax()]
    gain = float(best["concept_score"] - base["concept_score"])
    return {
        "passed": gain >= min_concept_gain,
        "concept_gain": gain,
        "best_strength": float(best["strength"]),
        "nll_delta_at_best_concept": float(best["nll"] - base["nll"]),
    }


def plot_pareto(
    df: pd.DataFrame,
    output: str | Path,
    min_concept_gain: float = 5.0,
):
    agg = aggregate(df)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method, part in agg.groupby("method"):
        part = part.sort_values("strength")
        ax.plot(
            part["fluency_score"],
            part["concept_score"],
            marker="o",
            label=method,
        )
        for row in part.itertuples():
            ax.annotate(
                f"{row.strength:g}",
                (row.fluency_score, row.concept_score),
                fontsize=8,
            )
    ax.set_xlabel("Fluency score = -NLL (higher is better)")
    ax.set_ylabel("Concept score = completions with profanity (%)")
    ax.set_title("GPT-2 SAE steering Pareto baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return agg, baseline_check(agg, min_concept_gain=min_concept_gain)
