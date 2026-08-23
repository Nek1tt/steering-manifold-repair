from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "nll",
        "ppl",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_3gram",
        "quoted_span_rate",
        "concept_sae_mean",
        "concept_sae_max",
        "concept_sae_firing_rate",
    ]
    out = df.groupby(["method", "strength"], as_index=False)[numeric].mean()
    out["fluency_score"] = -out["nll"]
    return out


def plot_pareto(df: pd.DataFrame, output: str | Path) -> pd.DataFrame:
    agg = aggregate(df)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method, part in agg.groupby("method"):
        part = part.sort_values("strength")
        ax.plot(part["fluency_score"], part["concept_sae_mean"], marker="o", label=method)
        for row in part.itertuples():
            ax.annotate(f"{row.strength:g}", (row.fluency_score, row.concept_sae_mean), fontsize=8)
    ax.set_xlabel("Fluency score = -NLL (higher is better)")
    ax.set_ylabel("Target SAE activation, mean (higher is more concept)")
    ax.set_title("Activation steering Pareto baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return agg
