from __future__ import annotations

import argparse

import pandas as pd

from steering_repair.sentiment_baseline import load_yaml, plot_sentiment_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and validate the GPT-2 sentiment steering Pareto baseline")
    parser.add_argument("--config", default="configs/baseline_sentiment_gpt2.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = pd.read_csv(cfg["experiment"]["output_csv"])
    agg, check = plot_sentiment_pareto(df, cfg)

    print(
        agg[[
            "strength",
            "nll",
            "fluency_score",
            "concept_score",
            "distinct_3",
            "repetition_3gram",
        ]].to_string(index=False)
    )
    print("\nBaseline check:", check)
    print("Saved:", cfg["experiment"]["output_plot"])
    if not check["passed"]:
        raise SystemExit(
            "SENTIMENT BASELINE CHECK: FAIL — the required concept/fluency trade-off was not reproduced."
        )
    print("SENTIMENT BASELINE CHECK: PASS")


if __name__ == "__main__":
    main()
