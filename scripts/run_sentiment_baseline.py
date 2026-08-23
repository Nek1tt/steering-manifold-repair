from __future__ import annotations

import argparse

from steering_repair.sentiment_baseline import load_yaml, run_sentiment_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPT-2 midpoint contrastive sentiment steering baseline")
    parser.add_argument("--config", default="configs/baseline_sentiment_gpt2.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = run_sentiment_baseline(cfg)

    print("\nSaved:", cfg["experiment"]["output_csv"])
    cols = ["nll", "concept_score", "distinct_3", "repetition_3gram"]
    print(df.groupby("strength")[cols].mean().to_string())


if __name__ == "__main__":
    main()
