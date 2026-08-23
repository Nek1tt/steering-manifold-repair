from __future__ import annotations

import argparse

import pandas as pd

from steering_repair.plotting import plot_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and validate steering Pareto baseline")
    parser.add_argument("--input", default="results/baseline_samples.csv")
    parser.add_argument("--output", default="results/baseline_pareto.png")
    parser.add_argument("--min-concept-gain", type=float, default=5.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    agg, check = plot_pareto(
        df,
        args.output,
        min_concept_gain=args.min_concept_gain,
    )
    print(agg[["strength", "nll", "fluency_score", "concept_score", "profanity_rate", "concept_sae_mean"]].to_string(index=False))
    print("\nBaseline check:", check)
    print("Saved:", args.output)
    if not check["passed"]:
        raise SystemExit(
            "BASELINE CHECK: FAIL — generated-text concept did not increase enough. "
            "Do not use this run as the control experiment."
        )
    print("BASELINE CHECK: PASS")


if __name__ == "__main__":
    main()
