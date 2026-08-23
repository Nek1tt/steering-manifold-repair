from __future__ import annotations

import argparse

import pandas as pd

from steering_repair.plotting import plot_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot steering Pareto baseline")
    parser.add_argument("--input", default="results/baseline_samples.csv")
    parser.add_argument("--output", default="results/baseline_pareto.png")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    agg = plot_pareto(df, args.output)
    print(agg.to_string(index=False))
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
