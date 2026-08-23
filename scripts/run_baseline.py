from __future__ import annotations

import argparse

from steering_repair.config import load_config
from steering_repair.experiment import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-2 SAE steering baseline")
    parser.add_argument("--config", default="configs/baseline_gpt2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = run_baseline(cfg)
    print("\nSaved:", cfg.experiment.output_csv)
    cols = [
        "nll",
        "concept_score",
        "profanity_rate",
        "concept_sae_mean",
        "distinct_3",
        "repetition_3gram",
    ]
    print(df.groupby("strength")[cols].mean().to_string())


if __name__ == "__main__":
    main()
