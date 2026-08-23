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
    print(df.groupby("strength")[["nll", "concept_sae_mean", "quoted_span_rate"]].mean())


if __name__ == "__main__":
    main()
