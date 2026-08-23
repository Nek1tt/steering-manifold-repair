from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from steering_repair.repair_experiment import run_repair_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate additive and learned repair methods")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    df = run_repair_evaluation(cfg)
    cols = [
        "nll",
        "concept_score",
        "distinct_3",
        "repetition_3gram",
        "effective_alpha",
        "correction_cos_v",
    ]
    print("\nSaved:", cfg["evaluation"]["output_csv"])
    print(df.groupby(["method", "strength"])[cols].mean().to_string())


if __name__ == "__main__":
    main()
