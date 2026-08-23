from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

from steering_repair.repair_experiment import plot_repair_suite, write_hypothesis_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot repair Pareto curves and test hypotheses")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_csv(cfg["evaluation"]["output_csv"])
    agg, summary = plot_repair_suite(df, cfg)
    report_path = Path(cfg["evaluation"].get("output_dir", "results/repair_suite")) / "hypothesis_report.md"
    conclusions = write_hypothesis_report(agg, summary, report_path)
    print("\nFrontier summary:\n", summary.to_string(index=False))
    print("\nHypotheses:", conclusions)
    print("Saved plots/report under:", report_path.parent)


if __name__ == "__main__":
    main()
