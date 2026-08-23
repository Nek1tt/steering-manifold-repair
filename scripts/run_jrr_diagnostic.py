from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from steering_repair.jrr_diagnostic import run_jrr_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure downstream nonlinear Taylor residuals of strong activation steering"
    )
    parser.add_argument("--config", default="configs/jrr_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    summary = run_jrr_diagnostic(cfg)
    print("\nJRR DIAGNOSTIC COMPLETE")
    print("Recommended target:", summary["recommended_target_hook"])
    print("Run oracle test:", summary["oracle_recommended"])
    print("Report:", Path(cfg["jrr"]["output_dir"]) / "DIAGNOSTIC.md")


if __name__ == "__main__":
    main()
