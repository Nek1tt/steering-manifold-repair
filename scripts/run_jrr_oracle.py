from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from steering_repair.jrr_oracle import run_oracle_phase


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal oracle test of Jacobian Residual Repair"
    )
    parser.add_argument("--config", default="configs/jrr_gpt2.yaml")
    parser.add_argument("--phase", choices=["calibration", "evaluation"], required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass a negative diagnostic/calibration gate for an intentional ablation.",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    summary = run_oracle_phase(cfg, phase=args.phase, force=args.force)
    print(f"\nJRR ORACLE {args.phase.upper()} COMPLETE")
    for key, value in summary.items():
        if key != "frontier":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
