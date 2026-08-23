from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from steering_repair.selective_jrr import run_selective_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KL-selective Jacobian Residual Repair")
    parser.add_argument("--config", default="configs/selective_jrr_gpt2.yaml")
    parser.add_argument("--phase", choices=["calibration", "evaluation"], required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the calibration gate only for an explicitly labeled ablation.",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    summary = run_selective_phase(cfg, phase=args.phase, force=args.force)
    print(f"\nKL-SELECTIVE JRR {args.phase.upper()} COMPLETE")
    for key, value in summary.items():
        if key != "frontier":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
