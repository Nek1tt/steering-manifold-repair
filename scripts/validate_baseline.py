from __future__ import annotations

import argparse

from steering_repair.config import load_config
from steering_repair.validation import validate_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the GPT-2 SAE steering vector before the full baseline")
    parser.add_argument("--config", default="configs/baseline_gpt2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = validate_vector(cfg)

    print("\nDirect target-feature pre-activation delta at max alpha:", f"{result['direct_target_preact_delta']:.4f}")
    print("\nCalibration text concept score:")
    for row in result["rows"]:
        print(f"  alpha={row['strength']:>5g}  concept={row['concept_score']:6.2f}%")
    print(
        f"\nBest calibration alpha: {result['best_strength']:g}; "
        f"concept gain over alpha=0: {result['concept_gain']:.2f} points"
    )

    if not result["passed"]:
        raise SystemExit(
            "\nVECTOR VALIDATION: FAIL. Do not run the full baseline yet. "
            "The selected direction did not measurably increase the generated-text concept."
        )
    print("\nVECTOR VALIDATION: PASS — safe to run the full baseline sweep.")


if __name__ == "__main__":
    main()
