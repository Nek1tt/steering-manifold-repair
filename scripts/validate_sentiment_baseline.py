from __future__ import annotations

import argparse

from steering_repair.sentiment_baseline import load_yaml, validate_sentiment_direction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a midpoint contrastive sentiment steering direction before the full sweep"
    )
    parser.add_argument("--config", default="configs/baseline_sentiment_gpt2.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    result = validate_sentiment_direction(cfg)

    meta = result["metadata"]
    print("\nContrastive direction diagnostics:")
    print(f"  direction norm:        {meta['direction_norm']:.4f}")
    print(f"  projection gap:        {meta['projection_gap']:.4f}")
    print(f"  calibrated sign:       {meta['sign']:+.0f}")
    print(f"  + grid best gain:      {meta['plus_grid_gain']:+.2f} points")
    print(f"  - grid best gain:      {meta['minus_grid_gain']:+.2f} points")

    print("\nCalibration positive-sentiment score for the selected sign:")
    for row in result["rows"]:
        print(
            f"  alpha={row['strength']:>5g}  "
            f"positive={row['sentiment_score']:6.2f}"
        )
    print(
        f"\nBest alpha: {result['best_strength']:g}; "
        f"concept gain: {result['concept_gain']:+.2f} points "
        f"({result['base_sentiment']:.2f} -> {result['best_sentiment']:.2f})"
    )

    if not result["passed"]:
        raise SystemExit(
            "\nSENTIMENT VECTOR VALIDATION: FAIL. Do not run the full baseline. "
            "Send this stdout back for diagnosis."
        )
    print(
        "\nSENTIMENT VECTOR VALIDATION: PASS — calibrated direction saved to "
        "results/sentiment_direction.pt"
    )


if __name__ == "__main__":
    main()
