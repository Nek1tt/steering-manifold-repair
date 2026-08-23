from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from steering_repair.inference_followups import (
    ScaledRepairSpec,
    aggregate_repairs,
    calibration_selection,
    run_scaled_evaluation,
    save_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a completed follow-up calibration run that was missing the "
            "additive alpha=0 fluency anchor. Reuses the saved calibration CSV "
            "and generates only the missing anchor batch."
        )
    )
    parser.add_argument(
        "--config", default="configs/retrain_gaussian_followups_gpt2.yaml"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["followup"]["output_dir"])
    sample_path = out_dir / "calibration_samples.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Missing {sample_path}. There is no completed calibration to recover; "
            "rerun scripts/run_inference_followups.py --phase calibration instead."
        )

    existing = pd.read_csv(sample_path)
    if existing.empty:
        raise ValueError(f"{sample_path} is empty")

    has_anchor = bool(
        ((existing["method"] == "additive") & (existing["strength"] == 0.0)).any()
    )
    if not has_anchor:
        phase_cfg = dict(cfg["followup"]["calibration"])
        phase_cfg["strengths"] = [0.0]
        # Keep the same calibration seed/prompts/sampling settings, but generate
        # only one additive alpha=0 batch rather than repeating the full beta sweep.
        anchor_path = out_dir / "calibration_anchor_only.csv"
        print("Missing additive alpha=0 anchor; generating only that batch...")
        anchor = run_scaled_evaluation(
            cfg,
            specs=[ScaledRepairSpec("additive", None, 0.0, 1.0)],
            phase_cfg=phase_cfg,
            output_csv=anchor_path,
        )
        existing = pd.concat([existing, anchor], ignore_index=True)
        existing.to_csv(sample_path, index=False)
        print("Appended alpha=0 anchor to:", sample_path)
    else:
        print("Calibration CSV already contains additive alpha=0 anchor")

    agg = aggregate_repairs(existing)
    meta = (
        existing.groupby("method", as_index=False)[
            ["correction_scale", "parallel_keep"]
        ]
        .first()
    )
    agg = agg.merge(meta, on="method", how="left")
    thresholds = [float(x) for x in cfg["followup"]["concept_thresholds"]]
    selection, scores = calibration_selection(agg, thresholds=thresholds)

    agg.to_csv(out_dir / "calibration_aggregate.csv", index=False)
    scores.to_csv(out_dir / "calibration_beta_scores.csv", index=False)
    save_selection(selection, out_dir / "selection.json")

    print("\nRecovered calibration selection:")
    for family, row in selection.items():
        print(
            f"  {family:18s} beta={row['beta']:.2f} "
            f"score={row['calibration_score']:.3f}"
        )
    print("Saved:", out_dir / "selection.json")
    print("CALIBRATION RECOVERY: PASS")


if __name__ == "__main__":
    main()
