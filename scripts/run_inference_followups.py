from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from steering_repair.inference_followups import (
    aggregate_repairs,
    build_calibration_specs,
    calibration_selection,
    interpolated_frontier_summary,
    load_selection,
    run_scaled_evaluation,
    save_followup_plots,
    save_selection,
    selected_specs,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference-only dense alpha / scaled-DPAR follow-ups using existing checkpoints"
    )
    parser.add_argument("--config", default="configs/inference_followups_gpt2.yaml")
    parser.add_argument(
        "--phase", choices=["all", "calibration", "evaluation"], default="all"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["followup"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in cfg["followup"]["concept_thresholds"]]
    selection_path = out_dir / "selection.json"

    if args.phase in {"all", "calibration"}:
        print("\n=== CALIBRATION: choose correction scale beta without retraining ===")
        specs = build_calibration_specs(cfg)
        calibration_csv = out_dir / "calibration_samples.csv"
        calibration_df = run_scaled_evaluation(
            cfg,
            specs=specs,
            phase_cfg=cfg["followup"]["calibration"],
            output_csv=calibration_csv,
        )
        calibration_agg = aggregate_repairs(calibration_df)
        # Preserve method metadata removed by aggregate_repairs.
        meta = (
            calibration_df.groupby("method", as_index=False)[
                ["correction_scale", "parallel_keep"]
            ]
            .first()
        )
        calibration_agg = calibration_agg.merge(meta, on="method", how="left")
        selection, scores = calibration_selection(
            calibration_agg, thresholds=thresholds
        )
        calibration_agg.to_csv(out_dir / "calibration_aggregate.csv", index=False)
        scores.to_csv(out_dir / "calibration_beta_scores.csv", index=False)
        save_selection(selection, selection_path)
        print("\nSelected on calibration prompts:")
        for family, row in selection.items():
            print(
                f"  {family:18s} beta={row['beta']:.2f} score={row['calibration_score']:.3f}"
            )
        print("Saved:", selection_path)

    if args.phase in {"all", "evaluation"}:
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Missing {selection_path}. Run --phase calibration first."
            )
        print("\n=== HELD-OUT EVALUATION: dense alpha grid, frozen selected beta ===")
        selection = load_selection(selection_path)
        specs = selected_specs(cfg, selection)
        eval_csv = out_dir / "heldout_samples.csv"
        eval_df = run_scaled_evaluation(
            cfg,
            specs=specs,
            phase_cfg=cfg["followup"]["evaluation"],
            output_csv=eval_csv,
        )
        eval_agg = aggregate_repairs(eval_df)
        eval_meta = (
            eval_df.groupby("method", as_index=False)[
                ["correction_scale", "parallel_keep"]
            ]
            .first()
        )
        eval_agg = eval_agg.merge(eval_meta, on="method", how="left")
        frontier = interpolated_frontier_summary(eval_agg, thresholds)
        eval_agg.to_csv(out_dir / "heldout_aggregate.csv", index=False)
        frontier.to_csv(out_dir / "heldout_interpolated_frontier.csv", index=False)

        calibration_agg = pd.read_csv(out_dir / "calibration_aggregate.csv")
        calibration_scores = pd.read_csv(out_dir / "calibration_beta_scores.csv")
        save_followup_plots(
            calibration_agg,
            calibration_scores,
            eval_agg,
            output_dir=out_dir,
        )
        write_summary(
            selection=selection,
            frontier=frontier,
            thresholds=thresholds,
            path=out_dir / "SUMMARY.md",
        )

        print("\nHeld-out interpolated frontier:")
        print(frontier.to_string(index=False))
        print("\nSaved inference-only follow-up results to:", out_dir)
        print("No denoiser training was performed.")


if __name__ == "__main__":
    main()
