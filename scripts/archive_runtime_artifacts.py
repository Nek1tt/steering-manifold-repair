from __future__ import annotations

import argparse
from pathlib import Path
import shutil


DEFAULT_FILES = {
    "results/sentiment_baseline_pareto.png": "experiments/successful_sentiment_baseline/sentiment_baseline_pareto.png",
    "results/repair_suite/repair_pareto.png": "experiments/repair_suite/repair_pareto.png",
    "results/repair_suite/effective_alpha.png": "experiments/repair_suite/effective_alpha.png",
    "results/repair_suite/correction_geometry.png": "experiments/repair_suite/correction_geometry.png",
}

FOLLOWUP_FILES = {
    "results/inference_followups/beta_calibration.png": "experiments/inference_only_followups/beta_calibration.png",
    "results/inference_followups/selected_dense_pareto.png": "experiments/inference_only_followups/selected_dense_pareto.png",
    "results/inference_followups/selected_effective_alpha.png": "experiments/inference_only_followups/selected_effective_alpha.png",
    "results/inference_followups/heldout_interpolated_frontier.csv": "experiments/inference_only_followups/heldout_interpolated_frontier.csv",
    "results/inference_followups/calibration_beta_scores.csv": "experiments/inference_only_followups/calibration_beta_scores.csv",
    "results/inference_followups/selection.json": "experiments/inference_only_followups/selection.json",
    "results/inference_followups/SUMMARY.md": "experiments/inference_only_followups/RESULTS.md",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy selected runtime artifacts into experiments/")
    parser.add_argument("--include-followup", action="store_true")
    args = parser.parse_args()

    mapping = dict(DEFAULT_FILES)
    if args.include_followup:
        mapping.update(FOLLOWUP_FILES)

    copied = 0
    missing = []
    for src_name, dst_name in mapping.items():
        src, dst = Path(src_name), Path(dst_name)
        if not src.exists():
            missing.append(src_name)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"copied {src} -> {dst}")

    print(f"\nArchived {copied} files.")
    if missing:
        print("Missing (safe to ignore if that stage has not run):")
        for item in missing:
            print(" -", item)


if __name__ == "__main__":
    main()
