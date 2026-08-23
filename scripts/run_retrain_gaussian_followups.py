from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh-runtime recovery: retrain only Gaussian denoiser, then run scaled-DPAR follow-ups"
    )
    parser.add_argument(
        "--config", default="configs/retrain_gaussian_followups_gpt2.yaml"
    )
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    cfg = args.config

    direction = Path("results/sentiment_direction.pt")
    if not direction.exists():
        run(
            "scripts/validate_sentiment_baseline.py",
            "--config",
            "configs/baseline_sentiment_gpt2.yaml",
        )

    run("scripts/preflight_repair_suite.py", "--config", cfg)

    cache_cmd = ["scripts/cache_activations.py", "--config", cfg]
    if args.force_cache:
        cache_cmd.append("--force")
    run(*cache_cmd)

    run("scripts/train_denoiser.py", "--config", cfg, "--kind", "gaussian")

    history_path = Path("results/retrained_denoiser_gaussian_history.json")
    history = json.loads(history_path.read_text())
    best = max(float(row["val_relative_mse_improvement"]) for row in history)
    print(f"\nBest Gaussian validation MSE improvement: {100.0 * best:.1f}%")
    if best < 0.50:
        raise RuntimeError(
            "Retrained Gaussian denoiser is substantially weaker than the previous run (<50% relative validation MSE improvement). Stop before expensive generation."
        )

    run("scripts/run_inference_followups.py", "--config", cfg, "--phase", "calibration")
    run("scripts/run_inference_followups.py", "--config", cfg, "--phase", "evaluation")

    print("\nRETRAIN + INFERENCE FOLLOW-UPS COMPLETED")
    print("Results: results/retrained_inference_followups")
    print("Checkpoint: checkpoints/retrained_denoiser_gaussian.pt")


if __name__ == "__main__":
    main()
