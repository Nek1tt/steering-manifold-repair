from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download
import yaml

from steering_repair.inference_followups import (
    ScaledRepairSpec,
    interpolated_frontier_summary,
    run_scaled_evaluation,
)
from steering_repair.repair_experiment import aggregate_repairs


HF_REPO_ID = "Nek1tt/steering-repair-gpt2"
HF_CHECKPOINT = "retrained_denoiser_gaussian.pt"
DEFAULT_CONFIG = "configs/retrain_gaussian_followups_gpt2.yaml"
DEFAULT_DIRECTION_CONFIG = "configs/baseline_sentiment_gpt2.yaml"
DEFAULT_OUTPUT_DIR = "results/hf_dpar_reproduction"


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(list(args), check=True)


def ensure_direction(cfg: dict) -> Path:
    path = Path(cfg["vector"]["cache_path"])
    if path.exists():
        print(f"Steering direction: {path}")
        return path

    print(f"Steering direction not found: {path}")
    print("Building it from the positive/negative text examples...")
    run(
        sys.executable,
        "scripts/validate_sentiment_baseline.py",
        "--config",
        DEFAULT_DIRECTION_CONFIG,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Direction validation completed, but {path} was not created."
        )
    return path


def prepare_checkpoint(
    cfg: dict,
    *,
    skip_download: bool,
) -> Path:
    expected = Path(cfg["denoisers"]["gaussian"]["checkpoint"])

    if skip_download:
        if not expected.exists():
            raise FileNotFoundError(
                f"--skip-download was requested, but local checkpoint is missing: {expected}"
            )
        print(f"Using local checkpoint: {expected}")
        return expected

    if expected.name != HF_CHECKPOINT:
        raise ValueError(
            "The frozen config no longer points to the published checkpoint name: "
            f"{expected.name!r} != {HF_CHECKPOINT!r}"
        )

    expected.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CHECKPOINT,
            local_dir=str(expected.parent),
            force_download=True,
        )
    )

    if not expected.exists():
        raise FileNotFoundError(
            f"Hugging Face download returned {downloaded}, but expected path is missing: "
            f"{expected}"
        )

    print(f"HF checkpoint: {expected}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the frozen additive / Gaussian / DPAR held-out evaluation "
            "using the published Hugging Face checkpoint."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=(
            "Use the checkpoint already present at the path from the config. "
            "Useful after a fresh retrain."
        ),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_direction(cfg)
    prepare_checkpoint(
        cfg,
        skip_download=args.skip_download,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ScaledRepairSpec(
            "additive",
            checkpoint_kind=None,
            correction_scale=0.0,
            parallel_keep=1.0,
        ),
        ScaledRepairSpec(
            "gaussian_vanilla_b100",
            checkpoint_kind="gaussian",
            correction_scale=1.0,
            parallel_keep=1.0,
        ),
        ScaledRepairSpec(
            "gaussian_dpar_b100",
            checkpoint_kind="gaussian",
            correction_scale=1.0,
            parallel_keep=0.0,
        ),
    ]

    samples = run_scaled_evaluation(
        cfg,
        specs=specs,
        phase_cfg=cfg["followup"]["evaluation"],
        output_csv=output_dir / "samples.csv",
    )

    aggregate = aggregate_repairs(samples)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)

    thresholds = [float(x) for x in cfg["followup"]["concept_thresholds"]]
    frontier = interpolated_frontier_summary(aggregate, thresholds)
    frontier.to_csv(output_dir / "interpolated_frontier.csv", index=False)

    indexed = frontier.set_index("method")
    c90 = "fluency_at_concept_90"
    additive_c90 = float(indexed.loc["additive", c90])
    dpar_c90 = float(indexed.loc["gaussian_dpar_b100", c90])

    vanilla = aggregate[
        (aggregate["method"] == "gaussian_vanilla_b100")
        & (aggregate["strength"] > 0)
    ]
    dpar = aggregate[
        (aggregate["method"] == "gaussian_dpar_b100")
        & (aggregate["strength"] > 0)
    ]

    vanilla_alpha_error = float(vanilla["alpha_preservation_error"].mean())
    dpar_alpha_error = float(dpar["alpha_preservation_error"].mean())
    dpar_alpha_error_max = float(dpar["alpha_preservation_error"].max())

    print("\nInterpolated held-out frontier:")
    print(frontier.to_string(index=False))

    print("\nReference comparison:")
    print(f"  additive F@C90 : {additive_c90:.2f}  (reference 66.46)")
    print(f"  DPAR F@C90     : {dpar_c90:.2f}  (reference 71.45)")
    print(f"  DPAR gain      : {dpar_c90 - additive_c90:+.2f}  (reference +4.99)")

    print("\nSteering preservation:")
    print(
        f"  vanilla mean |effective alpha - alpha| : "
        f"{vanilla_alpha_error:.6g}"
    )
    print(
        f"  DPAR mean |effective alpha - alpha|    : "
        f"{dpar_alpha_error:.6g}"
    )
    print(
        f"  DPAR max  |effective alpha - alpha|    : "
        f"{dpar_alpha_error_max:.6g}"
    )

    if dpar_alpha_error_max > 1e-4:
        raise RuntimeError(
            "DPAR alpha-preservation invariant failed: "
            f"max error={dpar_alpha_error_max:.6g}"
        )

    print("\nDPAR GEOMETRY CHECK: PASS")
    print(f"Saved reproduction artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
