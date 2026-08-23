from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import yaml


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(list(args), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full denoiser + DPAR experiment suite")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    yaml.safe_load(Path(args.config).read_text())

    # Fail fast on the external dataset + TransformerLens hook boundary before
    # spending time on the 80k-activation cache or denoiser training.
    run(sys.executable, "scripts/preflight_repair_suite.py", "--config", args.config)

    cache_cmd = [sys.executable, "scripts/cache_activations.py", "--config", args.config]
    if args.force_cache:
        cache_cmd.append("--force")
    run(*cache_cmd)
    run(sys.executable, "scripts/train_denoiser.py", "--config", args.config, "--kind", "gaussian")
    run(sys.executable, "scripts/train_denoiser.py", "--config", args.config, "--kind", "mixed")
    run(sys.executable, "scripts/eval_denoiser_reconstruction.py", "--config", args.config)
    run(sys.executable, "scripts/eval_repairs.py", "--config", args.config)
    run(sys.executable, "scripts/plot_repairs.py", "--config", args.config)
    print("\nRepair suite completed.")


if __name__ == "__main__":
    main()
