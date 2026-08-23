from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from steering_repair.train_denoiser import TrainConfig, train_denoiser


def main() -> None:
    parser = argparse.ArgumentParser(description="Train activation denoiser")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    parser.add_argument("--kind", choices=["gaussian", "mixed"], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg_raw = cfg["training"]
    train_cfg = TrainConfig(**{k: v for k, v in tcfg_raw.items() if k != "device"})
    out = train_denoiser(
        cache_path=cfg["activation_cache"]["path"],
        checkpoint_path=cfg["denoisers"][args.kind]["checkpoint"],
        history_path=cfg["denoisers"][args.kind]["history"],
        kind=args.kind,
        config=train_cfg,
        device=tcfg_raw.get("device", "auto"),
    )
    print(out)


if __name__ == "__main__":
    main()
