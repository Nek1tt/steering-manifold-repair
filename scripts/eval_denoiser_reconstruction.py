from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from steering_repair.activation_cache import load_activation_cache
from steering_repair.denoiser import CorruptionConfig
from steering_repair.train_denoiser import evaluate_denoiser, load_denoiser_checkpoint


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-evaluate denoisers on held-out Gaussian and structured corruptions")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["training"]
    device = resolve_device(tcfg.get("device", "auto"))
    cache = load_activation_cache(cfg["activation_cache"]["path"])
    train_bank = cache["train"].float()[: min(8192, cache["train"].shape[0])].to(device)
    val = cache["val"].float()
    corruption = CorruptionConfig(
        ratio_min=float(tcfg["ratio_min"]),
        ratio_max=float(tcfg["ratio_max"]),
        structured_probability=float(tcfg.get("structured_probability", 0.5)),
        identity_probability=float(tcfg.get("identity_probability", 0.05)),
    )

    rows = []
    for checkpoint_name in ("gaussian", "mixed"):
        model, _ = load_denoiser_checkpoint(
            cfg["denoisers"][checkpoint_name]["checkpoint"], device=device
        )
        for corruption_kind in ("gaussian", "structured"):
            metrics = evaluate_denoiser(
                model,
                val,
                kind=corruption_kind,
                corruption=corruption,
                train_bank=train_bank,
                device=device,
                batch_size=int(tcfg["batch_size"]),
            )
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "eval_corruption": corruption_kind,
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    out_dir = Path(cfg["evaluation"].get("output_dir", "results/repair_suite"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "denoiser_cross_reconstruction.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
