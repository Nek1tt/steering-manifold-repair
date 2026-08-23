from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import yaml

import torch

from steering_repair.activation_cache import (
    ActivationCacheConfig,
    cache_layer_activations,
    load_generic_texts,
)
from steering_repair.sentiment_baseline import load_gpt2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast real-data preflight before caching/training the repair suite"
    )
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = cfg["activation_cache"]
    base = ActivationCacheConfig(**{k: v for k, v in raw.items() if k != "path"})
    tiny = replace(
        base,
        max_texts=min(8, base.max_texts),
        batch_size=min(4, base.batch_size),
        max_seq_tokens=min(48, base.max_seq_tokens),
        max_activations=min(128, base.max_activations),
        token_stride=max(1, base.token_stride),
    )

    print("Repair-suite preflight")
    print("Python:", sys.version.split()[0])
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    texts = load_generic_texts(tiny)
    print(f"Dataset PASS: loaded {len(texts)} usable texts from {tiny.dataset_name}")

    model = load_gpt2(cfg)
    payload, metadata = cache_layer_activations(
        model,
        hook_name=cfg["vector"]["hook_name"],
        texts=texts,
        cfg=tiny,
    )
    train = payload["train"]
    val = payload["val"]
    expected_d = int(model.cfg.d_model)
    if train.shape[-1] != expected_d or val.shape[-1] != expected_d:
        raise RuntimeError(
            f"Activation width mismatch: expected {expected_d}, "
            f"train={train.shape[-1]}, val={val.shape[-1]}"
        )
    if metadata["hook_name"] != cfg["vector"]["hook_name"]:
        raise RuntimeError("Cached the wrong hook")

    print(
        "Hook/cache PASS: "
        f"train={tuple(train.shape)} val={tuple(val.shape)} "
        f"stop_at_layer={metadata.get('stop_at_layer')}"
    )
    print("REPAIR SUITE PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
