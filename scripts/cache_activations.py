from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import yaml

from steering_repair.activation_cache import (
    ActivationCacheConfig,
    cache_layer_activations,
    load_activation_cache,
    load_generic_texts,
    save_activation_cache,
)
from steering_repair.sentiment_baseline import load_gpt2


def _print_environment() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    try:
        import datasets

        print(f"datasets: {datasets.__version__}")
    except Exception as exc:
        print(f"datasets import failed: {exc!r}")
    try:
        import huggingface_hub

        print(f"huggingface_hub: {huggingface_hub.__version__}")
    except Exception as exc:
        print(f"huggingface_hub import failed: {exc!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache generic GPT-2 layer-6 activations")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    acfg_raw = cfg["activation_cache"]
    path = Path(acfg_raw["path"])

    if path.exists() and not args.force:
        try:
            existing = load_activation_cache(path)
            print(
                f"Activation cache already exists and is valid: {path} "
                f"train={tuple(existing['train'].shape)} val={tuple(existing['val'].shape)}"
            )
            return
        except Exception as exc:
            print(f"Existing cache is invalid ({exc!r}); rebuilding it.")
            path.unlink(missing_ok=True)

    _print_environment()
    cache_cfg = ActivationCacheConfig(
        **{k: v for k, v in acfg_raw.items() if k != "path"}
    )
    print(
        "Generic corpus: "
        f"dataset={cache_cfg.dataset_name!r} config={cache_cfg.dataset_config!r} "
        f"split={cache_cfg.split!r} streaming={cache_cfg.streaming}"
    )
    texts = load_generic_texts(cache_cfg)
    print(f"Loaded {len(texts)} usable generic texts")

    model = load_gpt2(cfg)
    payload, metadata = cache_layer_activations(
        model,
        hook_name=cfg["vector"]["hook_name"],
        texts=texts,
        cfg=cache_cfg,
    )
    save_activation_cache(path, payload, metadata)
    # Read it back before declaring success; this catches truncated writes.
    checked = load_activation_cache(path)
    print("Saved and verified:", path)
    print(
        f"train={tuple(checked['train'].shape)} val={tuple(checked['val'].shape)} "
        f"dtype={checked['train'].dtype}"
    )
    print(metadata)


if __name__ == "__main__":
    main()
