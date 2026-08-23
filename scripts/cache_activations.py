from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from steering_repair.activation_cache import (
    ActivationCacheConfig,
    cache_layer_activations,
    load_generic_texts,
    save_activation_cache,
)
from steering_repair.sentiment_baseline import load_gpt2


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache generic GPT-2 layer-6 activations")
    parser.add_argument("--config", default="configs/repair_suite_gpt2.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    acfg_raw = cfg["activation_cache"]
    path = Path(acfg_raw["path"])
    if path.exists() and not args.force:
        print(f"Activation cache already exists: {path} (use --force to rebuild)")
        return

    cache_cfg = ActivationCacheConfig(
        **{k: v for k, v in acfg_raw.items() if k != "path"}
    )
    texts = load_generic_texts(cache_cfg)
    print(f"Loaded {len(texts)} generic texts")
    model = load_gpt2(cfg)
    payload, metadata = cache_layer_activations(
        model,
        hook_name=cfg["vector"]["hook_name"],
        texts=texts,
        cfg=cache_cfg,
    )
    save_activation_cache(path, payload, metadata)
    print("Saved:", path)
    print(metadata)


if __name__ == "__main__":
    main()
