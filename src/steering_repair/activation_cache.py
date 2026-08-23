from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ActivationCacheConfig:
    dataset_name: str = "wikitext"
    dataset_config: str | None = "wikitext-2-raw-v1"
    split: str = "train"
    text_field: str = "text"
    max_texts: int = 3000
    batch_size: int = 16
    max_seq_tokens: int = 128
    max_activations: int = 80000
    min_chars: int = 40
    token_stride: int = 1
    val_fraction: float = 0.10
    seed: int = 1234


def load_generic_texts(cfg: ActivationCacheConfig) -> list[str]:
    from datasets import load_dataset

    if cfg.dataset_config:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    else:
        ds = load_dataset(cfg.dataset_name, split=cfg.split)

    texts: list[str] = []
    for row in ds:
        text = str(row.get(cfg.text_field, "")).strip()
        if len(text) < cfg.min_chars:
            continue
        texts.append(text)
        if len(texts) >= cfg.max_texts:
            break
    if not texts:
        raise RuntimeError("No usable generic texts were loaded for activation caching")
    return texts


@torch.no_grad()
def cache_layer_activations(
    model,
    *,
    hook_name: str,
    texts: list[str],
    cfg: ActivationCacheConfig,
) -> tuple[dict, dict]:
    """Collect natural residual-stream activations from generic text.

    Right padding cannot affect earlier causal-prefix states, so only valid token
    positions are retained and padding positions are discarded.
    """

    collected: list[torch.Tensor] = []
    total = 0
    batches = range(0, len(texts), cfg.batch_size)
    pbar = tqdm(batches, desc="cache layer activations", unit="batch")
    for start in pbar:
        batch = texts[start : start + cfg.batch_size]
        lengths = [
            min(
                cfg.max_seq_tokens,
                len(model.tokenizer.encode(text, add_special_tokens=False)) + 1,
            )
            for text in batch
        ]
        tokens = model.to_tokens(batch, prepend_bos=True, padding_side="right").to(
            model.cfg.device
        )
        tokens = tokens[:, : cfg.max_seq_tokens]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        acts = cache[hook_name]

        for row_idx, length in enumerate(lengths):
            length = min(int(length), acts.shape[1])
            if length <= 1:
                continue
            valid = acts[row_idx, 1:length: max(1, cfg.token_stride)]
            if valid.numel() == 0:
                continue
            remaining = cfg.max_activations - total
            if remaining <= 0:
                break
            valid = valid[:remaining]
            collected.append(valid.detach().to(dtype=torch.float16, device="cpu"))
            total += int(valid.shape[0])
        pbar.set_postfix(rows=total)
        if total >= cfg.max_activations:
            break

    if not collected:
        raise RuntimeError("Activation cache is empty")
    all_acts = torch.cat(collected, dim=0)[: cfg.max_activations]

    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(all_acts.shape[0], generator=generator)
    all_acts = all_acts[perm]
    n_val = max(1, int(round(cfg.val_fraction * all_acts.shape[0])))
    n_val = min(n_val, all_acts.shape[0] - 1)
    payload = {
        "train": all_acts[n_val:].contiguous(),
        "val": all_acts[:n_val].contiguous(),
    }
    metadata = {
        "hook_name": hook_name,
        "n_total": int(all_acts.shape[0]),
        "n_train": int(payload["train"].shape[0]),
        "n_val": int(payload["val"].shape[0]),
        "d_model": int(all_acts.shape[-1]),
        "dtype": "float16",
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "split": cfg.split,
        "max_seq_tokens": cfg.max_seq_tokens,
        "seed": cfg.seed,
    }
    return payload, metadata


def save_activation_cache(path: str | Path, payload: dict, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "metadata": metadata}, path)


def load_activation_cache(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "train" not in payload or "val" not in payload:
        raise ValueError(f"Invalid activation cache: {path}")
    return payload
