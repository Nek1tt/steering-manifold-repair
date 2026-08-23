from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ActivationCacheConfig:
    # Use the canonical Hub repository name. The historical shorthand
    # ``wikitext`` has become brittle across recent datasets/hub releases.
    dataset_name: str = "Salesforce/wikitext"
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
    streaming: bool = True


def _load_dataset_with_fallback(cfg: ActivationCacheConfig):
    """Load generic text without depending on legacy Hub dataset scripts.

    Hugging Face converted WikiText to Parquet and recent ``datasets`` releases
    no longer execute remote loading scripts. The canonical repository usually
    loads directly, but an explicit Parquet fallback keeps Colab runs robust
    against alias/config resolution regressions in ``datasets`` / hub clients.
    """

    from datasets import load_dataset

    errors: list[str] = []
    requested = cfg.dataset_name
    candidates: list[str] = []
    if requested in {"wikitext", "Salesforce/wikitext"}:
        candidates.append("Salesforce/wikitext")
    if requested not in candidates:
        candidates.append(requested)

    for name in candidates:
        try:
            kwargs = {"split": cfg.split, "streaming": cfg.streaming}
            if cfg.dataset_config:
                return load_dataset(name, cfg.dataset_config, **kwargs)
            return load_dataset(name, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on Hub state
            errors.append(f"load_dataset({name!r}, config={cfg.dataset_config!r}): {exc!r}")

    # Current Salesforce/wikitext is native Parquet. Loading the files
    # explicitly avoids remote-script/config machinery used by older datasets.
    if requested in {"wikitext", "Salesforce/wikitext"} and cfg.dataset_config:
        pattern = (
            "hf://datasets/Salesforce/wikitext/"
            f"{cfg.dataset_config}/{cfg.split}-*.parquet"
        )
        try:
            return load_dataset(
                "parquet",
                data_files={cfg.split: pattern},
                split=cfg.split,
                streaming=cfg.streaming,
            )
        except Exception as exc:  # pragma: no cover - depends on Hub state
            errors.append(f"explicit Parquet fallback {pattern!r}: {exc!r}")

    joined = "\n  - ".join(errors)
    raise RuntimeError(
        "Could not load the generic activation-caching corpus. Attempts:\n"
        f"  - {joined}\n"
        "If Hugging Face is temporarily rate-limited, set HF_TOKEN and rerun "
        "only scripts/cache_activations.py."
    )


def load_generic_texts(cfg: ActivationCacheConfig) -> list[str]:
    ds = _load_dataset_with_fallback(cfg)

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


def _stop_at_layer_for_hook(hook_name: str) -> int | None:
    """Return the earliest safe TransformerLens stop layer for resid_post hooks."""

    parts = hook_name.split(".")
    if len(parts) >= 3 and parts[0] == "blocks" and parts[2] == "hook_resid_post":
        try:
            return int(parts[1]) + 1
        except ValueError:
            return None
    return None


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
    positions are retained and padding positions are discarded. For a
    ``blocks.N.hook_resid_post`` target we stop immediately after block N; later
    transformer blocks and the unembedding are unnecessary for caching.
    """

    collected: list[torch.Tensor] = []
    total = 0
    stop_at_layer = _stop_at_layer_for_hook(hook_name)
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
        _, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name],
            return_type=None,
            stop_at_layer=stop_at_layer,
        )
        if hook_name not in cache:
            raise KeyError(
                f"Requested activation hook {hook_name!r} was not cached. "
                f"Available matching keys: {[k for k in cache.keys() if 'resid' in k][-10:]}"
            )
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
    if all_acts.shape[0] < 2:
        raise RuntimeError("Need at least two cached activations for a train/validation split")
    if not torch.isfinite(all_acts.float()).all():
        raise RuntimeError("Activation cache contains NaN or Inf values")

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
        "stop_at_layer": stop_at_layer,
        "n_total": int(all_acts.shape[0]),
        "n_train": int(payload["train"].shape[0]),
        "n_val": int(payload["val"].shape[0]),
        "d_model": int(all_acts.shape[-1]),
        "dtype": "float16",
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "split": cfg.split,
        "streaming": cfg.streaming,
        "max_seq_tokens": cfg.max_seq_tokens,
        "seed": cfg.seed,
    }
    return payload, metadata


def save_activation_cache(path: str | Path, payload: dict, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save({**payload, "metadata": metadata}, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_activation_cache(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "train" not in payload or "val" not in payload:
        raise ValueError(f"Invalid activation cache: {path}")
    train = payload["train"]
    val = payload["val"]
    if train.ndim != 2 or val.ndim != 2 or train.shape[-1] != val.shape[-1]:
        raise ValueError(
            f"Invalid activation cache tensor shapes: train={tuple(train.shape)}, "
            f"val={tuple(val.shape)}"
        )
    if train.shape[0] < 1 or val.shape[0] < 1:
        raise ValueError("Activation cache has an empty train or validation split")
    return payload
