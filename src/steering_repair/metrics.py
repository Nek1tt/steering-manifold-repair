from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn.functional as F

from .sae import encode_feature


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def distinct_n(text: str, n: int) -> float:
    words = _words(text)
    if len(words) < n or n <= 0:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_ngram(text: str, n: int = 3) -> float:
    words = _words(text)
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated_instances = sum(max(0, c - 1) for c in counts.values())
    return repeated_instances / len(grams)


def quoted_span_rate(text: str) -> float:
    """Quoted spans per 100 whitespace-separated tokens; a corroborating metric for feature 56907."""
    spans = re.findall(r'["“”‘’\']([^"“”‘’\']{1,40})["“”‘’\']', text)
    n_words = max(1, len(_words(text)))
    return 100.0 * len(spans) / n_words


@torch.no_grad()
def continuation_nll(model, tokens: torch.Tensor, prompt_len: int) -> float:
    """Score generated continuation under the clean, unsteered model."""
    logits = model(tokens, return_type="logits")
    # Prediction at position t-1 scores token t. Only generated tokens are included.
    start_pred = max(0, prompt_len - 1)
    pred_logits = logits[:, start_pred:-1, :]
    targets = tokens[:, prompt_len:]
    if targets.numel() == 0:
        return float("nan")
    nll = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    return float(nll.item())


@torch.no_grad()
def sae_concept_metrics(
    model,
    sae,
    tokens: torch.Tensor,
    *,
    prompt_len: int,
    hook_name: str,
    feature_id: int,
) -> dict[str, float]:
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    acts = cache[hook_name]
    target = encode_feature(sae, acts[:, prompt_len:, :], feature_id)
    if target.numel() == 0:
        return {
            "concept_sae_mean": float("nan"),
            "concept_sae_max": float("nan"),
            "concept_sae_firing_rate": float("nan"),
        }
    return {
        "concept_sae_mean": float(target.mean().item()),
        "concept_sae_max": float(target.max().item()),
        "concept_sae_firing_rate": float((target > 0).float().mean().item()),
    }


def text_metrics(text: str) -> dict[str, float]:
    return {
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "repetition_3gram": repetition_ngram(text, 3),
        "quoted_span_rate": quoted_span_rate(text),
    }


def ppl_from_nll(nll: float) -> float:
    if not math.isfinite(nll):
        return float("nan")
    return float(math.exp(min(20.0, nll)))
