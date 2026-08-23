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
    return sum(max(0, c - 1) for c in counts.values()) / len(grams)


def quoted_span_rate(text: str) -> float:
    spans = re.findall(r'["“”‘’\']([^"“”‘’\']{1,40})["“”‘’\']', text)
    return 100.0 * len(spans) / max(1, len(_words(text)))


# The default OpenAI SAE feature 64840 is explicitly described by the released
# viewer as a feature that activates in order to output profanity.  For the
# baseline we therefore score the generated text itself, rather than asking the
# same sparse feature to re-fire in a clean replay of that text.
_PROFANITY_RE = re.compile(
    r"^(?:fuck(?:ed|er|ers|ing|s)?|motherfuck(?:er|ers|ing)?|"
    r"shit(?:s|ty|ting)?|bullshit|bitch(?:es|y)?|asshole(?:s)?|"
    r"bastard(?:s)?|damn(?:ed)?|crap|dick(?:s|head|heads)?|"
    r"piss(?:ed|es|ing)?)$",
    flags=re.IGNORECASE,
)


def profanity_count(text: str) -> int:
    return sum(bool(_PROFANITY_RE.fullmatch(word)) for word in _words(text))


def profanity_rate(text: str) -> float:
    words = _words(text)
    return 100.0 * profanity_count(text) / max(1, len(words))


def profanity_any_pct(text: str) -> float:
    """0 or 100 per sample; after averaging this is % completions with the concept."""
    return 100.0 if profanity_count(text) > 0 else 0.0


@torch.no_grad()
def score_continuation(
    model,
    sae,
    tokens: torch.Tensor,
    *,
    prompt_len: int,
    hook_name: str,
    feature_id: int,
) -> dict[str, float]:
    """Compute clean-model NLL and SAE diagnostics in one clean forward pass."""
    logits, cache = model.run_with_cache(
        tokens, names_filter=[hook_name], return_type="logits"
    )
    start_pred = max(0, prompt_len - 1)
    pred_logits = logits[:, start_pred:-1, :]
    targets = tokens[:, prompt_len:]
    if targets.numel() == 0:
        nll = float("nan")
    else:
        nll = float(
            F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
            ).item()
        )

    target = encode_feature(sae, cache[hook_name][:, prompt_len:, :], feature_id)
    if target.numel() == 0:
        sae_metrics = {
            "concept_sae_mean": float("nan"),
            "concept_sae_max": float("nan"),
            "concept_sae_firing_rate": float("nan"),
        }
    else:
        sae_metrics = {
            "concept_sae_mean": float(target.mean().item()),
            "concept_sae_max": float(target.max().item()),
            "concept_sae_firing_rate": float((target > 0).float().mean().item()),
        }
    return {"nll": nll, **sae_metrics}


@torch.no_grad()
def continuation_nll(model, tokens: torch.Tensor, prompt_len: int) -> float:
    logits = model(tokens, return_type="logits")
    start_pred = max(0, prompt_len - 1)
    pred_logits = logits[:, start_pred:-1, :]
    targets = tokens[:, prompt_len:]
    if targets.numel() == 0:
        return float("nan")
    return float(
        F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        ).item()
    )


@torch.no_grad()
def sae_concept_metrics(model, sae, tokens: torch.Tensor, *, prompt_len: int, hook_name: str, feature_id: int) -> dict[str, float]:
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    target = encode_feature(sae, cache[hook_name][:, prompt_len:, :], feature_id)
    if target.numel() == 0:
        return {"concept_sae_mean": float("nan"), "concept_sae_max": float("nan"), "concept_sae_firing_rate": float("nan")}
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
        "profanity_rate": profanity_rate(text),
        "profanity_any_pct": profanity_any_pct(text),
    }


def select_concept_score(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(
            f"Unknown concept metric {metric_name!r}; available: {sorted(metrics)}"
        )
    return float(metrics[metric_name])


def ppl_from_nll(nll: float) -> float:
    if not math.isfinite(nll):
        return float("nan")
    return float(math.exp(min(20.0, nll)))
