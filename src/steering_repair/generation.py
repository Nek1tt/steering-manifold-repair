from __future__ import annotations

from contextlib import nullcontext

import torch

from .steering import SteeringHook


def _sample_top_p(logits: torch.Tensor, *, temperature: float, top_p: float, generator: torch.Generator) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
        return sorted_idx.gather(-1, sampled_sorted)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_one(
    model,
    prompt: str,
    *,
    hook_name: str,
    direction: torch.Tensor,
    strength: float,
    strength_mode: str,
    repair: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict:
    """Autoregressive generation with deterministic per-sample RNG and response-token steering."""
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    prompt_len = tokens.shape[1]
    device = tokens.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    hook = SteeringHook(
        direction=direction,
        strength=float(strength),
        strength_mode=strength_mode,
        repair=repair,
    )

    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    for _ in range(max_new_tokens):
        ctx = nullcontext() if strength == 0 else model.hooks(fwd_hooks=[(hook_name, hook)])
        with ctx:
            logits = model(tokens, return_type="logits")
        next_token = _sample_top_p(
            logits[:, -1, :], temperature=temperature, top_p=top_p, generator=generator
        )
        tokens = torch.cat([tokens, next_token], dim=1)
        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    continuation_tokens = tokens[:, prompt_len:]
    continuation = model.to_string(continuation_tokens[0])
    return {
        "tokens": tokens,
        "prompt_len": prompt_len,
        "continuation": continuation,
    }
