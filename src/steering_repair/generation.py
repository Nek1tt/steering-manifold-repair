from __future__ import annotations

from contextlib import nullcontext

import torch

from .steering import SteeringHook


def _trim_generated(tokens: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    if eos_id is None or tokens.numel() == 0:
        return tokens
    hits = (tokens == int(eos_id)).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return tokens
    return tokens[: int(hits[0].item())]


@torch.no_grad()
def generate_batch(
    model,
    prompts: list[str],
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
) -> list[dict]:
    """Batched, cached generation with the same steering hook on each decoding step."""
    if not prompts:
        return []

    padded_prompt_tokens = model.to_tokens(
        prompts, prepend_bos=True, padding_side="left"
    ).to(model.cfg.device)
    input_width = padded_prompt_tokens.shape[1]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hook = SteeringHook(
        direction=direction,
        strength=float(strength),
        strength_mode=strength_mode,
        repair=repair,
    )
    ctx = nullcontext() if strength == 0 else model.hooks(fwd_hooks=[(hook_name, hook)])
    with ctx:
        output = model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            stop_at_eos=True,
            do_sample=temperature > 0,
            top_p=top_p,
            temperature=max(temperature, 1e-6),
            use_past_kv_cache=True,
            prepend_bos=True,
            padding_side="left",
            return_type="tokens",
            verbose=False,
        )

    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    rows: list[dict] = []
    for i, prompt in enumerate(prompts):
        new_tokens = _trim_generated(output[i, input_width:], eos_id)
        prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
        full_tokens = torch.cat([prompt_tokens, new_tokens.unsqueeze(0)], dim=1)
        rows.append(
            {
                "tokens": full_tokens,
                "prompt_len": int(prompt_tokens.shape[1]),
                "continuation": model.to_string(new_tokens),
            }
        )
    return rows


@torch.no_grad()
def generate_one(model, prompt: str, **kwargs) -> dict:
    return generate_batch(model, [prompt], **kwargs)[0]
