from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import Config
from .generation import generate_one
from .metrics import continuation_nll, ppl_from_nll, sae_concept_metrics, text_metrics
from .sae import decoder_direction, load_openai_sae


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg: Config):
    from transformer_lens import HookedTransformer

    device = resolve_device(cfg.model.device)
    dtype = getattr(torch, cfg.model.dtype)
    model = HookedTransformer.from_pretrained(
        cfg.model.name,
        device=device,
        dtype=dtype,
        center_writing_weights=False,
    )
    model.eval()
    return model


def load_prompts(path: str | Path) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def run_baseline(cfg: Config) -> pd.DataFrame:
    model = load_model(cfg)
    sae = load_openai_sae(
        location=cfg.sae.location,
        layer=cfg.sae.layer,
        width=cfg.sae.width,
        device=model.cfg.device,
    )
    direction = decoder_direction(sae, cfg.sae.feature_id, unit_norm=True)
    prompts = load_prompts(cfg.experiment.prompts_path)

    jobs = [
        (prompt_idx, prompt, seed, strength)
        for strength in cfg.steering.strengths
        for seed in cfg.sampling.seeds
        for prompt_idx, prompt in enumerate(prompts)
    ]
    rows: list[dict] = []
    output_path = Path(cfg.experiment.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(jobs, desc="baseline generations", unit="sample")
    for job_idx, (prompt_idx, prompt, seed, strength) in enumerate(pbar, start=1):
        pbar.set_postfix(strength=f"{strength:.3g}", seed=seed, prompt=prompt_idx)
        generated = generate_one(
            model,
            prompt,
            hook_name=cfg.steering.hook_name,
            direction=direction,
            strength=strength,
            strength_mode=cfg.steering.strength_mode,
            repair=cfg.steering.repair,
            max_new_tokens=cfg.sampling.max_new_tokens,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            seed=seed,
        )
        tokens = generated["tokens"]
        prompt_len = generated["prompt_len"]
        continuation = generated["continuation"]
        nll = continuation_nll(model, tokens, prompt_len)
        concept = sae_concept_metrics(
            model,
            sae,
            tokens,
            prompt_len=prompt_len,
            hook_name=cfg.steering.hook_name,
            feature_id=cfg.sae.feature_id,
        )
        row = {
            "model": cfg.model.name,
            "layer": cfg.sae.layer,
            "feature_id": cfg.sae.feature_id,
            "feature_label": cfg.sae.feature_label,
            "method": cfg.steering.repair,
            "strength_mode": cfg.steering.strength_mode,
            "strength": float(strength),
            "seed": int(seed),
            "prompt_id": int(prompt_idx),
            "prompt": prompt,
            "continuation": continuation,
            "nll": nll,
            "ppl": ppl_from_nll(nll),
            **text_metrics(continuation),
            **concept,
        }
        rows.append(row)

        if cfg.experiment.save_every > 0 and job_idx % cfg.experiment.save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
