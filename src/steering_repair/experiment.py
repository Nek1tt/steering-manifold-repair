from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import Config
from .generation import generate_batch
from .metrics import ppl_from_nll, score_continuation, select_concept_score, text_metrics
from .sae import decoder_direction, load_openai_sae


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg: Config):
    from transformer_lens import HookedTransformer

    device = resolve_device(cfg.model.device)
    dtype = getattr(torch, cfg.model.dtype)
    # This matches the loading choice in OpenAI's released GPT-2 SAE example.
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

    output_path = Path(cfg.experiment.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # One cached generation call handles all prompts for a fixed (alpha, seed).
    # This is much faster than re-running the full prefix for every generated token.
    jobs = [
        (float(strength), int(seed))
        for strength in cfg.steering.strengths
        for seed in cfg.sampling.seeds
    ]
    pbar = tqdm(jobs, desc="baseline batches", unit="batch")
    for job_idx, (strength, seed) in enumerate(pbar, start=1):
        pbar.set_postfix(alpha=f"{strength:g}", seed=seed)
        generated_batch = generate_batch(
            model,
            prompts,
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

        for prompt_idx, (prompt, generated) in enumerate(zip(prompts, generated_batch)):
            tokens = generated["tokens"]
            prompt_len = generated["prompt_len"]
            continuation = generated["continuation"]
            internal = score_continuation(
                model,
                sae,
                tokens,
                prompt_len=prompt_len,
                hook_name=cfg.steering.hook_name,
                feature_id=cfg.sae.feature_id,
            )
            tm = text_metrics(continuation)
            rows.append(
                {
                    "model": cfg.model.name,
                    "layer": cfg.sae.layer,
                    "feature_id": cfg.sae.feature_id,
                    "feature_label": cfg.sae.feature_label,
                    "concept_metric": cfg.experiment.concept_metric,
                    "method": cfg.steering.repair,
                    "strength_mode": cfg.steering.strength_mode,
                    "strength": strength,
                    "seed": seed,
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "continuation": continuation,
                    "concept_score": select_concept_score(
                        tm, cfg.experiment.concept_metric
                    ),
                    "nll": internal["nll"],
                    "ppl": ppl_from_nll(internal["nll"]),
                    **tm,
                    "concept_sae_mean": internal["concept_sae_mean"],
                    "concept_sae_max": internal["concept_sae_max"],
                    "concept_sae_firing_rate": internal["concept_sae_firing_rate"],
                }
            )

        if cfg.experiment.save_every > 0 and job_idx % cfg.experiment.save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
