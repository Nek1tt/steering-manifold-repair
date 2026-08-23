from __future__ import annotations

import torch

from .experiment import load_model, load_prompts
from .generation import generate_batch
from .metrics import select_concept_score, text_metrics
from .sae import decoder_direction, feature_pre_activation, load_openai_sae
from .steering import SteeringHook


def validate_vector(cfg) -> dict:
    """Cheap preflight: verify that the chosen direction changes generated text."""
    model = load_model(cfg)
    sae = load_openai_sae(
        location=cfg.sae.location,
        layer=cfg.sae.layer,
        width=cfg.sae.width,
        device=model.cfg.device,
    )
    direction = decoder_direction(sae, cfg.sae.feature_id, unit_norm=True)
    prompts = load_prompts(cfg.experiment.calibration_prompts_path)

    # Mechanistic diagnostic only: this is not the acceptance criterion, because
    # an SAE decoder direction need not be exactly dual to its encoder row.
    max_alpha = max(cfg.steering.strengths)
    steering = SteeringHook(
        direction=direction,
        strength=max_alpha,
        strength_mode=cfg.steering.strength_mode,
        repair=cfg.steering.repair,
    )
    direct_deltas: list[float] = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
            _, cache = model.run_with_cache(
                tokens, names_filter=[cfg.steering.hook_name]
            )
            clean = cache[cfg.steering.hook_name][:, -1, :]
            steered = steering.apply_last(clean)
            before = feature_pre_activation(sae, clean, cfg.sae.feature_id)
            after = feature_pre_activation(sae, steered, cfg.sae.feature_id)
            direct_deltas.append(float((after - before).mean().item()))
    direct_delta = sum(direct_deltas) / len(direct_deltas)

    rows = []
    for alpha in cfg.steering.strengths:
        scores: list[float] = []
        for seed in cfg.sampling.seeds:
            generated = generate_batch(
                model,
                prompts,
                hook_name=cfg.steering.hook_name,
                direction=direction,
                strength=float(alpha),
                strength_mode=cfg.steering.strength_mode,
                repair=cfg.steering.repair,
                max_new_tokens=min(64, cfg.sampling.max_new_tokens),
                temperature=cfg.sampling.temperature,
                top_p=cfg.sampling.top_p,
                seed=int(seed),
            )
            scores.extend(
                select_concept_score(
                    text_metrics(item["continuation"]), cfg.experiment.concept_metric
                )
                for item in generated
            )
        rows.append(
            {
                "strength": float(alpha),
                "concept_score": sum(scores) / len(scores),
            }
        )

    base = next(row["concept_score"] for row in rows if row["strength"] == 0)
    best = max(rows, key=lambda row: row["concept_score"])
    gain = float(best["concept_score"] - base)
    # The assignment's requirement is behavioral: the generated-text concept
    # must increase. The encoder-preactivation delta is retained as a diagnostic.
    passed = gain > 0
    return {
        "passed": passed,
        "direct_target_preact_delta": direct_delta,
        "concept_gain": gain,
        "best_strength": float(best["strength"]),
        "rows": rows,
    }
