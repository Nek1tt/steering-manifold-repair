from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

from .inference_followups import interpolated_frontier_summary
from .jrr import (
    capture_source_last,
    decompose_remainder,
    downstream_map,
    model_directional_jvp,
    project_onto_direction,
    replace_target_last_hook,
    additive_last_hook,
    validate_hooks,
)
from .jrr_diagnostic import aggregate_behavior
from .jrr_oracle import additive_logits, oracle_jrr_logits, sample_top_p
from .metrics import continuation_nll, text_metrics
from .sentiment_baseline import SentimentJudge, load_direction, load_gpt2, load_lines


def kl_clean_to_current(clean_logits: torch.Tensor, current_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_clean || p_current) for next-token distributions."""
    clean_logp = torch.log_softmax(clean_logits.detach(), dim=-1)
    clean_p = clean_logp.exp()
    current_logp = torch.log_softmax(current_logits, dim=-1)
    return (clean_p * (clean_logp - current_logp)).sum(dim=-1).mean()


def orthogonalize_to_direction(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return x - project_onto_direction(x, direction)


def select_kl_harmful_component(
    remainder_orth: torch.Tensor,
    transported_direction: torch.Tensor,
    kl_gradient: torch.Tensor,
    *,
    positive_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the part of R_orth aligned with KL-increasing local sensitivity.

    The KL gradient is first projected into the complement of the transported
    first-order steering direction. Therefore the returned correction cannot
    change the component parallel to Jv. If positive_only=True, residual
    components that locally *reduce* KL from the clean model are preserved.
    """
    g = orthogonalize_to_direction(kl_gradient, transported_direction)
    denom = g.square().sum().clamp_min(1e-12)
    coeff = (remainder_orth * g).sum() / denom
    if positive_only:
        coeff = coeff.clamp_min(0.0)
    selected = coeff * g
    return selected, g, coeff


def _target_logits(model, tokens: torch.Tensor, target_hook: str, value: torch.Tensor) -> torch.Tensor:
    with model.hooks(fwd_hooks=[(target_hook, replace_target_last_hook(value))]):
        return model(tokens)[:, -1, :]


def kl_gradient_at_target(
    model,
    tokens: torch.Tensor,
    *,
    target_hook: str,
    target_value: torch.Tensor,
    clean_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return steered logits and d KL(clean || steered) / d target activation."""
    with torch.enable_grad():
        y = target_value.detach().requires_grad_(True)
        logits = _target_logits(model, tokens, target_hook, y)
        kl = kl_clean_to_current(clean_logits, logits)
        grad = torch.autograd.grad(kl, y, create_graph=False, retain_graph=False)[0]
    return logits.detach(), grad.detach(), float(kl.detach().item())


def kl_selective_jrr_logits(
    model,
    tokens: torch.Tensor,
    *,
    source_hook: str,
    target_hook: str,
    direction: torch.Tensor,
    alpha: float,
    beta: float,
    positive_only: bool,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact JRR teacher + adaptive one-dimensional harmful-mode selection."""
    zero = {
        "sel_remainder_norm": 0.0,
        "sel_orthogonal_norm": 0.0,
        "sel_selected_norm": 0.0,
        "sel_selected_fraction": 0.0,
        "sel_kl_gradient_norm": 0.0,
        "sel_kl_gradient_orth_norm": 0.0,
        "sel_alignment": 0.0,
        "sel_coeff": 0.0,
        "sel_kl_before": 0.0,
        "sel_kl_after": 0.0,
        "sel_transport_dot_removed": 0.0,
    }
    if alpha == 0.0:
        with torch.no_grad():
            return model(tokens)[:, -1, :], zero

    h = capture_source_last(model, tokens, source_hook)
    y0, transported_batch, _ = model_directional_jvp(
        model,
        tokens,
        source_hook=source_hook,
        target_hooks=[target_hook],
        source_value=h,
        direction=direction,
        cfg=cfg,
    )
    transported = transported_batch[0]

    with torch.no_grad():
        clean_logits = model(tokens)[:, -1, :]
        y_alpha = downstream_map(
            model,
            tokens,
            source_hook=source_hook,
            target_hooks=[target_hook],
            source_value=h + float(alpha) * direction,
        )[0]

    total = y_alpha - y0[0]
    remainder = total - float(alpha) * transported
    _, remainder_orth = decompose_remainder(remainder, transported)

    _, kl_grad, kl_before = kl_gradient_at_target(
        model,
        tokens,
        target_hook=target_hook,
        target_value=y_alpha,
        clean_logits=clean_logits,
    )
    selected, g_orth, coeff = select_kl_harmful_component(
        remainder_orth,
        transported,
        kl_grad,
        positive_only=positive_only,
    )
    removed = float(beta) * selected
    repaired = y_alpha - removed

    hooks = [
        (source_hook, additive_last_hook(direction, alpha)),
        (target_hook, replace_target_last_hook(repaired)),
    ]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        repaired_logits = model(tokens)[:, -1, :]
        kl_after = float(kl_clean_to_current(clean_logits, repaired_logits).item())

    rnorm = float(remainder_orth.norm().item())
    gnorm = float(g_orth.norm().item())
    alignment = 0.0
    if rnorm > 1e-12 and gnorm > 1e-12:
        alignment = float((remainder_orth * g_orth).sum().div(remainder_orth.norm() * g_orth.norm()).item())
    transport_dot = float((removed * transported).sum().abs().item())

    return repaired_logits, {
        "sel_remainder_norm": float(remainder.norm().item()),
        "sel_orthogonal_norm": rnorm,
        "sel_selected_norm": float(removed.norm().item()),
        "sel_selected_fraction": float(removed.norm().item()) / max(rnorm, 1e-12),
        "sel_kl_gradient_norm": float(kl_grad.norm().item()),
        "sel_kl_gradient_orth_norm": gnorm,
        "sel_alignment": alignment,
        "sel_coeff": float(coeff.item()),
        "sel_kl_before": kl_before,
        "sel_kl_after": kl_after,
        "sel_transport_dot_removed": transport_dot,
    }


def generate_one(
    model,
    prompt: str,
    *,
    method: str,
    source_hook: str,
    target_hook: str,
    direction: torch.Tensor,
    alpha: float,
    beta: float,
    positive_only: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    cfg: dict,
) -> dict:
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    prompt_len = int(tokens.shape[1])
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    eos_id = getattr(model.tokenizer, "eos_token_id", None)
    sums: dict[str, float] = {}
    steps = 0

    for _ in range(int(max_new_tokens)):
        if method == "additive":
            logits = additive_logits(
                model,
                tokens,
                source_hook=source_hook,
                direction=direction,
                alpha=alpha,
            )
            diag = {}
        elif method == "jrr_orth":
            logits, diag = oracle_jrr_logits(
                model,
                tokens,
                source_hook=source_hook,
                target_hook=target_hook,
                direction=direction,
                alpha=alpha,
                beta=beta,
                preserve_parallel=True,
                cfg=cfg,
            )
        elif method == "kl_jrr":
            logits, diag = kl_selective_jrr_logits(
                model,
                tokens,
                source_hook=source_hook,
                target_hook=target_hook,
                direction=direction,
                alpha=alpha,
                beta=beta,
                positive_only=positive_only,
                cfg=cfg,
            )
        else:
            raise ValueError(f"Unknown selective-JRR method: {method}")

        for key, value in diag.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        if diag:
            steps += 1

        next_token = sample_top_p(
            logits,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        tokens = torch.cat([tokens, next_token], dim=1)
        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    generated = tokens[0, prompt_len:]
    if eos_id is not None:
        hits = (generated == int(eos_id)).nonzero(as_tuple=False)
        if hits.numel():
            generated = generated[: int(hits[0].item())]
    eval_tokens = tokens[:, : prompt_len + int(generated.numel())]
    means = {key: value / max(1, steps) for key, value in sums.items()}
    means["sel_steps"] = float(steps)
    return {
        "tokens": eval_tokens,
        "prompt_len": prompt_len,
        "continuation": model.to_string(generated),
        **means,
    }


def _save_pareto(agg: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    for method, part in agg.groupby("method"):
        p = part.sort_values("strength")
        ax.plot(p.fluency_score, p.concept_score, marker="o", label=method)
        for row in p.itertuples():
            ax.annotate(str(row.strength), (row.fluency_score, row.concept_score), fontsize=7)
    ax.set_xlabel("Fluency score")
    ax.set_ylabel("Positive sentiment score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _same_alpha_table(agg: pd.DataFrame) -> pd.DataFrame:
    base = agg[agg.method == "additive"].set_index("strength")
    full = agg[agg.method == "jrr_orth"].set_index("strength") if "jrr_orth" in set(agg.method) else None
    rows = []
    for row in agg[agg.method == "kl_jrr"].itertuples():
        alpha = float(row.strength)
        if alpha not in base.index:
            continue
        b = base.loc[alpha]
        item = {
            "strength": alpha,
            "kl_concept": float(row.concept_score),
            "additive_concept": float(b.concept_score),
            "delta_concept_vs_additive": float(row.concept_score - b.concept_score),
            "kl_fluency": float(row.fluency_score),
            "additive_fluency": float(b.fluency_score),
            "delta_fluency_vs_additive": float(row.fluency_score - b.fluency_score),
            "delta_nll_vs_additive": float(row.nll - b.nll),
        }
        if full is not None and alpha in full.index:
            f = full.loc[alpha]
            item.update(
                {
                    "jrr_orth_concept": float(f.concept_score),
                    "concept_gain_vs_full_jrr": float(row.concept_score - f.concept_score),
                    "jrr_orth_fluency": float(f.fluency_score),
                    "fluency_gain_vs_full_jrr": float(row.fluency_score - f.fluency_score),
                }
            )
        rows.append(item)
    return pd.DataFrame(rows)


def _calibration_gate(delta: pd.DataFrame, cfg: dict) -> dict:
    ccfg = cfg["selective_jrr"]["calibration"]
    strong = {float(x) for x in ccfg.get("strong_strengths", [2.25, 3.0, 4.0])}
    min_f = float(ccfg.get("gate_min_fluency_gain", 5.0))
    max_c_loss = float(ccfg.get("gate_max_concept_loss", 5.0))
    part = delta[delta.strength.isin(strong)].copy()
    part["passes"] = (
        (part.delta_fluency_vs_additive >= min_f)
        & (part.delta_concept_vs_additive >= -max_c_loss)
    )
    passed = part[part.passes]
    return {
        "go_to_new_heldout": bool(len(passed) >= 1),
        "passing_strengths": [float(x) for x in passed.strength.tolist()],
        "required_min_fluency_gain": min_f,
        "allowed_max_concept_loss": max_c_loss,
        "strong_strengths": sorted(strong),
    }


def run_selective_phase(cfg: dict, *, phase: str, force: bool = False) -> dict:
    if phase not in {"calibration", "evaluation"}:
        raise ValueError("phase must be calibration or evaluation")

    scfg = cfg["selective_jrr"]
    pcfg = scfg[phase]
    out = Path(scfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if phase == "evaluation":
        cal_path = out / "calibration_summary.json"
        if not cal_path.exists():
            raise FileNotFoundError("Run selective-JRR calibration first")
        cal = json.loads(cal_path.read_text())
        if not cal.get("go_to_new_heldout", False) and not force:
            raise RuntimeError("Selective-JRR calibration gate failed; new held-out is locked")

    model = load_gpt2(cfg)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    direction, direction_meta = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    source_hook = str(scfg["source_hook"])
    target_hook = str(scfg["target_hook"])
    validate_hooks(model, source_hook, [target_hook])

    prompts = load_lines(pcfg["prompts_path"])[: int(pcfg.get("max_prompts", 999999))]
    strengths = [float(x) for x in pcfg["strengths"]]
    if 0.0 not in strengths:
        strengths = [0.0, *strengths]
    methods = list(pcfg.get("methods", ["additive", "kl_jrr"]))
    seeds = [int(x) for x in pcfg["seeds"]]
    beta = float(scfg.get("beta", 1.0))
    positive_only = bool(scfg.get("positive_only", True))

    judge = SentimentJudge(
        cfg["judge"]["model_name"],
        device=model.cfg.device,
        batch_size=int(cfg["judge"].get("batch_size", 32)),
    )

    rows: list[dict] = []
    output_csv = out / f"{phase}_samples.csv"
    jobs = [(m, a, s) for m in methods for a in strengths for s in seeds]
    for method, alpha, seed in tqdm(jobs, desc=f"KL-selective JRR {phase}", unit="job"):
        generated = [
            generate_one(
                model,
                prompt,
                method=method,
                source_hook=source_hook,
                target_hook=target_hook,
                direction=direction,
                alpha=alpha,
                beta=beta,
                positive_only=positive_only,
                max_new_tokens=int(pcfg.get("max_new_tokens", 32)),
                temperature=float(pcfg.get("temperature", 0.9)),
                top_p=float(pcfg.get("top_p", 0.95)),
                seed=seed + 100003 * prompt_id,
                cfg=cfg,
            )
            for prompt_id, prompt in enumerate(prompts)
        ]
        concepts = judge.score([item["continuation"] for item in generated])
        for prompt_id, (prompt, item, concept) in enumerate(zip(prompts, generated, concepts)):
            nll = continuation_nll(model, item["tokens"], item["prompt_len"])
            rows.append(
                {
                    "method": method,
                    "strength": alpha,
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "continuation": item["continuation"],
                    "concept_score": float(concept),
                    "nll": float(nll),
                    "ppl": float(math.exp(min(20.0, nll))),
                    **text_metrics(item["continuation"]),
                    **{k: v for k, v in item.items() if k.startswith("sel_")},
                    "target_hook": target_hook,
                    "beta": beta,
                    "positive_only": positive_only,
                    "direction_norm": float(direction.norm().item()),
                    "direction_sign": float(direction_meta.get("sign", 1.0)),
                }
            )
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    samples = pd.DataFrame(rows)
    agg = aggregate_behavior(samples)
    extra_cols = [c for c in samples.columns if c.startswith("sel_") and c != "sel_steps"]
    if extra_cols:
        extra = samples.groupby(["method", "strength"], as_index=False)[extra_cols].mean()
        agg = agg.merge(extra, on=["method", "strength"], how="left")
    agg.to_csv(out / f"{phase}_aggregate.csv", index=False)

    thresholds = [float(x) for x in scfg.get("concept_thresholds", [70, 75, 80])]
    frontier = interpolated_frontier_summary(agg, thresholds)
    frontier.to_csv(out / f"{phase}_frontier.csv", index=False)
    delta = _same_alpha_table(agg)
    delta.to_csv(out / f"{phase}_same_alpha.csv", index=False)
    _save_pareto(agg, out / f"{phase}_pareto.png", f"KL-selective JRR {phase}: {target_hook}")

    summary = {
        "phase": phase,
        "target_hook": target_hook,
        "beta": beta,
        "positive_only": positive_only,
        "methods": methods,
        "strengths": strengths,
        "n_prompts": len(prompts),
        "seeds": seeds,
        "frontier": frontier.to_dict(orient="records"),
    }
    if phase == "calibration":
        summary.update(_calibration_gate(delta, cfg))
    (out / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
