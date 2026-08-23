from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .jrr import (
    capture_source_last,
    cosine,
    decompose_remainder,
    downstream_map,
    model_directional_jvp,
    validate_hooks,
)
from .metrics import continuation_nll, text_metrics
from .repair_experiment import DenoiserSteeringHook, generate_with_hook
from .sentiment_baseline import SentimentJudge, load_direction, load_gpt2, load_lines


def measure_prompt_remainders(
    model,
    *,
    prompt: str,
    prompt_id: int,
    direction: torch.Tensor,
    source_hook: str,
    target_hooks: list[str],
    strengths: list[float],
    cfg: dict,
) -> tuple[list[dict], str]:
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    h = capture_source_last(model, tokens, source_hook)
    y0, jvp, jvp_mode = model_directional_jvp(
        model,
        tokens,
        source_hook=source_hook,
        target_hooks=target_hooks,
        source_value=h,
        direction=direction,
        cfg=cfg,
    )
    rows: list[dict] = []
    for alpha in strengths:
        alpha = float(alpha)
        with torch.no_grad():
            y_alpha = downstream_map(
                model,
                tokens,
                source_hook=source_hook,
                target_hooks=target_hooks,
                source_value=h + alpha * direction,
            )
        total = y_alpha - y0
        linear = alpha * jvp
        remainder = total - linear
        for i, target_hook in enumerate(target_hooks):
            r = remainder[i]
            t = jvp[i]
            parallel, orthogonal = decompose_remainder(r, t)
            total_norm = float(total[i].norm().item())
            r_norm = float(r.norm().item())
            o_norm = float(orthogonal.norm().item())
            p_norm = float(parallel.norm().item())
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "target_hook": target_hook,
                    "strength": alpha,
                    "jvp_mode": jvp_mode,
                    "transported_norm": float(t.norm().item()),
                    "total_delta_norm": total_norm,
                    "linear_delta_norm": float(linear[i].norm().item()),
                    "residual_norm": r_norm,
                    "residual_fraction": r_norm / max(total_norm, 1e-12),
                    "residual_over_alpha2": r_norm / (alpha * alpha)
                    if alpha != 0
                    else float("nan"),
                    "residual_parallel_norm": p_norm,
                    "residual_orthogonal_norm": o_norm,
                    "residual_orthogonal_fraction": o_norm / max(r_norm, 1e-12),
                    "residual_parallel_fraction": p_norm / max(r_norm, 1e-12),
                    "cos_total_transport": cosine(total[i], t),
                    "cos_residual_transport": cosine(r, t),
                }
            )
    return rows, jvp_mode


def aggregate_behavior(samples: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "nll",
        "ppl",
        "concept_score",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_3gram",
    ]
    out = samples.groupby(["method", "strength"], as_index=False)[numeric].mean()
    base_rows = out[(out.method == "additive") & (out.strength == 0.0)]
    if base_rows.empty:
        raise ValueError("Need additive alpha=0 as fluency anchor")
    base = base_rows.iloc[0]
    base_nll = float(base.nll)
    base_d3 = max(float(base.distinct_3), 1e-8)
    base_rep = max(1.0 - float(base.repetition_3gram), 1e-8)
    fluency = []
    for row in out.itertuples():
        nll_factor = math.exp(-max(0.0, float(row.nll) - base_nll))
        div_factor = min(1.0, max(0.0, float(row.distinct_3) / base_d3))
        rep_quality = max(0.0, 1.0 - float(row.repetition_3gram))
        rep_factor = min(1.0, rep_quality / base_rep)
        fluency.append(100.0 * nll_factor * div_factor * rep_factor)
    out["fluency_score"] = fluency
    return out


def run_additive_behavior_probe(
    model, direction: torch.Tensor, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jcfg = cfg["jrr"]
    dcfg = jcfg["diagnostic"]
    prompts = load_lines(dcfg["prompts_path"])[: int(dcfg.get("max_prompts", 8))]
    strengths = [float(x) for x in dcfg["strengths"]]
    seed = int(dcfg.get("behavior_seed", 37))
    judge = SentimentJudge(
        cfg["judge"]["model_name"],
        device=model.cfg.device,
        batch_size=int(cfg["judge"].get("batch_size", 32)),
    )
    rows: list[dict] = []
    for alpha in tqdm(strengths, desc="JRR behavior probe", unit="alpha"):
        hook = DenoiserSteeringHook(
            raw_direction=direction, alpha=alpha, denoiser=None
        )
        generated = generate_with_hook(
            model,
            prompts,
            hook_name=jcfg["source_hook"],
            hook_obj=hook,
            max_new_tokens=int(dcfg.get("behavior_max_new_tokens", 32)),
            temperature=float(dcfg.get("temperature", 0.9)),
            top_p=float(dcfg.get("top_p", 0.95)),
            seed=seed,
        )
        concepts = judge.score([x["continuation"] for x in generated])
        for prompt_id, (prompt, item, concept) in enumerate(
            zip(prompts, generated, concepts)
        ):
            nll = continuation_nll(model, item["tokens"], item["prompt_len"])
            rows.append(
                {
                    "method": "additive",
                    "strength": alpha,
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "continuation": item["continuation"],
                    "concept_score": float(concept),
                    "nll": float(nll),
                    "ppl": float(math.exp(min(20.0, nll))),
                    **text_metrics(item["continuation"]),
                }
            )
    samples = pd.DataFrame(rows)
    return samples, aggregate_behavior(samples)


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid.x.rank().corr(valid.y.rank()))


def _loglog_slope(part: pd.DataFrame, column: str) -> float:
    p = (
        part[(part.strength > 0) & (part[column] > 0)]
        .groupby("strength", as_index=False)[column]
        .mean()
    )
    if len(p) < 3:
        return float("nan")
    return float(
        np.polyfit(np.log(p.strength.to_numpy()), np.log(p[column].to_numpy()), 1)[0]
    )


def summarize_diagnostic(
    aggregate: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    gate_corr: float,
    gate_orthogonal_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for target, part in aggregate.groupby("target_hook"):
        merged = part.merge(
            behavior[["strength", "nll", "fluency_score"]], on="strength"
        )
        positive = merged[merged.strength > 0]
        slope = _loglog_slope(positive, "residual_norm")
        corr_nll = _rank_corr(positive.residual_orthogonal_norm, positive.nll)
        corr_fluency = _rank_corr(
            positive.residual_orthogonal_norm, positive.fluency_score
        )
        mean_orth = float(positive.residual_orthogonal_fraction.mean())
        scaling = (
            max(0.0, 1.0 - abs(slope - 2.0) / 1.5)
            if math.isfinite(slope)
            else 0.0
        )
        score = (
            max(0.0, corr_nll if math.isfinite(corr_nll) else 0.0)
            + max(0.0, -corr_fluency if math.isfinite(corr_fluency) else 0.0)
            + 0.5 * scaling
            + 0.25 * mean_orth
        )
        rows.append(
            {
                "target_hook": target,
                "loglog_residual_slope": slope,
                "rank_corr_orthogonal_residual_vs_nll": corr_nll,
                "rank_corr_orthogonal_residual_vs_fluency": corr_fluency,
                "mean_orthogonal_fraction": mean_orth,
                "mechanistic_score": score,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "mechanistic_score", ascending=False
    ).reset_index(drop=True)
    best = table.iloc[0]
    corr_signal = max(
        float(best.rank_corr_orthogonal_residual_vs_nll)
        if pd.notna(best.rank_corr_orthogonal_residual_vs_nll)
        else -1.0,
        -float(best.rank_corr_orthogonal_residual_vs_fluency)
        if pd.notna(best.rank_corr_orthogonal_residual_vs_fluency)
        else -1.0,
    )
    go = bool(
        corr_signal >= gate_corr
        and float(best.mean_orthogonal_fraction) >= gate_orthogonal_fraction
    )
    return table, {
        "recommended_target_hook": str(best.target_hook),
        "oracle_recommended": go,
        "correlation_signal": corr_signal,
        "loglog_residual_slope": float(best.loglog_residual_slope),
        "mean_orthogonal_fraction": float(best.mean_orthogonal_fraction),
    }


def save_plots(
    aggregate: pd.DataFrame,
    behavior: pd.DataFrame,
    output_dir: Path,
    recommended: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for target, part in aggregate.groupby("target_hook"):
        p = part[part.strength > 0].sort_values("strength")
        ax.loglog(p.strength, p.residual_norm, marker="o", label=target)
    ax.set_xlabel("alpha")
    ax.set_ylabel("||nonlinear remainder||")
    ax.set_title("JRR diagnostic: Taylor remainder scaling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "remainder_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for target, part in aggregate.groupby("target_hook"):
        p = part[part.strength > 0].sort_values("strength")
        ax.plot(
            p.strength,
            p.residual_orthogonal_fraction,
            marker="o",
            label=target,
        )
    ax.set_xlabel("alpha")
    ax.set_ylabel("orthogonal fraction of nonlinear remainder")
    ax.set_title("Collateral fraction of downstream nonlinearity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "orthogonal_remainder_fraction.png", dpi=180)
    plt.close(fig)

    part = aggregate[aggregate.target_hook == recommended].merge(
        behavior[["strength", "fluency_score"]], on="strength"
    )
    part = part[part.strength > 0]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(part.residual_orthogonal_norm, part.fluency_score)
    for row in part.itertuples():
        ax.annotate(
            f"a={row.strength:g}",
            (row.residual_orthogonal_norm, row.fluency_score),
            fontsize=8,
        )
    ax.set_xlabel("||orthogonal nonlinear remainder||")
    ax.set_ylabel("additive fluency score")
    ax.set_title(f"Mechanistic link at {recommended}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "orthogonal_remainder_vs_fluency.png", dpi=180)
    plt.close(fig)


def run_jrr_diagnostic(cfg: dict) -> dict:
    output_dir = Path(cfg["jrr"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_gpt2(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    direction, meta = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    source_hook = cfg["jrr"]["source_hook"]
    dcfg = cfg["jrr"]["diagnostic"]
    target_hooks = list(dcfg["target_hooks"])
    validate_hooks(model, source_hook, target_hooks)
    prompts = load_lines(dcfg["prompts_path"])[: int(dcfg.get("max_prompts", 8))]
    strengths = [float(x) for x in dcfg["strengths"]]
    if 0.0 not in strengths:
        strengths = [0.0, *strengths]

    rows: list[dict] = []
    modes = []
    for prompt_id, prompt in enumerate(
        tqdm(prompts, desc="JRR nonlinear diagnostic", unit="prompt")
    ):
        new_rows, mode = measure_prompt_remainders(
            model,
            prompt=prompt,
            prompt_id=prompt_id,
            direction=direction,
            source_hook=source_hook,
            target_hooks=target_hooks,
            strengths=strengths,
            cfg=cfg,
        )
        rows.extend(new_rows)
        modes.append(mode)
        pd.DataFrame(rows).to_csv(output_dir / "diagnostic_samples.csv", index=False)

    samples = pd.DataFrame(rows)
    numeric = [
        "transported_norm",
        "total_delta_norm",
        "linear_delta_norm",
        "residual_norm",
        "residual_fraction",
        "residual_over_alpha2",
        "residual_parallel_norm",
        "residual_orthogonal_norm",
        "residual_orthogonal_fraction",
        "residual_parallel_fraction",
        "cos_total_transport",
        "cos_residual_transport",
    ]
    aggregate = samples.groupby(["target_hook", "strength"], as_index=False)[numeric].mean()
    aggregate.to_csv(output_dir / "diagnostic_aggregate.csv", index=False)
    behavior_samples, behavior = run_additive_behavior_probe(model, direction, cfg)
    behavior_samples.to_csv(output_dir / "behavior_samples.csv", index=False)
    behavior.to_csv(output_dir / "behavior_aggregate.csv", index=False)

    table, summary = summarize_diagnostic(
        aggregate,
        behavior,
        gate_corr=float(dcfg.get("gate_min_rank_correlation", 0.4)),
        gate_orthogonal_fraction=float(
            dcfg.get("gate_min_orthogonal_fraction", 0.2)
        ),
    )
    table.to_csv(output_dir / "target_layer_summary.csv", index=False)
    summary.update(
        {
            "source_hook": source_hook,
            "target_hooks": target_hooks,
            "n_prompts": len(prompts),
            "strengths": strengths,
            "direction_norm": float(direction.norm().item()),
            "direction_sign": float(meta.get("sign", 1.0)),
            "jvp_modes_used": sorted(set(modes)),
        }
    )
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# JRR diagnostic summary",
        "",
        f"Recommended downstream hook: `{summary['recommended_target_hook']}`",
        f"Proceed to oracle repair: **{summary['oracle_recommended']}**",
        "",
        table.to_markdown(index=False),
        "",
        "Interpretation: a slope near 2 supports second-order growth; positive residual-vs-NLL or negative residual-vs-fluency correlation supports the degradation hypothesis.",
        "",
        "Do not touch held-out prompts until oracle calibration passes.",
    ]
    (output_dir / "DIAGNOSTIC.md").write_text("\n".join(lines))
    save_plots(aggregate, behavior, output_dir, summary["recommended_target_hook"])
    return summary
