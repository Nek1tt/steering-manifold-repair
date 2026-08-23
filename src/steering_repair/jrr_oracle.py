from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

from .inference_followups import interpolated_frontier_summary
from .jrr import (
    additive_last_hook,
    apply_jrr_repair,
    capture_source_last,
    decompose_remainder,
    downstream_map,
    model_directional_jvp,
    replace_target_last_hook,
    validate_hooks,
)
from .jrr_diagnostic import aggregate_behavior
from .metrics import continuation_nll, text_metrics
from .sentiment_baseline import SentimentJudge, load_direction, load_gpt2, load_lines


@dataclass
class OracleStepDiagnostics:
    remainder_norm: float = 0.0
    orthogonal_norm: float = 0.0
    parallel_norm: float = 0.0
    removed_norm: float = 0.0
    transported_norm: float = 0.0
    total_delta_norm: float = 0.0
    steps: int = 0

    def add(self, diag: dict[str, float]) -> None:
        self.remainder_norm += diag["jrr_remainder_norm"]
        self.orthogonal_norm += diag["jrr_orthogonal_norm"]
        self.parallel_norm += diag["jrr_parallel_norm"]
        self.removed_norm += diag["jrr_removed_norm"]
        self.transported_norm += diag["jrr_transported_norm"]
        self.total_delta_norm += diag["jrr_total_delta_norm"]
        self.steps += 1

    def mean_dict(self) -> dict[str, float]:
        n = max(1, self.steps)
        return {
            "jrr_remainder_norm": self.remainder_norm / n,
            "jrr_orthogonal_norm": self.orthogonal_norm / n,
            "jrr_parallel_norm": self.parallel_norm / n,
            "jrr_removed_norm": self.removed_norm / n,
            "jrr_transported_norm": self.transported_norm / n,
            "jrr_total_delta_norm": self.total_delta_norm / n,
            "jrr_steps": float(self.steps),
        }


def resolve_target_hook(cfg: dict) -> str:
    target = str(cfg["jrr"]["oracle"].get("target_hook", "auto"))
    if target != "auto":
        return target
    path = Path(cfg["jrr"]["output_dir"]) / "diagnostic_summary.json"
    if not path.exists():
        raise FileNotFoundError("Run JRR diagnostic first")
    return str(json.loads(path.read_text())["recommended_target_hook"])


def additive_logits(
    model,
    tokens: torch.Tensor,
    *,
    source_hook: str,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha == 0.0:
        with torch.no_grad():
            return model(tokens)[:, -1, :]
    hooks = [(source_hook, additive_last_hook(direction, alpha))]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        return model(tokens)[:, -1, :]


def oracle_jrr_logits(
    model,
    tokens: torch.Tensor,
    *,
    source_hook: str,
    target_hook: str,
    direction: torch.Tensor,
    alpha: float,
    beta: float,
    preserve_parallel: bool,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = {
        "jrr_remainder_norm": 0.0,
        "jrr_orthogonal_norm": 0.0,
        "jrr_parallel_norm": 0.0,
        "jrr_removed_norm": 0.0,
        "jrr_transported_norm": 0.0,
        "jrr_total_delta_norm": 0.0,
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
        y_alpha = downstream_map(
            model,
            tokens,
            source_hook=source_hook,
            target_hooks=[target_hook],
            source_value=h + alpha * direction,
        )[0]
    total = y_alpha - y0[0]
    remainder = total - alpha * transported
    parallel, orthogonal = decompose_remainder(remainder, transported)
    repaired, removed = apply_jrr_repair(
        y_alpha,
        remainder,
        transported,
        beta=beta,
        preserve_parallel=preserve_parallel,
    )
    hooks = [
        (source_hook, additive_last_hook(direction, alpha)),
        (target_hook, replace_target_last_hook(repaired)),
    ]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        logits = model(tokens)[:, -1, :]
    return logits, {
        "jrr_remainder_norm": float(remainder.norm().item()),
        "jrr_orthogonal_norm": float(orthogonal.norm().item()),
        "jrr_parallel_norm": float(parallel.norm().item()),
        "jrr_removed_norm": float(removed.norm().item()),
        "jrr_transported_norm": float(transported.norm().item()),
        "jrr_total_delta_norm": float(total.norm().item()),
    }


def sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits / max(float(temperature), 1e-6)
    sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1, generator=generator)
    return sorted_indices.gather(-1, sampled)


def generate_manual_one(
    model,
    prompt: str,
    *,
    method: str,
    source_hook: str,
    target_hook: str,
    direction: torch.Tensor,
    alpha: float,
    beta: float,
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
    diag = OracleStepDiagnostics()

    for _ in range(int(max_new_tokens)):
        if method == "additive":
            logits = additive_logits(
                model,
                tokens,
                source_hook=source_hook,
                direction=direction,
                alpha=alpha,
            )
        elif method in {"jrr_orth", "jrr_full"}:
            logits, step = oracle_jrr_logits(
                model,
                tokens,
                source_hook=source_hook,
                target_hook=target_hook,
                direction=direction,
                alpha=alpha,
                beta=beta,
                preserve_parallel=(method == "jrr_orth"),
                cfg=cfg,
            )
            diag.add(step)
        else:
            raise ValueError(f"Unknown method: {method}")
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
    return {
        "tokens": eval_tokens,
        "prompt_len": prompt_len,
        "continuation": model.to_string(generated),
        **diag.mean_dict(),
    }


def save_pareto(agg: pd.DataFrame, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    for method, part in agg.groupby("method"):
        p = part.sort_values("strength")
        ax.plot(p.fluency_score, p.concept_score, marker="o", label=method)
        for row in p.itertuples():
            ax.annotate(
                f"{row.strength:g}",
                (row.fluency_score, row.concept_score),
                fontsize=7,
            )
    ax.set_xlabel("Fluency score")
    ax.set_ylabel("Positive sentiment score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def oracle_gate(
    frontier: pd.DataFrame, thresholds: list[float], min_gain: float
) -> dict:
    additive = frontier[frontier.method == "additive"]
    best_method = None
    best_gain = -float("inf")
    gains = {}
    for method, row_df in frontier[frontier.method != "additive"].groupby("method"):
        row = row_df.iloc[0]
        method_gains = {}
        local_best = -float("inf")
        for threshold in thresholds:
            col = f"fluency_at_concept_{int(threshold)}"
            a = (
                float(additive.iloc[0][col])
                if not additive.empty and pd.notna(additive.iloc[0][col])
                else float("nan")
            )
            b = float(row[col]) if pd.notna(row[col]) else float("nan")
            gain = b - a if math.isfinite(a) and math.isfinite(b) else float("nan")
            method_gains[f"C{int(threshold)}"] = (
                gain if math.isfinite(gain) else None
            )
            if math.isfinite(gain):
                local_best = max(local_best, gain)
        gains[method] = method_gains
        if local_best > best_gain:
            best_gain = local_best
            best_method = method
    return {
        "go_to_heldout": bool(
            best_method is not None and best_gain >= float(min_gain)
        ),
        "selected_method": best_method,
        "best_calibration_fluency_gain": best_gain
        if math.isfinite(best_gain)
        else None,
        "required_gain": float(min_gain),
        "gains_by_threshold": gains,
    }


def run_oracle_phase(cfg: dict, *, phase: str, force: bool = False) -> dict:
    if phase not in {"calibration", "evaluation"}:
        raise ValueError("phase must be calibration or evaluation")
    out = Path(cfg["jrr"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    diag_path = out / "diagnostic_summary.json"
    if not diag_path.exists():
        raise FileNotFoundError("Run scripts/run_jrr_diagnostic.py first")
    diagnostic = json.loads(diag_path.read_text())
    if (
        phase == "calibration"
        and not diagnostic.get("oracle_recommended", False)
        and not force
    ):
        raise RuntimeError(
            "Diagnostic gate is negative. Inspect DIAGNOSTIC.md; use --force only for a deliberate ablation."
        )

    ocfg = cfg["jrr"]["oracle"]
    pcfg = ocfg[phase]
    methods = list(ocfg.get("methods", ["additive", "jrr_orth"]))
    if phase == "evaluation":
        cal_path = out / "oracle_calibration_summary.json"
        if not cal_path.exists():
            raise FileNotFoundError("Run oracle calibration first")
        cal = json.loads(cal_path.read_text())
        if not cal.get("go_to_heldout", False) and not force:
            raise RuntimeError(
                "Calibration gate failed; held-out evaluation is intentionally locked"
            )
        selected = cal.get("selected_method")
        methods = ["additive", selected] if selected else ["additive", "jrr_orth"]

    model = load_gpt2(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    direction, meta = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    source_hook = cfg["jrr"]["source_hook"]
    target_hook = resolve_target_hook(cfg)
    validate_hooks(model, source_hook, [target_hook])
    prompts = load_lines(pcfg["prompts_path"])[: int(pcfg.get("max_prompts", 999999))]
    strengths = [float(x) for x in pcfg["strengths"]]
    if 0.0 not in strengths:
        strengths = [0.0, *strengths]
    seeds = [int(x) for x in pcfg["seeds"]]
    beta = float(ocfg.get("beta", 1.0))
    judge = SentimentJudge(
        cfg["judge"]["model_name"],
        device=model.cfg.device,
        batch_size=int(cfg["judge"].get("batch_size", 32)),
    )

    rows: list[dict] = []
    output_csv = out / f"oracle_{phase}_samples.csv"
    jobs = [(m, a, s) for m in methods for a in strengths for s in seeds]
    for method, alpha, seed in tqdm(jobs, desc=f"JRR oracle {phase}", unit="job"):
        generated = [
            generate_manual_one(
                model,
                prompt,
                method=method,
                source_hook=source_hook,
                target_hook=target_hook,
                direction=direction,
                alpha=alpha,
                beta=beta,
                max_new_tokens=int(pcfg.get("max_new_tokens", 24)),
                temperature=float(pcfg.get("temperature", 0.9)),
                top_p=float(pcfg.get("top_p", 0.95)),
                seed=seed + 100003 * prompt_id,
                cfg=cfg,
            )
            for prompt_id, prompt in enumerate(prompts)
        ]
        concepts = judge.score([x["continuation"] for x in generated])
        for prompt_id, (prompt, item, concept) in enumerate(
            zip(prompts, generated, concepts)
        ):
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
                    **{k: v for k, v in item.items() if k.startswith("jrr_")},
                    "target_hook": target_hook,
                    "beta": beta,
                    "direction_norm": float(direction.norm().item()),
                    "direction_sign": float(meta.get("sign", 1.0)),
                }
            )
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    samples = pd.DataFrame(rows)
    agg = aggregate_behavior(samples)
    agg.to_csv(out / f"oracle_{phase}_aggregate.csv", index=False)
    thresholds = [float(x) for x in ocfg.get("concept_thresholds", [80, 85, 90])]
    frontier = interpolated_frontier_summary(agg, thresholds)
    frontier.to_csv(out / f"oracle_{phase}_frontier.csv", index=False)
    save_pareto(
        agg,
        out / f"oracle_{phase}_pareto.png",
        f"JRR oracle {phase}: {target_hook}",
    )

    if phase == "calibration":
        summary = oracle_gate(
            frontier,
            thresholds,
            min_gain=float(ocfg.get("gate_min_fluency_gain", 2.0)),
        )
        summary.update(
            {
                "target_hook": target_hook,
                "beta": beta,
                "methods": methods,
                "strengths": strengths,
                "n_prompts": len(prompts),
                "seeds": seeds,
            }
        )
        (out / "oracle_calibration_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        (out / "ORACLE_CALIBRATION.md").write_text(
            "\n".join(
                [
                    "# JRR oracle calibration",
                    "",
                    f"Target hook: `{target_hook}`",
                    f"Proceed to held-out evaluation: **{summary['go_to_heldout']}**",
                    f"Selected method: `{summary['selected_method']}`",
                    f"Best calibration fluency gain: {summary['best_calibration_fluency_gain']}",
                    "",
                    frontier.to_markdown(index=False),
                    "",
                    "Calibration prompts only. Held-out evaluation stays locked unless this gate passes.",
                ]
            )
        )
    else:
        summary = {
            "target_hook": target_hook,
            "beta": beta,
            "methods": methods,
            "strengths": strengths,
            "n_prompts": len(prompts),
            "seeds": seeds,
            "frontier": frontier.to_dict(orient="records"),
        }
        (out / "oracle_evaluation_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    return summary
