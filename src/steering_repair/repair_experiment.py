from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

from .metrics import continuation_nll, text_metrics
from .sentiment_baseline import SentimentJudge, load_direction, load_gpt2, load_lines
from .train_denoiser import load_denoiser_checkpoint


@dataclass
class RepairDiagnostics:
    count: int = 0
    ratio_sum: float = 0.0
    correction_cos_sum: float = 0.0
    correction_parallel_fraction_sum: float = 0.0
    correction_to_steer_norm_sum: float = 0.0
    effective_alpha_sum: float = 0.0

    def add(
        self,
        *,
        ratio: torch.Tensor,
        correction: torch.Tensor,
        steering_delta: torch.Tensor,
        repaired_delta: torch.Tensor,
        raw_direction: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            v = raw_direction.to(device=correction.device, dtype=correction.dtype)
            v_unit = v / v.norm().clamp_min(1e-12)
            corr_norm = correction.norm(dim=-1).clamp_min(1e-12)
            parallel = correction @ v_unit
            cos = parallel / corr_norm
            parallel_fraction = parallel.abs() / corr_norm
            steer_norm = steering_delta.norm(dim=-1).clamp_min(1e-12)
            correction_to_steer = correction.norm(dim=-1) / steer_norm
            vv = v.square().sum().clamp_min(1e-12)
            effective_alpha = (repaired_delta * v).sum(dim=-1) / vv
            n = int(correction.shape[0])
            self.count += n
            self.ratio_sum += float(ratio.sum().item())
            self.correction_cos_sum += float(cos.sum().item())
            self.correction_parallel_fraction_sum += float(parallel_fraction.sum().item())
            self.correction_to_steer_norm_sum += float(correction_to_steer.sum().item())
            self.effective_alpha_sum += float(effective_alpha.sum().item())

    def mean_dict(self) -> dict[str, float]:
        d = max(1, self.count)
        return {
            "noise_ratio": self.ratio_sum / d,
            "correction_cos_v": self.correction_cos_sum / d,
            "correction_parallel_fraction": self.correction_parallel_fraction_sum / d,
            "correction_to_steer_norm": self.correction_to_steer_norm_sum / d,
            "effective_alpha": self.effective_alpha_sum / d,
        }


class DenoiserSteeringHook:
    """Apply additive steering followed by an optional learned repair.

    parallel_keep=1 is vanilla denoising. parallel_keep=0 is DPAR, which
    discards the correction component parallel to the intended steering vector.
    """

    def __init__(
        self,
        *,
        raw_direction: torch.Tensor,
        alpha: float,
        denoiser=None,
        parallel_keep: float = 1.0,
        norm_preserving: bool = False,
    ) -> None:
        self.raw_direction = raw_direction.detach()
        self.alpha = float(alpha)
        self.denoiser = denoiser
        self.parallel_keep = float(parallel_keep)
        self.norm_preserving = bool(norm_preserving)
        self.diagnostics = RepairDiagnostics()

    def __call__(self, resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        if self.alpha == 0.0:
            return resid
        clean = resid[:, -1, :]
        v = self.raw_direction.to(device=clean.device, dtype=clean.dtype)
        steering_delta = self.alpha * v.unsqueeze(0).expand_as(clean)
        z = clean + steering_delta
        ratio = steering_delta.norm(dim=-1) / clean.norm(dim=-1).clamp_min(1e-8)

        if self.norm_preserving:
            repaired = z * (
                clean.norm(dim=-1, keepdim=True)
                / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            )
        elif self.denoiser is None:
            repaired = z
        else:
            denoised = self.denoiser(z, ratio)
            raw_correction = denoised - z
            v_unit = v / v.norm().clamp_min(1e-12)
            parallel = (raw_correction @ v_unit)[:, None] * v_unit[None, :]
            correction = raw_correction - (1.0 - self.parallel_keep) * parallel
            repaired = z + correction

        correction = repaired - z
        self.diagnostics.add(
            ratio=ratio,
            correction=correction,
            steering_delta=steering_delta,
            repaired_delta=repaired - clean,
            raw_direction=v,
        )
        out = resid.clone()
        out[:, -1, :] = repaired
        return out


def _trim_generated(tokens: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    if eos_id is None or tokens.numel() == 0:
        return tokens
    hits = (tokens == int(eos_id)).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return tokens
    return tokens[: int(hits[0].item())]


@torch.no_grad()
def generate_with_hook(
    model,
    prompts: list[str],
    *,
    hook_name: str,
    hook_obj,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict]:
    padded = model.to_tokens(prompts, prepend_bos=True, padding_side="left").to(
        model.cfg.device
    )
    input_width = padded.shape[1]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ctx = nullcontext() if hook_obj.alpha == 0 else model.hooks(fwd_hooks=[(hook_name, hook_obj)])
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
    rows = []
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


def _method_specs(cfg: dict, loaded: dict[str, object]) -> dict[str, dict]:
    specs = {
        "additive": {"denoiser": None, "parallel_keep": 1.0},
        "norm_preserving": {"denoiser": None, "parallel_keep": 1.0, "norm_preserving": True},
    }
    if "gaussian" in loaded:
        specs.update(
            {
                "gaussian": {"denoiser": loaded["gaussian"], "parallel_keep": 1.0},
                "gaussian_lambda05": {"denoiser": loaded["gaussian"], "parallel_keep": 0.5},
                "gaussian_dpar": {"denoiser": loaded["gaussian"], "parallel_keep": 0.0},
            }
        )
    if "mixed" in loaded:
        specs.update(
            {
                "mixed": {"denoiser": loaded["mixed"], "parallel_keep": 1.0},
                "mixed_dpar": {"denoiser": loaded["mixed"], "parallel_keep": 0.0},
            }
        )
    requested = cfg["evaluation"].get("methods")
    if requested:
        missing = [name for name in requested if name not in specs]
        if missing:
            raise ValueError(f"Requested repair methods unavailable: {missing}")
        specs = {name: specs[name] for name in requested}
    return specs


def run_repair_evaluation(cfg: dict) -> pd.DataFrame:
    model = load_gpt2(cfg)
    direction, direction_meta = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    prompts = load_lines(cfg["evaluation"]["prompts_path"])
    judge_cfg = cfg["judge"]
    judge = SentimentJudge(
        judge_cfg["model_name"],
        device=model.cfg.device,
        batch_size=int(judge_cfg.get("batch_size", 32)),
    )

    loaded: dict[str, object] = {}
    checkpoints = cfg["denoisers"]
    for kind in ("gaussian", "mixed"):
        path = Path(checkpoints[kind]["checkpoint"])
        if path.exists():
            denoiser, _ = load_denoiser_checkpoint(path, device=model.cfg.device)
            loaded[kind] = denoiser
        elif checkpoints[kind].get("required", True):
            raise FileNotFoundError(f"Missing {kind} checkpoint: {path}")
    specs = _method_specs(cfg, loaded)

    ecfg = cfg["evaluation"]
    strengths = [float(x) for x in ecfg["strengths"]]
    seeds = [int(x) for x in ecfg["seeds"]]
    output_path = Path(ecfg["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(method, alpha, seed) for method in specs for alpha in strengths for seed in seeds]
    rows: list[dict] = []
    pbar = tqdm(jobs, desc="repair evaluation batches", unit="batch")
    for job_idx, (method, alpha, seed) in enumerate(pbar, start=1):
        pbar.set_postfix(method=method, alpha=f"{alpha:g}", seed=seed)
        spec = specs[method]
        hook = DenoiserSteeringHook(
            raw_direction=direction,
            alpha=alpha,
            denoiser=spec.get("denoiser"),
            parallel_keep=float(spec.get("parallel_keep", 1.0)),
            norm_preserving=bool(spec.get("norm_preserving", False)),
        )
        generated = generate_with_hook(
            model,
            prompts,
            hook_name=cfg["vector"]["hook_name"],
            hook_obj=hook,
            max_new_tokens=int(ecfg.get("max_new_tokens", 64)),
            temperature=float(ecfg.get("temperature", 0.9)),
            top_p=float(ecfg.get("top_p", 0.95)),
            seed=seed,
        )
        concept_scores = judge.score([x["continuation"] for x in generated])
        diag = hook.diagnostics.mean_dict()
        if alpha == 0.0:
            diag["effective_alpha"] = 0.0

        for prompt_idx, (prompt, item, concept) in enumerate(zip(prompts, generated, concept_scores)):
            nll = continuation_nll(model, item["tokens"], item["prompt_len"])
            tm = text_metrics(item["continuation"])
            rows.append(
                {
                    "method": method,
                    "strength": alpha,
                    "seed": seed,
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "continuation": item["continuation"],
                    "concept_score": float(concept),
                    "nll": float(nll),
                    "ppl": float(math.exp(min(20.0, nll))),
                    **tm,
                    **diag,
                    "direction_norm": float(direction.norm().item()),
                    "direction_sign": float(direction_meta.get("sign", 1.0)),
                }
            )
        if int(ecfg.get("save_every", 1)) > 0 and job_idx % int(ecfg.get("save_every", 1)) == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def aggregate_repairs(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "nll",
        "ppl",
        "concept_score",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_3gram",
        "noise_ratio",
        "correction_cos_v",
        "correction_parallel_fraction",
        "correction_to_steer_norm",
        "effective_alpha",
    ]
    out = df.groupby(["method", "strength"], as_index=False)[numeric].mean()
    base_rows = out[(out["method"] == "additive") & (out["strength"] == 0)]
    if base_rows.empty:
        raise ValueError("Need additive alpha=0 as the common fluency anchor")
    base = base_rows.iloc[0]
    base_nll = float(base.nll)
    base_d3 = max(float(base.distinct_3), 1e-8)
    base_rep_quality = max(1.0 - float(base.repetition_3gram), 1e-8)

    fluencies = []
    for row in out.itertuples():
        nll_factor = math.exp(-max(0.0, float(row.nll) - base_nll))
        diversity_factor = min(1.0, max(0.0, float(row.distinct_3) / base_d3))
        rep_quality = max(0.0, 1.0 - float(row.repetition_3gram))
        repetition_factor = min(1.0, rep_quality / base_rep_quality)
        fluencies.append(100.0 * nll_factor * diversity_factor * repetition_factor)
    out["fluency_score"] = fluencies
    out["alpha_preservation_error"] = (out["effective_alpha"] - out["strength"]).abs()
    return out


def frontier_summary(agg: pd.DataFrame, thresholds=(70.0, 80.0, 90.0, 95.0)) -> pd.DataFrame:
    rows = []
    for method, part in agg.groupby("method"):
        row = {"method": method}
        for threshold in thresholds:
            eligible = part[part["concept_score"] >= threshold]
            row[f"max_fluency_at_concept_{int(threshold)}"] = (
                float(eligible["fluency_score"].max()) if not eligible.empty else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_repair_suite(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = aggregate_repairs(df)
    output_dir = Path(cfg["evaluation"].get("output_dir", "results/repair_suite"))
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for method, part in agg.groupby("method"):
        part = part.sort_values("strength")
        ax.plot(part["fluency_score"], part["concept_score"], marker="o", label=method)
        for row in part.itertuples():
            ax.annotate(f"{row.strength:g}", (row.fluency_score, row.concept_score), fontsize=7)
    ax.set_xlabel("Fluency score ↑")
    ax.set_ylabel("Positive sentiment score ↑")
    ax.set_title("Steering repair Pareto comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "repair_pareto.png", dpi=180)
    plt.close(fig)

    learned = agg[agg["method"].str.contains("gaussian|mixed", regex=True)].copy()
    if not learned.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for method, part in learned.groupby("method"):
            part = part.sort_values("strength")
            ax.plot(part["strength"], part["effective_alpha"], marker="o", label=method)
        lo = float(learned["strength"].min())
        hi = float(learned["strength"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", label="perfect alpha preservation")
        ax.set_xlabel("Requested alpha")
        ax.set_ylabel("Effective alpha after repair")
        ax.set_title("Does the repair cancel the intended steering direction?")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "effective_alpha.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for method, part in learned.groupby("method"):
            part = part.sort_values("strength")
            ax.plot(part["strength"], part["correction_cos_v"], marker="o", label=method)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("alpha")
        ax.set_ylabel("cos(repair correction, steering vector)")
        ax.set_title("Correction geometry")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "correction_geometry.png", dpi=180)
        plt.close(fig)

    summary = frontier_summary(agg)
    agg.to_csv(output_dir / "repair_aggregate.csv", index=False)
    summary.to_csv(output_dir / "frontier_summary.csv", index=False)
    return agg, summary


def write_hypothesis_report(agg: pd.DataFrame, summary: pd.DataFrame, path: str | Path) -> dict:
    methods = set(agg["method"])
    conclusions: dict[str, dict] = {}

    def _frontier(method: str, threshold: int) -> float:
        row = summary[summary["method"] == method]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][f"max_fluency_at_concept_{threshold}"])

    if {"additive", "gaussian"} <= methods:
        gains = []
        for threshold in (80, 90):
            a, g = _frontier("additive", threshold), _frontier("gaussian", threshold)
            if math.isfinite(a) and math.isfinite(g):
                gains.append(g - a)
        conclusions["H1_gaussian_repair_improves_frontier"] = {
            "supported": bool(gains and max(gains) > 1.0),
            "best_fluency_gain_points": max(gains) if gains else None,
        }

    if {"gaussian", "gaussian_dpar"} <= methods:
        vanilla = agg[(agg.method == "gaussian") & (agg.strength > 0)]
        dpar = agg[(agg.method == "gaussian_dpar") & (agg.strength > 0)]
        conclusions["H2_dpar_preserves_steering_better"] = {
            "supported": bool(
                not vanilla.empty
                and not dpar.empty
                and dpar.alpha_preservation_error.mean() + 1e-6 < vanilla.alpha_preservation_error.mean()
            ),
            "vanilla_mean_alpha_error": float(vanilla.alpha_preservation_error.mean()) if not vanilla.empty else None,
            "dpar_mean_alpha_error": float(dpar.alpha_preservation_error.mean()) if not dpar.empty else None,
        }

    if {"gaussian_dpar", "mixed_dpar"} <= methods:
        gains = []
        for threshold in (80, 90, 95):
            g, m = _frontier("gaussian_dpar", threshold), _frontier("mixed_dpar", threshold)
            if math.isfinite(g) and math.isfinite(m):
                gains.append(m - g)
        conclusions["H3_structured_training_improves_dpar"] = {
            "supported": bool(gains and max(gains) > 1.0),
            "best_fluency_gain_points": max(gains) if gains else None,
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Repair hypothesis check", ""]
    for name, result in conclusions.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Supported by this run: **{result.get('supported')}**")
        for key, value in result.items():
            if key != "supported":
                lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("These checks are descriptive diagnostics for the current run, not statistical significance tests.")
    path.write_text("\n".join(lines))
    path.with_suffix(".json").write_text(json.dumps(conclusions, indent=2))
    return conclusions
