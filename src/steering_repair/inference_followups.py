from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

from .metrics import continuation_nll, text_metrics
from .repair_experiment import RepairDiagnostics, aggregate_repairs, generate_with_hook
from .sentiment_baseline import SentimentJudge, load_direction, load_gpt2, load_lines
from .train_denoiser import load_denoiser_checkpoint


@dataclass(frozen=True)
class ScaledRepairSpec:
    name: str
    checkpoint_kind: str | None
    correction_scale: float
    parallel_keep: float


class ScaledDenoiserSteeringHook:
    """Add steering and apply a scaled, optionally direction-projected correction.

    beta=correction_scale controls correction magnitude. parallel_keep=1 is
    vanilla denoising; parallel_keep=0 is DPAR.  The two knobs are intentionally
    separated so we can test whether the previous DPAR failure came from its
    geometry or simply from applying too much MSE-trained correction.
    """

    def __init__(
        self,
        *,
        raw_direction: torch.Tensor,
        alpha: float,
        denoiser=None,
        correction_scale: float = 1.0,
        parallel_keep: float = 1.0,
    ) -> None:
        self.raw_direction = raw_direction.detach()
        self.alpha = float(alpha)
        self.denoiser = denoiser
        self.correction_scale = float(correction_scale)
        self.parallel_keep = float(parallel_keep)
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

        if self.denoiser is None or self.correction_scale == 0.0:
            repaired = z
        else:
            denoised = self.denoiser(z, ratio)
            raw_correction = denoised - z
            v_unit = v / v.norm().clamp_min(1e-12)
            parallel = (raw_correction @ v_unit)[:, None] * v_unit[None, :]
            filtered = raw_correction - (1.0 - self.parallel_keep) * parallel
            repaired = z + self.correction_scale * filtered

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


def _beta_tag(beta: float) -> str:
    return f"b{int(round(beta * 100)):03d}"


def build_calibration_specs(cfg: dict) -> list[ScaledRepairSpec]:
    fcfg = cfg["followup"]
    specs = [ScaledRepairSpec("additive", None, 0.0, 1.0)]

    for beta in fcfg["gaussian_vanilla_betas"]:
        beta = float(beta)
        specs.append(
            ScaledRepairSpec(
                f"gaussian_vanilla_{_beta_tag(beta)}", "gaussian", beta, 1.0
            )
        )
    for beta in fcfg["gaussian_dpar_betas"]:
        beta = float(beta)
        specs.append(
            ScaledRepairSpec(
                f"gaussian_dpar_{_beta_tag(beta)}", "gaussian", beta, 0.0
            )
        )
    for beta in fcfg.get("mixed_dpar_betas", []):
        beta = float(beta)
        specs.append(
            ScaledRepairSpec(
                f"mixed_dpar_{_beta_tag(beta)}", "mixed", beta, 0.0
            )
        )
    return specs


def selected_specs(cfg: dict, selection: dict) -> list[ScaledRepairSpec]:
    # Always retain the assignment-proposed full Gaussian denoiser and full
    # Gaussian DPAR as controls, in addition to calibration-selected scaled
    # variants. Deduplication keeps the run compact when beta=1 wins.
    requested = [
        ScaledRepairSpec("additive", None, 0.0, 1.0),
        ScaledRepairSpec("gaussian_vanilla_b100", "gaussian", 1.0, 1.0),
        ScaledRepairSpec("gaussian_dpar_b100", "gaussian", 1.0, 0.0),
    ]
    for family, checkpoint_kind, parallel_keep in (
        ("gaussian_vanilla", "gaussian", 1.0),
        ("gaussian_dpar", "gaussian", 0.0),
        ("mixed_dpar", "mixed", 0.0),
    ):
        row = selection.get(family)
        if row is None:
            continue
        beta = float(row["beta"])
        requested.append(
            ScaledRepairSpec(
                f"{family}_{_beta_tag(beta)}", checkpoint_kind, beta, parallel_keep
            )
        )

    seen = set()
    out = []
    for spec in requested:
        if spec.name not in seen:
            seen.add(spec.name)
            out.append(spec)
    return out


def _load_checkpoints(cfg: dict, specs: Iterable[ScaledRepairSpec], device) -> dict[str, object]:
    needed = {s.checkpoint_kind for s in specs if s.checkpoint_kind is not None}
    loaded: dict[str, object] = {}
    for kind in sorted(needed):
        path = Path(cfg["denoisers"][kind]["checkpoint"])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing preserved {kind} checkpoint: {path}. This follow-up is "
                "inference-only and will not retrain it."
            )
        model, _ = load_denoiser_checkpoint(path, device=device)
        loaded[kind] = model
    return loaded


def run_scaled_evaluation(
    cfg: dict,
    *,
    specs: list[ScaledRepairSpec],
    phase_cfg: dict,
    output_csv: str | Path,
) -> pd.DataFrame:
    model = load_gpt2(cfg)
    direction, direction_meta = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    prompts = load_lines(phase_cfg["prompts_path"])
    judge = SentimentJudge(
        cfg["judge"]["model_name"],
        device=model.cfg.device,
        batch_size=int(cfg["judge"].get("batch_size", 32)),
    )
    loaded = _load_checkpoints(cfg, specs, model.cfg.device)

    strengths = [float(x) for x in phase_cfg["strengths"]]
    seeds = [int(x) for x in phase_cfg["seeds"]]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(spec, alpha, seed) for spec in specs for alpha in strengths for seed in seeds]
    rows: list[dict] = []
    pbar = tqdm(jobs, desc="inference-only repair sweep", unit="batch")
    for job_idx, (spec, alpha, seed) in enumerate(pbar, start=1):
        pbar.set_postfix(method=spec.name, alpha=f"{alpha:g}", seed=seed)
        hook = ScaledDenoiserSteeringHook(
            raw_direction=direction,
            alpha=alpha,
            denoiser=loaded.get(spec.checkpoint_kind),
            correction_scale=spec.correction_scale,
            parallel_keep=spec.parallel_keep,
        )
        generated = generate_with_hook(
            model,
            prompts,
            hook_name=cfg["vector"]["hook_name"],
            hook_obj=hook,
            max_new_tokens=int(phase_cfg.get("max_new_tokens", 64)),
            temperature=float(phase_cfg.get("temperature", 0.9)),
            top_p=float(phase_cfg.get("top_p", 0.95)),
            seed=seed,
        )
        concept_scores = judge.score([x["continuation"] for x in generated])
        diag = hook.diagnostics.mean_dict()
        if alpha == 0.0:
            diag["effective_alpha"] = 0.0

        for prompt_idx, (prompt, item, concept) in enumerate(
            zip(prompts, generated, concept_scores)
        ):
            nll = continuation_nll(model, item["tokens"], item["prompt_len"])
            tm = text_metrics(item["continuation"])
            rows.append(
                {
                    "method": spec.name,
                    "checkpoint_kind": spec.checkpoint_kind or "none",
                    "correction_scale": spec.correction_scale,
                    "parallel_keep": spec.parallel_keep,
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
        if job_idx % int(phase_cfg.get("save_every", 1)) == 0:
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def interpolated_fluency_at_threshold(part: pd.DataFrame, threshold: float) -> float:
    """Best fluency obtainable at concept >= threshold, with alpha interpolation.

    We include linear crossings between adjacent alpha points. This specifically
    addresses the coarse-alpha artifact observed in the first repair suite.
    """
    part = part.sort_values("strength")
    best = float("nan")
    eligible = part[part["concept_score"] >= threshold]
    if not eligible.empty:
        best = float(eligible["fluency_score"].max())

    rows = list(part.itertuples())
    for a, b in zip(rows[:-1], rows[1:]):
        ca, cb = float(a.concept_score), float(b.concept_score)
        if (ca - threshold) * (cb - threshold) > 0 or ca == cb:
            continue
        t = (threshold - ca) / (cb - ca)
        if 0.0 <= t <= 1.0:
            f = float(a.fluency_score) + t * (
                float(b.fluency_score) - float(a.fluency_score)
            )
            best = f if math.isnan(best) else max(best, f)
    return best


def interpolated_frontier_summary(
    agg: pd.DataFrame, thresholds: Iterable[float]
) -> pd.DataFrame:
    rows = []
    for method, part in agg.groupby("method"):
        row = {"method": method}
        for threshold in thresholds:
            row[f"fluency_at_concept_{int(threshold)}"] = interpolated_fluency_at_threshold(
                part, float(threshold)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_selection(
    agg: pd.DataFrame, *, thresholds: list[float]
) -> tuple[dict, pd.DataFrame]:
    frontier = interpolated_frontier_summary(agg, thresholds)
    rows = []
    for row in frontier.itertuples(index=False):
        method = row.method
        if method == "additive":
            continue
        vals = [
            getattr(row, f"fluency_at_concept_{int(t)}") for t in thresholds
        ]
        finite = [float(x) for x in vals if pd.notna(x)]
        score = sum(finite) / len(thresholds) if finite else -1.0
        # Unreached thresholds receive zero rather than disappearing from the
        # denominator, so a weak-concept method cannot win by covering one easy
        # threshold only.
        beta = float(
            agg.loc[agg["method"] == method, "correction_scale"].iloc[0]
        )
        family = method.rsplit("_b", 1)[0]
        rows.append({"family": family, "method": method, "beta": beta, "score": score, **{f"F@C{int(t)}": vals[i] for i, t in enumerate(thresholds)}})

    scores = pd.DataFrame(rows)
    selection: dict[str, dict] = {}
    for family, part in scores.groupby("family"):
        winner = part.sort_values(["score", "beta"], ascending=[False, True]).iloc[0]
        selection[family] = {
            "method": str(winner.method),
            "beta": float(winner.beta),
            "calibration_score": float(winner.score),
        }
    return selection, scores


def _friendly(name: str) -> str:
    return (
        name.replace("gaussian_vanilla_", "Gaussian ")
        .replace("gaussian_dpar_", "DPAR ")
        .replace("mixed_dpar_", "Mixed DPAR ")
        .replace("b", "β=")
        .replace("025", "0.25")
        .replace("010", "0.10")
        .replace("050", "0.50")
        .replace("075", "0.75")
        .replace("100", "1.00")
    )


def save_followup_plots(
    calibration_agg: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    eval_agg: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for family, part in calibration_scores.groupby("family"):
        part = part.sort_values("beta")
        ax.plot(part["beta"], part["score"], marker="o", label=family)
    ax.set_xlabel("Correction scale β")
    ax.set_ylabel("Calibration frontier score ↑")
    ax.set_title("How much denoiser correction should be applied?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "beta_calibration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    for method, part in eval_agg.groupby("method"):
        part = part.sort_values("strength")
        ax.plot(
            part["fluency_score"],
            part["concept_score"],
            marker="o",
            label=_friendly(method),
        )
        for row in part.itertuples():
            ax.annotate(f"{row.strength:g}", (row.fluency_score, row.concept_score), fontsize=7)
    ax.set_xlabel("Fluency score ↑")
    ax.set_ylabel("Positive sentiment score ↑")
    ax.set_title("Dense inference-only steering repair comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_dense_pareto.png", dpi=180)
    plt.close(fig)

    learned = eval_agg[eval_agg["method"] != "additive"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for method, part in learned.groupby("method"):
        part = part.sort_values("strength")
        ax.plot(part["strength"], part["effective_alpha"], marker="o", label=_friendly(method))
    lo = float(eval_agg["strength"].min())
    hi = float(eval_agg["strength"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", label="perfect preservation")
    ax.set_xlabel("Requested alpha")
    ax.set_ylabel("Effective alpha after repair")
    ax.set_title("Scaled repair: steering preservation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "selected_effective_alpha.png", dpi=180)
    plt.close(fig)


def write_summary(
    *,
    selection: dict,
    frontier: pd.DataFrame,
    thresholds: list[float],
    path: str | Path,
) -> None:
    path = Path(path)
    lines = [
        "# Inference-only follow-up results",
        "",
        "No denoiser weights were updated in this experiment. Existing Gaussian and mixed checkpoints were reused.",
        "",
        "## Calibration-selected correction scales",
        "",
    ]
    for family, row in selection.items():
        lines.append(
            f"- **{family}**: beta={row['beta']:.2f} (calibration score={row['calibration_score']:.3f})"
        )
    lines += ["", "## Held-out dense frontier", "", frontier.to_markdown(index=False), ""]

    if "additive" in set(frontier["method"]):
        base = frontier.set_index("method").loc["additive"]
        lines += ["## Descriptive gains over additive", ""]
        for method, row in frontier.set_index("method").iterrows():
            if method == "additive":
                continue
            gains = []
            for threshold in thresholds:
                col = f"fluency_at_concept_{int(threshold)}"
                a, b = base[col], row[col]
                if pd.notna(a) and pd.notna(b):
                    gains.append(f"C>={int(threshold)}: {float(b-a):+.2f}")
            if gains:
                lines.append(f"- **{method}** — " + ", ".join(gains))
        lines.append("")
    path.write_text("\n".join(lines))


def save_selection(selection: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2))


def load_selection(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
