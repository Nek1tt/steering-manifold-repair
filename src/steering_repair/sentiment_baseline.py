from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from .generation import generate_batch
from .metrics import continuation_nll, text_metrics


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_lines(path: str | Path) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines in {path}")
    return lines


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gpt2(cfg: dict):
    from transformer_lens import HookedTransformer

    model_cfg = cfg["model"]
    device = resolve_device(model_cfg.get("device", "auto"))
    dtype = getattr(torch, model_cfg.get("dtype", "float32"))
    model = HookedTransformer.from_pretrained(
        model_cfg.get("name", "gpt2"),
        device=device,
        dtype=dtype,
        center_writing_weights=False,
    )
    model.eval()
    return model


@torch.no_grad()
def pooled_last_activation(
    model,
    text: str,
    *,
    hook_name: str,
    pool_last_n: int = 4,
) -> torch.Tensor:
    """Pool the last few token activations for one text, avoiding padding artifacts."""
    tokens = model.to_tokens(text, prepend_bos=True).to(model.cfg.device)
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    acts = cache[hook_name][0]
    content = acts[1:] if acts.shape[0] > 1 else acts
    n = max(1, min(int(pool_last_n), int(content.shape[0])))
    return content[-n:].mean(dim=0).detach()


@torch.no_grad()
def build_contrastive_direction(model, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Build v = E[h|positive] - E[h|negative] at the midpoint layer."""
    vcfg = cfg["vector"]
    positives = load_lines(vcfg["positive_path"])
    negatives = load_lines(vcfg["negative_path"])
    hook_name = vcfg["hook_name"]
    pool_last_n = int(vcfg.get("pool_last_n", 4))

    pos_acts = torch.stack(
        [
            pooled_last_activation(
                model, text, hook_name=hook_name, pool_last_n=pool_last_n
            )
            for text in tqdm(positives, desc="positive vector examples", leave=False)
        ]
    )
    neg_acts = torch.stack(
        [
            pooled_last_activation(
                model, text, hook_name=hook_name, pool_last_n=pool_last_n
            )
            for text in tqdm(negatives, desc="negative vector examples", leave=False)
        ]
    )

    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    direction = pos_mean - neg_mean
    if not torch.isfinite(direction).all() or direction.norm().item() < 1e-8:
        raise RuntimeError("Contrastive sentiment direction is invalid or near-zero")

    unit = direction / direction.norm().clamp_min(1e-12)
    pos_projection = float((pos_acts @ unit).mean().item())
    neg_projection = float((neg_acts @ unit).mean().item())
    diagnostics = {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "direction_norm": float(direction.norm().item()),
        "positive_projection_mean": pos_projection,
        "negative_projection_mean": neg_projection,
        "projection_gap": pos_projection - neg_projection,
    }
    return direction.detach(), diagnostics


def save_direction(path: str | Path, direction: torch.Tensor, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction.detach().cpu(), "metadata": metadata}, path)


def load_direction(path: str | Path, device: str | torch.device) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "direction" not in payload:
        raise ValueError(f"Invalid direction cache: {path}")
    direction = payload["direction"].to(device)
    return direction, dict(payload.get("metadata", {}))


class SentimentJudge:
    """Independent local text-level judge returning P(positive) in [0, 100]."""

    def __init__(self, model_name: str, device: str, batch_size: int = 32) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = int(batch_size)

        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        positive = [idx for idx, label in id2label.items() if "pos" in label.lower()]
        self.positive_idx = positive[0] if positive else max(id2label)

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [
                text if text.strip() else " "
                for text in texts[start : start + self.batch_size]
            ]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, self.positive_idx]
            scores.extend((100.0 * probs).detach().cpu().tolist())
        return [float(x) for x in scores]


def _make_judge(cfg: dict, model_device: str) -> SentimentJudge:
    jcfg = cfg["judge"]
    return SentimentJudge(
        jcfg["model_name"],
        device=model_device,
        batch_size=int(jcfg.get("batch_size", 32)),
    )


def _run_generation_batch(
    model,
    prompts: list[str],
    direction: torch.Tensor,
    cfg: dict,
    *,
    strength: float,
    seed: int,
    max_new_tokens: int | None = None,
):
    scfg = cfg["sampling"]
    return generate_batch(
        model,
        prompts,
        hook_name=cfg["vector"]["hook_name"],
        direction=direction,
        strength=float(strength),
        strength_mode=cfg["steering"].get("strength_mode", "vector_alpha"),
        repair=cfg["steering"].get("repair", "identity"),
        max_new_tokens=int(max_new_tokens or scfg.get("max_new_tokens", 64)),
        temperature=float(scfg.get("temperature", 0.9)),
        top_p=float(scfg.get("top_p", 0.95)),
        seed=int(seed),
    )


def quick_sentiment_scores(
    model,
    judge: SentimentJudge,
    prompts: list[str],
    direction: torch.Tensor,
    cfg: dict,
    *,
    strengths: Iterable[float],
    seed: int,
    max_new_tokens: int = 48,
) -> list[dict]:
    rows: list[dict] = []
    for strength in strengths:
        generated = _run_generation_batch(
            model,
            prompts,
            direction,
            cfg,
            strength=float(strength),
            seed=int(seed),
            max_new_tokens=max_new_tokens,
        )
        scores = judge.score([item["continuation"] for item in generated])
        rows.append(
            {
                "strength": float(strength),
                "sentiment_score": float(sum(scores) / max(1, len(scores))),
            }
        )
    return rows


def _grid_gain(rows: list[dict]) -> tuple[float, dict, float]:
    base = next(row["sentiment_score"] for row in rows if row["strength"] == 0.0)
    best = max(rows, key=lambda row: row["sentiment_score"])
    return float(best["sentiment_score"] - base), best, float(base)


def validate_sentiment_direction(cfg: dict) -> dict:
    """Choose the causal sign on held-out prompts and require measurable text-level gain."""
    model = load_gpt2(cfg)
    raw_direction, vector_diag = build_contrastive_direction(model, cfg)
    prompts = load_lines(cfg["experiment"]["calibration_prompts_path"])
    judge = _make_judge(cfg, model.cfg.device)
    seed = int(cfg["sampling"]["seeds"][0])
    strengths = [float(x) for x in cfg["steering"]["strengths"]]
    max_new_tokens = min(48, int(cfg["sampling"].get("max_new_tokens", 64)))

    # We deliberately calibrate sign on a separate prompt set. The full alpha
    # grid is cheap for GPT-2 Small and avoids choosing a sign from a noisy
    # single coefficient.
    plus_rows = quick_sentiment_scores(
        model,
        judge,
        prompts,
        raw_direction,
        cfg,
        strengths=strengths,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    minus_rows = quick_sentiment_scores(
        model,
        judge,
        prompts,
        -raw_direction,
        cfg,
        strengths=strengths,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    plus_gain, plus_best, plus_base = _grid_gain(plus_rows)
    minus_gain, minus_best, minus_base = _grid_gain(minus_rows)

    if plus_gain >= minus_gain:
        sign = 1.0
        direction = raw_direction
        rows = plus_rows
        gain = plus_gain
        best = plus_best
        base = plus_base
    else:
        sign = -1.0
        direction = -raw_direction
        rows = minus_rows
        gain = minus_gain
        best = minus_best
        base = minus_base

    threshold = float(cfg["experiment"].get("min_concept_gain", 8.0))
    passed = gain >= threshold and best["strength"] > 0

    metadata = {
        **vector_diag,
        "sign": sign,
        "plus_grid_gain": float(plus_gain),
        "minus_grid_gain": float(minus_gain),
        "plus_best_strength": float(plus_best["strength"]),
        "minus_best_strength": float(minus_best["strength"]),
        "validation_best_strength": float(best["strength"]),
        "validation_concept_gain": float(gain),
    }
    save_direction(cfg["vector"]["cache_path"], direction, metadata)

    result = {
        "passed": passed,
        "base_sentiment": float(base),
        "best_sentiment": float(best["sentiment_score"]),
        "best_strength": float(best["strength"]),
        "concept_gain": float(gain),
        "rows": rows,
        "metadata": metadata,
    }
    validation_path = Path("results/sentiment_validation.json")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(result, indent=2))
    return result


def run_sentiment_baseline(cfg: dict) -> pd.DataFrame:
    model = load_gpt2(cfg)
    cache_path = Path(cfg["vector"]["cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing calibrated direction {cache_path}. "
            "Run scripts/validate_sentiment_baseline.py first."
        )
    direction, metadata = load_direction(cache_path, model.cfg.device)
    prompts = load_lines(cfg["experiment"]["prompts_path"])
    judge = _make_judge(cfg, model.cfg.device)
    strengths = [float(x) for x in cfg["steering"]["strengths"]]
    seeds = [int(x) for x in cfg["sampling"]["seeds"]]

    output_path = Path(cfg["experiment"]["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    jobs = [(strength, seed) for strength in strengths for seed in seeds]
    pbar = tqdm(jobs, desc="sentiment baseline batches", unit="batch")
    for job_idx, (strength, seed) in enumerate(pbar, start=1):
        pbar.set_postfix(alpha=f"{strength:g}", seed=seed)
        generated = _run_generation_batch(
            model, prompts, direction, cfg, strength=strength, seed=seed
        )
        continuations = [item["continuation"] for item in generated]
        concept_scores = judge.score(continuations)

        for prompt_idx, (prompt, item, concept_score) in enumerate(
            zip(prompts, generated, concept_scores)
        ):
            nll = continuation_nll(model, item["tokens"], item["prompt_len"])
            tm = text_metrics(item["continuation"])
            rows.append(
                {
                    "model": cfg["model"].get("name", "gpt2"),
                    "layer": int(cfg["vector"]["layer"]),
                    "vector_type": cfg["vector"].get(
                        "type", "contrastive_mean_difference"
                    ),
                    "direction_norm": float(direction.norm().item()),
                    "direction_sign": float(metadata.get("sign", 1.0)),
                    "method": cfg["steering"].get("repair", "identity"),
                    "strength": float(strength),
                    "seed": int(seed),
                    "prompt_id": int(prompt_idx),
                    "prompt": prompt,
                    "continuation": item["continuation"],
                    "concept_score": float(concept_score),
                    "nll": float(nll),
                    "ppl": float(math.exp(min(20.0, nll))),
                    **tm,
                }
            )

        save_every = int(cfg["experiment"].get("save_every", 1))
        if save_every > 0 and job_idx % save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "nll",
        "ppl",
        "concept_score",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_3gram",
    ]
    out = df.groupby(["method", "strength"], as_index=False)[numeric].mean()
    if not (out["strength"] == 0).any():
        raise ValueError("Baseline results must contain alpha=0")
    base = out.loc[out["strength"] == 0].iloc[0]
    base_nll = float(base["nll"])
    base_d3 = max(float(base["distinct_3"]), 1e-8)
    base_rep_quality = max(1.0 - float(base["repetition_3gram"]), 1e-8)

    def fluency(row) -> float:
        # Each factor is anchored to alpha=0 and capped at 1. This prevents a
        # collapsed/repetitive continuation from looking "more fluent" merely
        # because its clean-model NLL happens to decrease.
        nll_factor = math.exp(-max(0.0, float(row.nll) - base_nll))
        diversity_factor = min(1.0, max(0.0, float(row.distinct_3) / base_d3))
        rep_quality = max(0.0, 1.0 - float(row.repetition_3gram))
        repetition_factor = min(1.0, rep_quality / base_rep_quality)
        return 100.0 * nll_factor * diversity_factor * repetition_factor

    out["fluency_score"] = [fluency(row) for row in out.itertuples()]
    return out


def sentiment_baseline_check(agg: pd.DataFrame, cfg: dict) -> dict:
    base = agg.loc[agg["strength"] == 0].iloc[0]
    best = agg.loc[agg["concept_score"].idxmax()]
    concept_gain = float(best["concept_score"] - base["concept_score"])
    concept_threshold = float(cfg["experiment"].get("min_concept_gain", 8.0))
    fluency_threshold = float(cfg["experiment"].get("min_fluency_drop", 5.0))

    higher_concept = agg[
        agg["concept_score"] >= base["concept_score"] + concept_threshold / 2.0
    ]
    if higher_concept.empty:
        max_fluency_drop = 0.0
    else:
        max_fluency_drop = float(
            base["fluency_score"] - higher_concept["fluency_score"].min()
        )

    return {
        "passed": concept_gain >= concept_threshold
        and max_fluency_drop >= fluency_threshold,
        "concept_gain": concept_gain,
        "best_strength": float(best["strength"]),
        "best_concept_score": float(best["concept_score"]),
        "fluency_at_best_concept": float(best["fluency_score"]),
        "max_fluency_drop_with_concept": max_fluency_drop,
    }


def plot_sentiment_pareto(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, dict]:
    agg = aggregate_sentiment(df)
    check = sentiment_baseline_check(agg, cfg)
    output = Path(cfg["experiment"]["output_plot"])
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for method, part in agg.groupby("method"):
        part = part.sort_values("strength")
        ax.plot(
            part["fluency_score"],
            part["concept_score"],
            marker="o",
            label=method,
        )
        for row in part.itertuples():
            ax.annotate(
                f"{row.strength:g}",
                (row.fluency_score, row.concept_score),
                fontsize=8,
            )
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 100)
    ax.set_xlabel(
        "Fluency score ↑  (clean-NLL × distinct-3 × anti-repetition)"
    )
    ax.set_ylabel("Positive sentiment score ↑  (local independent judge)")
    ax.set_title("GPT-2 midpoint contrastive steering Pareto baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return agg, check
