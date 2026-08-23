from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt2"
    device: str = "auto"
    dtype: str = "float32"


@dataclass(frozen=True)
class SAEConfig:
    location: str = "resid_post_mlp"
    layer: int = 8
    width: str = "128k"
    feature_id: int = 64840
    feature_label: str = "profanity_predictor_1"


@dataclass(frozen=True)
class SteeringConfig:
    hook_name: str = "blocks.8.hook_resid_post"
    strength_mode: str = "sae_alpha"
    strengths: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0)
    repair: str = "identity"


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 64
    temperature: float = 0.9
    top_p: float = 0.95
    seeds: tuple[int, ...] = (11, 23)


@dataclass(frozen=True)
class ExperimentConfig:
    prompts_path: str = "data/prompts.txt"
    calibration_prompts_path: str = "data/calibration_prompts.txt"
    concept_metric: str = "profanity_any_pct"
    output_csv: str = "results/baseline_samples.csv"
    save_every: int = 1
    min_concept_gain: float = 5.0


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    sae: SAEConfig
    steering: SteeringConfig
    sampling: SamplingConfig
    experiment: ExperimentConfig


def _tuple_floats(values: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in values)


def _tuple_ints(values: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    steering = dict(data.get("steering", {}))
    sampling = dict(data.get("sampling", {}))
    if "strengths" in steering:
        steering["strengths"] = _tuple_floats(steering["strengths"])
    if "seeds" in sampling:
        sampling["seeds"] = _tuple_ints(sampling["seeds"])

    cfg = Config(
        model=ModelConfig(**data.get("model", {})),
        sae=SAEConfig(**data.get("sae", {})),
        steering=SteeringConfig(**steering),
        sampling=SamplingConfig(**sampling),
        experiment=ExperimentConfig(**data.get("experiment", {})),
    )
    expected_hook = f"blocks.{cfg.sae.layer}.hook_resid_post"
    if cfg.steering.hook_name != expected_hook:
        raise ValueError(
            f"hook_name={cfg.steering.hook_name!r} does not match SAE layer {cfg.sae.layer}; "
            f"expected {expected_hook!r}."
        )
    if cfg.steering.strength_mode != "sae_alpha":
        raise ValueError(
            "The reproducible baseline must use strength_mode='sae_alpha': alpha is defined "
            "in the normalized coordinate system of the OpenAI v5 SAE."
        )
    if 0.0 not in cfg.steering.strengths:
        raise ValueError("The baseline sweep must include alpha=0.")
    return cfg
