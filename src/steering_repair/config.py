from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt2-small"
    device: str = "auto"
    dtype: str = "float32"


@dataclass(frozen=True)
class SAEConfig:
    location: str = "resid_post_mlp"
    layer: int = 8
    width: str = "128k"
    feature_id: int = 56907
    feature_label: str = "words_in_quotes"


@dataclass(frozen=True)
class SteeringConfig:
    hook_name: str = "blocks.8.hook_resid_post"
    strength_mode: str = "norm_ratio"
    strengths: tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0)
    repair: str = "identity"


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.9
    top_p: float = 0.95
    seeds: tuple[int, ...] = (11, 23, 37)


@dataclass(frozen=True)
class ExperimentConfig:
    prompts_path: str = "data/prompts.txt"
    output_csv: str = "results/baseline_samples.csv"
    save_every: int = 5


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
            f"expected {expected_hook!r} for resid_post_mlp."
        )
    return cfg
