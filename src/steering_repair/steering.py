from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


RepairFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def identity_repair(clean: torch.Tensor, steered: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    del clean, direction
    return steered


def norm_preserving_repair(clean: torch.Tensor, steered: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    del direction
    target = clean.norm(dim=-1, keepdim=True)
    current = steered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return steered * (target / current)


def get_repair(name: str) -> RepairFn:
    repairs: dict[str, RepairFn] = {
        "identity": identity_repair,
        "norm_preserving": norm_preserving_repair,
    }
    try:
        return repairs[name]
    except KeyError as exc:
        raise ValueError(f"Unknown repair {name!r}. Available: {sorted(repairs)}") from exc


@dataclass
class SteeringHook:
    """Steer only the final sequence position, matching response-token steering."""

    direction: torch.Tensor
    strength: float
    strength_mode: str = "sae_alpha"
    repair: str = "identity"

    def __post_init__(self) -> None:
        self.direction = self.direction / self.direction.norm().clamp_min(1e-12)
        self._repair_fn = get_repair(self.repair)

    def _delta(self, clean_last: torch.Tensor) -> torch.Tensor:
        v = self.direction.to(device=clean_last.device, dtype=clean_last.dtype)
        v = v.unsqueeze(0).expand_as(clean_last)
        if self.strength_mode == "norm_ratio":
            return self.strength * clean_last.norm(dim=-1, keepdim=True) * v
        if self.strength_mode == "raw":
            return self.strength * v
        if self.strength_mode == "sae_alpha":
            # OpenAI v5 TopK SAEs normalize each token vector by its scalar std.
            # h_norm -> h_norm + alpha*v therefore maps back to
            # h -> h + alpha*std(h)*v in the raw residual stream.
            return self.strength * clean_last.std(dim=-1, keepdim=True) * v
        raise ValueError(f"Unknown strength_mode={self.strength_mode!r}")

    def apply_last(self, clean_last: torch.Tensor) -> torch.Tensor:
        if self.strength == 0.0:
            return clean_last
        steered_last = clean_last + self._delta(clean_last)
        return self._repair_fn(clean_last, steered_last, self.direction)

    def __call__(self, resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        if self.strength == 0.0:
            return resid
        out = resid.clone()
        out[:, -1, :] = self.apply_last(out[:, -1, :])
        return out
