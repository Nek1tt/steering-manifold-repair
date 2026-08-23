from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


RepairFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def identity_repair(clean: torch.Tensor, steered: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    del clean, direction
    return steered


def norm_preserving_repair(clean: torch.Tensor, steered: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Cheap control: keep the original activation L2 norm after steering."""
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
    """TransformerLens forward hook for steering the final sequence position only."""

    direction: torch.Tensor
    strength: float
    strength_mode: str = "norm_ratio"
    repair: str = "identity"

    def __post_init__(self) -> None:
        self.direction = self.direction / self.direction.norm().clamp_min(1e-12)
        self._repair_fn = get_repair(self.repair)

    def _delta(self, clean_last: torch.Tensor) -> torch.Tensor:
        v = self.direction.to(device=clean_last.device, dtype=clean_last.dtype)
        v = v.unsqueeze(0).expand_as(clean_last)

        if self.strength_mode == "norm_ratio":
            # ||delta|| = strength * ||h|| for each item in the batch.
            scale = self.strength * clean_last.norm(dim=-1, keepdim=True)
            return scale * v
        if self.strength_mode == "raw":
            return self.strength * v
        if self.strength_mode == "sae_alpha":
            # OpenAI v5 SAEs normalize each activation vector by its scalar std.
            # A decoder-space alpha therefore maps back to raw space by multiplying by std(h).
            scale = self.strength * clean_last.std(dim=-1, keepdim=True)
            return scale * v
        raise ValueError(f"Unknown strength_mode={self.strength_mode!r}")

    def __call__(self, resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        if self.strength == 0.0:
            return resid
        out = resid.clone()
        clean_last = out[:, -1, :]
        steered_last = clean_last + self._delta(clean_last)
        repaired_last = self._repair_fn(clean_last, steered_last, self.direction)
        out[:, -1, :] = repaired_last
        return out
