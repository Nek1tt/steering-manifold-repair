from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualActivationDenoiser(nn.Module):
    """Small residual MLP denoiser conditioned on relative corruption magnitude.

    The model predicts a repair residual and returns ``z + residual``. The last
    projection is zero-initialized so the network starts as the identity map.
    The predicted residual is rescaled by the per-sample RMS of ``z`` so the
    network can generalize across natural activation-scale variation even though
    its content pathway is layer-normalized.
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 1536) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.in_norm = nn.LayerNorm(self.d_model)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, noise_ratio: torch.Tensor) -> torch.Tensor:
        if noise_ratio.ndim == 0:
            noise_ratio = noise_ratio.expand(z.shape[0])
        if noise_ratio.ndim == 1:
            noise_ratio = noise_ratio[:, None]
        if noise_ratio.shape[0] != z.shape[0]:
            raise ValueError("noise_ratio batch dimension must match z")
        cond = torch.log1p(noise_ratio.clamp_min(0.0))
        x = self.in_norm(z) + self.noise_embed(cond.to(dtype=z.dtype, device=z.device))
        rms = z.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return z + rms * self.net(x)


@dataclass(frozen=True)
class CorruptionConfig:
    ratio_min: float = 0.02
    ratio_max: float = 3.0
    structured_probability: float = 0.5
    identity_probability: float = 0.05


def sample_log_uniform(
    batch_size: int,
    *,
    low: float,
    high: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if low <= 0 or high <= 0 or high < low:
        raise ValueError("log-uniform bounds must satisfy 0 < low <= high")
    lo = math.log(low)
    hi = math.log(high)
    u = torch.rand(batch_size, device=device, dtype=dtype)
    return torch.exp(lo + (hi - lo) * u)


def _unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def gaussian_corruption(
    clean: torch.Tensor,
    ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = _unit_rows(torch.randn_like(clean))
    scale = ratio[:, None] * clean.norm(dim=-1, keepdim=True)
    delta = eps * scale
    return clean + delta, delta


def structured_corruption(
    clean: torch.Tensor,
    ratio: torch.Tensor,
    bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt along random natural activation-difference directions.

    The bank is generic LM activation data and must stay independent of the
    held-out evaluation steering direction.
    """

    if bank.ndim != 2 or bank.shape[-1] != clean.shape[-1]:
        raise ValueError("bank must have shape [n_activations, d_model]")
    if bank.shape[0] < 2:
        raise ValueError("structured corruption needs at least two bank rows")
    n = clean.shape[0]
    i = torch.randint(0, bank.shape[0], (n,), device=bank.device)
    j = torch.randint(0, bank.shape[0], (n,), device=bank.device)
    same = i == j
    if same.any():
        j = torch.where(same, (j + 1) % bank.shape[0], j)
    direction = (bank[i] - bank[j]).to(device=clean.device, dtype=clean.dtype)
    direction = _unit_rows(direction)
    scale = ratio[:, None] * clean.norm(dim=-1, keepdim=True)
    delta = direction * scale
    return clean + delta, delta


def corrupt_batch(
    clean: torch.Tensor,
    *,
    kind: str,
    config: CorruptionConfig,
    bank: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return corrupted, ratio, delta, structured_mask for a training batch."""

    ratio = sample_log_uniform(
        clean.shape[0],
        low=config.ratio_min,
        high=config.ratio_max,
        device=clean.device,
        dtype=clean.dtype,
    )
    identity_mask = torch.rand(clean.shape[0], device=clean.device) < config.identity_probability
    ratio = ratio.masked_fill(identity_mask, 0.0)

    kind = kind.lower()
    if kind == "gaussian":
        corrupted, delta = gaussian_corruption(clean, ratio)
        structured_mask = torch.zeros(clean.shape[0], device=clean.device, dtype=torch.bool)
    elif kind in {"structured", "mixed"}:
        if bank is None:
            raise ValueError(f"{kind} corruption requires an activation bank")
        if kind == "structured":
            structured_mask = torch.ones(clean.shape[0], device=clean.device, dtype=torch.bool)
        else:
            structured_mask = (
                torch.rand(clean.shape[0], device=clean.device) < config.structured_probability
            )
        gaussian_z, gaussian_delta = gaussian_corruption(clean, ratio)
        structured_z, structured_delta = structured_corruption(clean, ratio, bank)
        mask = structured_mask[:, None]
        corrupted = torch.where(mask, structured_z, gaussian_z)
        delta = torch.where(mask, structured_delta, gaussian_delta)
    else:
        raise ValueError("kind must be one of: gaussian, structured, mixed")

    return corrupted, ratio, delta, structured_mask


def reconstruction_metrics(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    denoised: torch.Tensor,
) -> dict[str, float]:
    noisy_mse = F.mse_loss(corrupted, clean).item()
    denoised_mse = F.mse_loss(denoised, clean).item()
    improvement = 1.0 - denoised_mse / max(noisy_mse, 1e-12)
    return {
        "noisy_mse": float(noisy_mse),
        "denoised_mse": float(denoised_mse),
        "relative_mse_improvement": float(improvement),
    }
