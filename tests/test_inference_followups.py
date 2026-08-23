import pandas as pd
import torch

from steering_repair.inference_followups import (
    ScaledDenoiserSteeringHook,
    interpolated_fluency_at_threshold,
)


class _OppositeDenoiser(torch.nn.Module):
    def __init__(self, direction: torch.Tensor, amount: float):
        super().__init__()
        self.register_buffer("direction", direction)
        self.amount = float(amount)

    def forward(self, z, ratio):
        del ratio
        return z - self.amount * self.direction.unsqueeze(0)


def _effective_alpha(out, resid, v):
    delta = out[:, -1] - resid[:, -1]
    return (delta * v).sum(dim=-1) / v.square().sum()


def test_scaled_dpar_preserves_alpha_for_any_beta():
    v = torch.randn(8)
    resid = torch.randn(4, 3, 8)
    denoiser = _OppositeDenoiser(v, amount=0.8)
    for beta in (0.1, 0.25, 0.5, 1.0):
        hook = ScaledDenoiserSteeringHook(
            raw_direction=v,
            alpha=2.0,
            denoiser=denoiser,
            correction_scale=beta,
            parallel_keep=0.0,
        )
        out = hook(resid)
        eff = _effective_alpha(out, resid, v)
        assert torch.allclose(eff, torch.full_like(eff, 2.0), atol=1e-5)


def test_scaled_vanilla_cancellation_scales_with_beta():
    v = torch.randn(8)
    resid = torch.randn(4, 3, 8)
    denoiser = _OppositeDenoiser(v, amount=0.8)
    hook = ScaledDenoiserSteeringHook(
        raw_direction=v,
        alpha=2.0,
        denoiser=denoiser,
        correction_scale=0.25,
        parallel_keep=1.0,
    )
    out = hook(resid)
    eff = _effective_alpha(out, resid, v)
    assert torch.allclose(eff, torch.full_like(eff, 1.8), atol=1e-5)


def test_zero_beta_equals_additive_steering():
    v = torch.randn(8)
    resid = torch.randn(4, 3, 8)
    denoiser = _OppositeDenoiser(v, amount=1.0)
    hook = ScaledDenoiserSteeringHook(
        raw_direction=v,
        alpha=1.5,
        denoiser=denoiser,
        correction_scale=0.0,
        parallel_keep=0.0,
    )
    out = hook(resid)
    expected = resid[:, -1] + 1.5 * v
    assert torch.allclose(out[:, -1], expected, atol=1e-5)


def test_interpolated_frontier_handles_coarse_alpha_crossing():
    part = pd.DataFrame(
        {
            "strength": [1.0, 2.0],
            "concept_score": [84.0, 94.0],
            "fluency_score": [89.0, 63.0],
        }
    )
    # C=90 is 60% of the way from 84 to 94, so F=89 + .6*(63-89)=73.4.
    got = interpolated_fluency_at_threshold(part, 90.0)
    assert abs(got - 73.4) < 1e-6
