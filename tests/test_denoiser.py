import torch

from steering_repair.denoiser import (
    CorruptionConfig,
    ResidualActivationDenoiser,
    corrupt_batch,
    gaussian_corruption,
    structured_corruption,
)
from steering_repair.repair_experiment import DenoiserSteeringHook


def test_residual_denoiser_starts_as_identity():
    model = ResidualActivationDenoiser(d_model=8, hidden_dim=16)
    z = torch.randn(4, 8)
    ratio = torch.rand(4)
    out = model(z, ratio)
    assert torch.allclose(out, z)


def test_gaussian_corruption_has_requested_relative_norm():
    clean = torch.randn(6, 8)
    ratio = torch.tensor([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
    corrupted, delta = gaussian_corruption(clean, ratio)
    measured = delta.norm(dim=-1) / clean.norm(dim=-1)
    assert torch.allclose(measured, ratio, rtol=1e-5, atol=1e-5)
    assert torch.allclose(corrupted, clean + delta)


def test_structured_corruption_has_requested_relative_norm():
    clean = torch.randn(5, 8)
    bank = torch.randn(20, 8)
    ratio = torch.linspace(0.1, 1.0, 5)
    _, delta = structured_corruption(clean, ratio, bank)
    measured = delta.norm(dim=-1) / clean.norm(dim=-1)
    assert torch.allclose(measured, ratio, rtol=1e-5, atol=1e-5)


def test_mixed_corruption_shapes_and_identity_samples():
    clean = torch.randn(64, 8)
    bank = torch.randn(128, 8)
    cfg = CorruptionConfig(
        ratio_min=0.2,
        ratio_max=0.3,
        structured_probability=0.5,
        identity_probability=1.0,
    )
    corrupted, ratio, delta, mask = corrupt_batch(clean, kind="mixed", config=cfg, bank=bank)
    assert corrupted.shape == clean.shape
    assert ratio.shape == (64,)
    assert delta.shape == clean.shape
    assert mask.shape == (64,)
    assert torch.count_nonzero(ratio) == 0
    assert torch.allclose(corrupted, clean)


class _OppositeDenoiser(torch.nn.Module):
    def __init__(self, direction: torch.Tensor, amount: float):
        super().__init__()
        self.register_buffer("direction", direction)
        self.amount = amount

    def forward(self, z, ratio):
        del ratio
        return z - self.amount * self.direction.unsqueeze(0)


def test_dpar_preserves_requested_effective_alpha():
    v = torch.randn(8)
    alpha = 2.0
    denoiser = _OppositeDenoiser(v, amount=0.75)
    hook = DenoiserSteeringHook(
        raw_direction=v,
        alpha=alpha,
        denoiser=denoiser,
        parallel_keep=0.0,
    )
    resid = torch.randn(3, 2, 8)
    out = hook(resid)
    delta = out[:, -1] - resid[:, -1]
    effective_alpha = (delta * v).sum(dim=-1) / v.square().sum()
    assert torch.allclose(effective_alpha, torch.full_like(effective_alpha, alpha), atol=1e-5)


def test_vanilla_denoiser_can_cancel_alpha():
    v = torch.randn(8)
    alpha = 2.0
    denoiser = _OppositeDenoiser(v, amount=0.75)
    hook = DenoiserSteeringHook(
        raw_direction=v,
        alpha=alpha,
        denoiser=denoiser,
        parallel_keep=1.0,
    )
    resid = torch.randn(3, 2, 8)
    out = hook(resid)
    delta = out[:, -1] - resid[:, -1]
    effective_alpha = (delta * v).sum(dim=-1) / v.square().sum()
    assert torch.allclose(effective_alpha, torch.full_like(effective_alpha, alpha - 0.75), atol=1e-5)
