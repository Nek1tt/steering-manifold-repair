import torch

from steering_repair.steering import SteeringHook


def test_norm_ratio_changes_only_last_position():
    resid = torch.randn(2, 4, 8)
    original = resid.clone()
    hook = SteeringHook(torch.randn(8), strength=0.5, strength_mode="norm_ratio")
    out = hook(resid)
    assert torch.allclose(out[:, :-1], original[:, :-1])
    delta = out[:, -1] - original[:, -1]
    expected = 0.5 * original[:, -1].norm(dim=-1)
    assert torch.allclose(delta.norm(dim=-1), expected, rtol=1e-5, atol=1e-5)


def test_sae_alpha_matches_normalized_coordinate_scale():
    resid = torch.randn(3, 2, 8)
    hook = SteeringHook(torch.randn(8), strength=7.0, strength_mode="sae_alpha")
    out = hook(resid)
    delta = out[:, -1] - resid[:, -1]
    expected = 7.0 * resid[:, -1].std(dim=-1)
    assert torch.allclose(delta.norm(dim=-1), expected, rtol=1e-5, atol=1e-5)


def test_vector_alpha_preserves_raw_direction_magnitude():
    resid = torch.randn(2, 3, 4)
    raw_vector = torch.tensor([1.0, -2.0, 0.5, 3.0])
    alpha = 1.75
    hook = SteeringHook(raw_vector, strength=alpha, strength_mode="vector_alpha")
    out = hook(resid)
    expected = alpha * raw_vector
    delta = out[:, -1] - resid[:, -1]
    assert torch.allclose(delta, expected.unsqueeze(0).expand_as(delta), rtol=1e-5, atol=1e-5)
    assert torch.allclose(out[:, :-1], resid[:, :-1])


def test_norm_preserving_repair_restores_norm():
    resid = torch.randn(2, 3, 8)
    hook = SteeringHook(
        torch.randn(8),
        strength=0.8,
        strength_mode="norm_ratio",
        repair="norm_preserving",
    )
    out = hook(resid)
    assert torch.allclose(
        out[:, -1].norm(dim=-1),
        resid[:, -1].norm(dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )
