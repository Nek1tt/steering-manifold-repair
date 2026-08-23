import torch

from steering_repair.selective_jrr import (
    kl_clean_to_current,
    select_kl_harmful_component,
)


def test_kl_is_zero_for_identical_logits_and_positive_otherwise():
    clean = torch.tensor([[1.0, 2.0, -1.0]])
    same = kl_clean_to_current(clean, clean)
    shifted = kl_clean_to_current(clean, torch.tensor([[2.0, -1.0, 0.5]]))
    assert abs(float(same)) < 1e-7
    assert float(shifted) > 0.0


def test_selected_component_is_orthogonal_to_transported_direction():
    remainder = torch.tensor([2.0, 3.0, 4.0])
    transported = torch.tensor([1.0, 0.0, 0.0])
    grad = torch.tensor([5.0, 2.0, 1.0])
    selected, g_orth, coeff = select_kl_harmful_component(
        remainder, transported, grad, positive_only=True
    )
    assert float(coeff) > 0.0
    assert torch.allclose(g_orth[0], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose((selected * transported).sum(), torch.tensor(0.0), atol=1e-7)


def test_positive_only_preserves_kl_helping_residual():
    transported = torch.tensor([1.0, 0.0, 0.0])
    remainder = torch.tensor([0.0, -2.0, 0.0])
    grad = torch.tensor([0.0, 1.0, 0.0])
    selected, _, coeff = select_kl_harmful_component(
        remainder, transported, grad, positive_only=True
    )
    assert torch.allclose(selected, torch.zeros_like(selected))
    assert float(coeff) == 0.0


def test_selective_repair_does_not_remove_unrelated_orthogonal_mode():
    transported = torch.tensor([1.0, 0.0, 0.0, 0.0])
    remainder = torch.tensor([0.0, 3.0, 4.0, 0.0])
    grad = torch.tensor([0.0, 1.0, 0.0, 0.0])
    selected, _, _ = select_kl_harmful_component(
        remainder, transported, grad, positive_only=True
    )
    assert torch.allclose(selected, torch.tensor([0.0, 3.0, 0.0, 0.0]), atol=1e-6)
    kept = remainder - selected
    assert torch.allclose(kept, torch.tensor([0.0, 0.0, 4.0, 0.0]), atol=1e-6)
