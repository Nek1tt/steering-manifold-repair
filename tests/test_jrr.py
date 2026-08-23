import torch

from steering_repair.jrr import (
    apply_jrr_repair,
    decompose_remainder,
    directional_jvp_generic,
)


def test_remainder_decomposition_is_orthogonal():
    torch.manual_seed(0)
    r = torch.randn(5, 12)
    t = torch.randn(5, 12)
    parallel, orthogonal = decompose_remainder(r, t)
    assert torch.allclose(parallel + orthogonal, r, atol=1e-6)
    dot = (orthogonal * t).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)


def test_jrr_preserves_parallel_nonlinearity_and_removes_orthogonal_part():
    t = torch.tensor([1.0, 0.0, 0.0])
    remainder = torch.tensor([2.0, 3.0, 4.0])
    y_alpha = torch.tensor([10.0, 20.0, 30.0])
    repaired, removed = apply_jrr_repair(
        y_alpha,
        remainder,
        t,
        beta=1.0,
        preserve_parallel=True,
    )
    assert torch.allclose(removed, torch.tensor([0.0, 3.0, 4.0]))
    assert torch.allclose(repaired, torch.tensor([10.0, 17.0, 26.0]))


def test_full_remainder_ablation_recovers_linearized_state():
    y0 = torch.tensor([1.0, 2.0])
    jv = torch.tensor([3.0, 5.0])
    alpha = 2.0
    remainder = torch.tensor([7.0, -4.0])
    y_alpha = y0 + alpha * jv + remainder
    repaired, removed = apply_jrr_repair(
        y_alpha,
        remainder,
        jv,
        beta=1.0,
        preserve_parallel=False,
    )
    assert torch.allclose(removed, remainder)
    assert torch.allclose(repaired, y0 + alpha * jv)


def test_directional_jvp_finds_quadratic_second_order_remainder():
    torch.manual_seed(1)
    a = torch.randn(6, 6)
    x = torch.randn(6)
    v = torch.randn(6)

    def fn(z):
        return a @ z + z.square()

    y0, jv, mode = directional_jvp_generic(fn, x, v, mode="autograd")
    assert mode == "autograd"
    for alpha in (0.1, 0.5, 1.0, 2.0):
        exact = fn(x + alpha * v)
        remainder = exact - y0 - alpha * jv
        expected = alpha * alpha * v.square()
        assert torch.allclose(remainder, expected, atol=1e-5, rtol=1e-5)


def test_finite_difference_jvp_matches_autograd():
    torch.manual_seed(2)
    x = torch.randn(10)
    v = torch.randn(10)

    def fn(z):
        return torch.tanh(z) + 0.2 * z.square()

    _, j_auto, _ = directional_jvp_generic(fn, x, v, mode="autograd")
    _, j_fd, mode = directional_jvp_generic(
        fn,
        x,
        v,
        mode="finite_difference",
        finite_difference_epsilon=1e-3,
    )
    assert mode == "finite_difference"
    rel = (j_auto - j_fd).norm() / j_auto.norm().clamp_min(1e-8)
    assert float(rel) < 2e-3
