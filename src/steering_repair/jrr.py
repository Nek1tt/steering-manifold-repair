from __future__ import annotations

from typing import Callable, Iterable

import torch

TensorFn = Callable[[torch.Tensor], torch.Tensor]


def project_onto_direction(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project x onto direction along the final dimension."""
    d = direction
    while d.ndim < x.ndim:
        d = d.unsqueeze(0)
    denom = d.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    coeff = (x * d).sum(dim=-1, keepdim=True) / denom
    return coeff * d


def decompose_remainder(
    remainder: torch.Tensor, transported_direction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a nonlinear remainder into transported-direction parallel/orthogonal parts."""
    parallel = project_onto_direction(remainder, transported_direction)
    return parallel, remainder - parallel


def apply_jrr_repair(
    y_alpha: torch.Tensor,
    remainder: torch.Tensor,
    transported_direction: torch.Tensor,
    *,
    beta: float = 1.0,
    preserve_parallel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove selected nonlinear remainder from a downstream state."""
    if preserve_parallel:
        _, removed = decompose_remainder(remainder, transported_direction)
    else:
        removed = remainder
    return y_alpha - float(beta) * removed, removed


def directional_jvp_generic(
    fn: TensorFn,
    x: torch.Tensor,
    direction: torch.Tensor,
    *,
    mode: str = "autograd",
    finite_difference_epsilon: float = 0.01,
    fallback_to_finite_difference: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Evaluate f(x) and J_f(x) @ direction, with a robust FD fallback."""
    mode = str(mode).lower()
    eps = float(finite_difference_epsilon)
    if eps <= 0:
        raise ValueError("finite_difference_epsilon must be positive")

    if mode == "finite_difference":
        with torch.no_grad():
            y0 = fn(x)
            yp = fn(x + eps * direction)
            ym = fn(x - eps * direction)
        return y0.detach(), ((yp - ym) / (2.0 * eps)).detach(), "finite_difference"

    if mode != "autograd":
        raise ValueError("jvp mode must be 'autograd' or 'finite_difference'")

    try:
        with torch.enable_grad():
            x0 = x.detach().requires_grad_(True)
            y0, jvp = torch.autograd.functional.jvp(
                fn,
                (x0,),
                (direction.detach(),),
                create_graph=False,
                strict=False,
            )
        return y0.detach(), jvp.detach(), "autograd"
    except (RuntimeError, NotImplementedError) as exc:
        if not fallback_to_finite_difference:
            raise
        print(
            f"JVP autograd failed ({type(exc).__name__}: {exc}); "
            "using central finite difference."
        )
        with torch.no_grad():
            y0 = fn(x)
            yp = fn(x + eps * direction)
            ym = fn(x - eps * direction)
        return (
            y0.detach(),
            ((yp - ym) / (2.0 * eps)).detach(),
            "finite_difference_fallback",
        )


def replace_last_hook(value: torch.Tensor):
    def hook(resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        if resid.shape[0] != 1:
            raise ValueError("JRR counterfactual map currently expects batch size 1")
        out = resid.clone()
        v = value if value.ndim == 2 else value.unsqueeze(0)
        out[:, -1, :] = v.to(device=resid.device, dtype=resid.dtype)
        return out

    return hook


def capture_last_hook(storage: dict[str, torch.Tensor], key: str):
    def hook(resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        storage[key] = resid[:, -1, :]
        return resid

    return hook


def additive_last_hook(direction: torch.Tensor, alpha: float):
    def hook(resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        out = resid.clone()
        v = direction.to(device=resid.device, dtype=resid.dtype)
        out[:, -1, :] = out[:, -1, :] + float(alpha) * v
        return out

    return hook


def replace_target_last_hook(value: torch.Tensor):
    def hook(resid: torch.Tensor, hook=None) -> torch.Tensor:
        del hook
        out = resid.clone()
        v = value if value.ndim == 2 else value.unsqueeze(0)
        out[:, -1, :] = v.to(device=resid.device, dtype=resid.dtype)
        return out

    return hook


@torch.no_grad()
def capture_source_last(model, tokens: torch.Tensor, source_hook: str) -> torch.Tensor:
    _, cache = model.run_with_cache(tokens, names_filter=[source_hook])
    return cache[source_hook][0, -1, :].detach()


def downstream_map(
    model,
    tokens: torch.Tensor,
    *,
    source_hook: str,
    target_hooks: list[str],
    source_value: torch.Tensor,
) -> torch.Tensor:
    """Map one last-token source residual state to one or more downstream states."""
    captured: dict[str, torch.Tensor] = {}
    hooks = [(source_hook, replace_last_hook(source_value))]
    hooks.extend((name, capture_last_hook(captured, name)) for name in target_hooks)
    with model.hooks(fwd_hooks=hooks):
        model(tokens)
    missing = [name for name in target_hooks if name not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture downstream hooks: {missing}")
    return torch.stack([captured[name][0] for name in target_hooks], dim=0)


def model_directional_jvp(
    model,
    tokens: torch.Tensor,
    *,
    source_hook: str,
    target_hooks: list[str],
    source_value: torch.Tensor,
    direction: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    fn = lambda x: downstream_map(
        model,
        tokens,
        source_hook=source_hook,
        target_hooks=target_hooks,
        source_value=x,
    )
    dcfg = cfg["jrr"]["diagnostic"]
    return directional_jvp_generic(
        fn,
        source_value,
        direction,
        mode=dcfg.get("jvp_mode", "autograd"),
        finite_difference_epsilon=float(
            dcfg.get("finite_difference_epsilon", 0.01)
        ),
        fallback_to_finite_difference=bool(
            dcfg.get("fallback_to_finite_difference", True)
        ),
    )


def validate_hooks(model, source_hook: str, target_hooks: Iterable[str]) -> None:
    available = set(model.hook_dict)
    targets = list(target_hooks)
    names = [source_hook, *targets]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown TransformerLens hook names: {missing}")
    if source_hook in set(targets):
        raise ValueError(
            "target hooks must be downstream of and distinct from source_hook"
        )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom.item()) < 1e-12:
        return 0.0
    return float((a * b).sum().div(denom).item())
