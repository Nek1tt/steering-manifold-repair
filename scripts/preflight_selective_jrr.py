from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from steering_repair.jrr import capture_source_last, decompose_remainder, downstream_map, model_directional_jvp, replace_target_last_hook
from steering_repair.selective_jrr import kl_clean_to_current, kl_gradient_at_target, select_kl_harmful_component
from steering_repair.sentiment_baseline import load_direction, load_gpt2, load_lines


def logits_from_target(model, tokens, target_hook, value):
    with model.hooks(fwd_hooks=[(target_hook, replace_target_last_hook(value))]):
        return model(tokens)[:, -1, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selective_jrr_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scfg = cfg["selective_jrr"]
    source_hook = scfg["source_hook"]
    target_hook = scfg["target_hook"]
    alpha = 2.25

    model = load_gpt2(cfg)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    direction, _ = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    prompt = load_lines(scfg["calibration"]["prompts_path"])[0]
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    h = capture_source_last(model, tokens, source_hook)
    y0, jv_batch, mode = model_directional_jvp(
        model, tokens, source_hook=source_hook, target_hooks=[target_hook],
        source_value=h, direction=direction, cfg=cfg,
    )
    jv = jv_batch[0]
    with torch.no_grad():
        clean_logits = model(tokens)[:, -1, :]
        y_alpha = downstream_map(
            model, tokens, source_hook=source_hook, target_hooks=[target_hook],
            source_value=h + alpha * direction,
        )[0]
    remainder = y_alpha - y0[0] - alpha * jv
    _, r_orth = decompose_remainder(remainder, jv)
    _, grad, kl0 = kl_gradient_at_target(
        model, tokens, target_hook=target_hook, target_value=y_alpha,
        clean_logits=clean_logits,
    )
    selected, g_orth, coeff = select_kl_harmful_component(
        r_orth, jv, grad, positive_only=True,
    )

    if float(g_orth.norm().item()) < 1e-10:
        raise RuntimeError("KL gradient vanished after Jv projection")
    d = g_orth / g_orth.norm().clamp_min(1e-12)
    eps = 1e-3
    with torch.no_grad():
        kp = float(kl_clean_to_current(clean_logits, logits_from_target(model, tokens, target_hook, y_alpha + eps * d)).item())
        km = float(kl_clean_to_current(clean_logits, logits_from_target(model, tokens, target_hook, y_alpha - eps * d)).item())
    finite_diff = (kp - km) / (2.0 * eps)
    autograd_value = float((grad * d).sum().item())
    relative_error = abs(finite_diff - autograd_value) / max(abs(autograd_value), 1e-8)
    cosine_to_jv = float((selected * jv).sum().abs().div(selected.norm().clamp_min(1e-12) * jv.norm().clamp_min(1e-12)).item())

    print("KL-selective JRR preflight")
    print("Jv mode:", mode)
    print("KL before repair:", kl0)
    print("selected coefficient:", float(coeff.item()))
    print("selected fraction:", float(selected.norm().div(r_orth.norm().clamp_min(1e-12)).item()))
    print("gradient derivative autograd:", autograd_value)
    print("gradient derivative finite difference:", finite_diff)
    print("relative error:", relative_error)
    print("abs cosine(selected, Jv):", cosine_to_jv)

    if relative_error > 0.15:
        raise RuntimeError("KL gradient finite-difference check failed")
    if cosine_to_jv > 1e-5:
        raise RuntimeError("Selected correction is not orthogonal to transported steering")
    if not torch.isfinite(selected).all():
        raise RuntimeError("Selected correction contains non-finite values")
    print("KL-SELECTIVE JRR PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
