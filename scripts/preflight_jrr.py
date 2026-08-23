from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
import yaml

from steering_repair.jrr import (
    capture_source_last,
    cosine,
    directional_jvp_generic,
    downstream_map,
    model_directional_jvp,
    validate_hooks,
)
from steering_repair.sentiment_baseline import load_direction, load_gpt2, load_lines


def _block_index(hook_name: str) -> int:
    match = re.search(r"blocks\.(\d+)\.", hook_name)
    if match is None:
        raise ValueError(f"Cannot parse Transformer block index from hook {hook_name!r}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-model numerical preflight for Jacobian Residual Repair"
    )
    parser.add_argument("--config", default="configs/jrr_gpt2.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    jcfg = cfg["jrr"]
    dcfg = jcfg["diagnostic"]
    source_hook = str(jcfg["source_hook"])
    target_hooks = [str(x) for x in dcfg["target_hooks"]]
    source_idx = _block_index(source_hook)
    bad = [name for name in target_hooks if _block_index(name) <= source_idx]
    if bad:
        raise RuntimeError(
            f"JRR targets must be strictly downstream of {source_hook}: {bad}"
        )

    model = load_gpt2(cfg)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    validate_hooks(model, source_hook, target_hooks)

    direction, _ = load_direction(cfg["vector"]["cache_path"], model.cfg.device)
    if not torch.isfinite(direction).all() or float(direction.norm().item()) < 1e-8:
        raise RuntimeError("Saved steering direction is non-finite or near zero")

    prompt = load_lines(dcfg["prompts_path"])[0]
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    h = capture_source_last(model, tokens, source_hook)

    _, primary_jvp, primary_mode = model_directional_jvp(
        model,
        tokens,
        source_hook=source_hook,
        target_hooks=target_hooks,
        source_value=h,
        direction=direction,
        cfg=cfg,
    )

    check_eps = float(dcfg.get("preflight_check_epsilon", 0.02))
    min_cos = float(dcfg.get("preflight_min_jvp_cosine", 0.98))
    max_rel = float(dcfg.get("preflight_max_jvp_relative_error", 0.15))

    fn = lambda x: downstream_map(
        model,
        tokens,
        source_hook=source_hook,
        target_hooks=target_hooks,
        source_value=x,
    )
    _, check_jvp, _ = directional_jvp_generic(
        fn,
        h,
        direction,
        mode="finite_difference",
        finite_difference_epsilon=check_eps,
        fallback_to_finite_difference=False,
    )

    rows = []
    for i, target in enumerate(target_hooks):
        a = primary_jvp[i]
        b = check_jvp[i]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError(f"Non-finite Jv at {target}")
        a_norm = float(a.norm().item())
        b_norm = float(b.norm().item())
        if min(a_norm, b_norm) < 1e-8:
            raise RuntimeError(f"Near-zero Jv at {target}")
        cos = cosine(a, b)
        rel = float((a - b).norm().div(a.norm().clamp_min(1e-12)).item())
        rows.append((target, a_norm, b_norm, cos, rel))

    print("JRR real-model preflight")
    print("device:", model.cfg.device)
    print("primary Jv mode:", primary_mode)
    print("finite-difference check epsilon:", check_eps)
    for target, a_norm, b_norm, cos, rel in rows:
        print(
            f"  {target}: primary_norm={a_norm:.5f} check_norm={b_norm:.5f} "
            f"cos={cos:.6f} rel_error={rel:.6f}"
        )

    worst_cos = min(row[3] for row in rows)
    worst_rel = max(row[4] for row in rows)
    if not math.isfinite(worst_cos) or worst_cos < min_cos:
        raise RuntimeError(
            f"Jv numerical stability FAIL: minimum cosine {worst_cos:.6f} < {min_cos:.6f}"
        )
    if not math.isfinite(worst_rel) or worst_rel > max_rel:
        raise RuntimeError(
            f"Jv numerical stability FAIL: maximum relative error {worst_rel:.6f} > {max_rel:.6f}"
        )

    print("JRR PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
