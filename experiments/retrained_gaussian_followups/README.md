# Retrained Gaussian + scaled repair follow-up

Date: 2026-08-23

This is the final focused follow-up to the repair suite. It reproduces the Gaussian denoiser from a fresh runtime and then tests inference-time correction scaling (`beta`) for vanilla Gaussian repair and DPAR on a calibration/held-out split.

## Protocol

- GPT-2 Small, intervention at `blocks.6.hook_resid_post`.
- Gaussian denoiser retrained from generic WikiText-2 layer-6 activations.
- Training: 80k activations, 5 epochs, seed 2026.
- Calibration: 8 prompts, seed 37.
- Held-out evaluation: 20 prompts x seeds 11/23.
- Alpha grid: `0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0`.
- DPAR beta sweep: `0.10, 0.25, 0.50, 0.75, 1.00`.
- Vanilla beta sweep: `0.25, 0.50, 0.75, 1.00`.

## Reproducibility check

The fresh retrain reproduced the original Gaussian denoiser training history **exactly** at every logged epoch. Final validation relative MSE improvement is **67.8%**, with `val_denoised_mse=2.822651`.

The archived checkpoint is structurally valid: `d_model=768`, `hidden_dim=1536`, 5,316,096 parameters, and loads into `ResidualActivationDenoiser` with a finite forward pass.

This exact history match is a strong check that the activation-cache/training path is deterministic under the frozen seeds and config.

## Calibration result

- Gaussian DPAR selected **beta=0.25** (score 69.56).
- Gaussian vanilla selected **beta=0.25** (score 72.61).

For DPAR, beta=0.25 only narrowly beats beta=0.50 on the calibration objective (69.56 vs 67.85), so the exact optimum should not be treated as highly stable.

## Held-out frontier

| method | F@C80 | F@C85 | F@C90 | F@C95 |
|---|---:|---:|---:|---:|
| additive | 94.73 | 70.79 | 66.46 | — |
| DPAR beta=0.25 (calibration-selected) | 89.53 | 79.16 | 55.16 | 48.84 |
| DPAR beta=1.00 control | 89.47 | 75.03 | 71.45 | — |
| vanilla beta=0.25 (calibration-selected) | 89.11 | 81.06 | 47.65 | — |
| vanilla beta=1.00 control | 80.90 | 75.74 | 52.60 | — |

Relative to additive:
- selected DPAR beta=0.25: **-5.19 / +8.38 / -11.30** at C80/C85/C90;
- full DPAR beta=1.00: **-5.26 / +4.24 / +4.99**;
- selected vanilla beta=0.25: **-5.62 / +10.28 / -18.81**.

## Interpretation

### F1 — Was the previous repair simply too strong?

**Partially supported, but not as a global claim.** Smaller correction clearly creates a useful mid-concept region. Around concept 85, beta=0.25 improves fluency substantially over additive. However the same calibration-selected beta loses at concept 80 and 90, so there is no single scaled setting that uniformly dominates additive.

### F2 — Is DPAR geometry useful after magnitude calibration?

**Mechanistically yes; downstream evidence remains mixed.** DPAR continues to preserve effective alpha essentially exactly, while vanilla repair still subtracts part of the steering direction. But at the calibration-selected beta=0.25, vanilla is actually slightly stronger on the C85 frontier. The distinctive DPAR benefit reappears at high concept strength: full DPAR beta=1 gives the best C90 held-out frontier among the tested controls.

### F4 — Was the old C~90 DPAR result only a coarse-alpha artifact?

**No: the effect survives the denser alpha grid, but it is modest.** Full DPAR reaches F@C90=71.45 versus additive 66.46, a +4.99 descriptive gain. This is much smaller than the misleading discrete-grid advantage from the first repair suite, but it does not disappear under interpolation.

## Seed sensitivity

The two held-out seeds tell the same qualitative story at C90 for full DPAR:

| seed | additive F@C90 | DPAR beta=1 F@C90 | delta |
|---:|---:|---:|---:|
| 11 | 61.23 | 71.76 | +10.53 |
| 23 | 69.04 | 89.75 | +20.71 |

The concept judge remains high-variance and non-monotonic across alpha, so these values should still be described as **descriptive evidence**, not statistical proof.

## Final conclusion

The strongest defensible result of the project is now:

> Vanilla denoising has a measurable geometric failure mode: it cancels the requested steering direction. DPAR removes this failure exactly. Once correction magnitude is separated from geometry, small denoiser corrections can improve the concept/fluency frontier locally, and the dense follow-up shows that full DPAR retains a modest high-concept advantage over additive steering. However, no single calibration-selected beta uniformly dominates the additive baseline, so the evidence supports a promising mechanism and local Pareto gains rather than a universal steering-repair method.

For the final assignment report, this should be presented as the positive mechanistic contribution, with the downstream Pareto result explicitly qualified as local / threshold-dependent.
