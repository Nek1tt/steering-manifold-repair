# Repair suite — Gaussian denoising, DPAR, and structured corruption

Date: 2026-08-23

This experiment starts from the frozen successful GPT-2 midpoint sentiment-steering baseline and tests the assignment-proposed one-step denoiser plus two extensions:

1. **Gaussian denoiser** — residual MLP trained on generic layer-6 activations with isotropic corruption.
2. **DPAR (Direction-Preserving Activation Repair)** — remove the denoiser correction component parallel to the requested steering direction.
3. **Structured corruption** — train a second denoiser on a 50/50 mixture of isotropic noise and natural activation-difference directions.

The held-out sentiment steering vector is never used for denoiser training.

## Run size

- LM: GPT-2 Small.
- Intervention: `blocks.6.hook_resid_post`.
- Frozen evaluation: 20 prompts × 2 seeds.
- Steering strengths: `0, 0.5, 0.75, 1, 1.5, 2, 3, 4`.
- Methods: 7.
- Total generated completions: **2240**.
- Denoiser training cache: up to 80k generic layer-6 activations.
- Training: 5 epochs.

## Executive result

The run gives a useful **mechanistic positive result but not yet a robust downstream Pareto win**.

### What worked

**DPAR validates the cancellation hypothesis.** Vanilla Gaussian denoising increasingly subtracts the intended steering direction as alpha grows. At requested `alpha=4`, the Gaussian denoiser leaves only `effective_alpha=2.36` and the repair/steering cosine reaches `-0.615`. DPAR keeps `effective_alpha≈alpha` to numerical precision while still applying a substantial orthogonal correction.

Mean absolute alpha-preservation error reported by the automated check:

- vanilla Gaussian: **0.4354**
- Gaussian DPAR: **5.2e-8**

This directly demonstrates that a vanilla denoiser can look like a repair method partly by weakening the requested intervention.

### What did not work

**Vanilla Gaussian denoising does not improve the frozen Pareto frontier.** The denoiser reconstructs held-out Gaussian-corrupted activations well, but this does not translate to better generated-text fluency at matched concept strength.

At concept score ≥90 on the discrete sweep:

- additive: fluency **63.35**
- Gaussian: **26.23**
- Gaussian DPAR: **71.05**

The `+7.69` Gaussian-DPAR advantage over additive at this threshold is interesting but should **not** yet be advertised as a robust Pareto improvement. The alpha grid is coarse: linearly interpolating the additive points around concept 90 gives roughly fluency 73, so the apparent discrete advantage can disappear under interpolation. At equal alpha, full-strength DPAR generally increases NLL.

Therefore the correct conclusion from this run is:

> DPAR solves a real geometric failure mode of vanilla denoising, but the learned orthogonal correction is not yet aligned well enough with downstream language-model fluency.

### Structured corruption is a clean negative result

The structured-training hypothesis works at the activation reconstruction level but fails downstream.

Cross-corruption reconstruction:

| checkpoint | eval corruption | MSE improvement |
|---|---|---:|
| Gaussian | Gaussian | 67.9% |
| Gaussian | structured | 50.8% |
| Mixed | Gaussian | 49.9% |
| Mixed | structured | 68.7% |

So the two denoisers clearly specialize to their training corruption geometry. However, `mixed` and `mixed_dpar` do not improve the sentiment-steering Pareto frontier; neither reaches concept ≥90 in the evaluated grid. This is evidence that **better reconstruction of a chosen perturbation family is not sufficient for better steering repair**.

## Main tables

### Frozen frontier summary

| method | fluency @ concept≥70 | ≥80 | ≥90 | ≥95 |
|---|---:|---:|---:|---:|
| additive | 99.56 | 88.74 | 63.35 | 18.01 |
| Gaussian | 100.00 | 72.46 | 26.23 | — |
| Gaussian DPAR | 100.00 | 89.34 | 71.05 | — |
| Gaussian λ=0.5 | 100.00 | 71.24 | 53.57 | 9.65 |
| mixed | 70.15 | 70.15 | — | — |
| mixed DPAR | 83.14 | 73.01 | — | — |
| norm preserving | 97.89 | 86.67 | — | — |

### Key geometry points

| method | alpha | effective alpha | correction cosine | correction / steering norm |
|---|---:|---:|---:|---:|
| Gaussian | 1.5 | 1.364 | -0.257 | 0.341 |
| Gaussian | 2.0 | 1.680 | -0.361 | 0.435 |
| Gaussian | 4.0 | 2.358 | -0.615 | 0.665 |
| Gaussian DPAR | 1.5 | 1.500 | ~0 | 0.326 |
| Gaussian DPAR | 2.0 | 2.000 | ~0 | 0.407 |
| Gaussian DPAR | 4.0 | 4.000 | ~0 | 0.520 |

The DPAR correction is therefore not the identity: at `alpha=2` its orthogonal correction norm is about 41% of the steering perturbation norm. It simply prevents that correction from erasing the protected direction.

## Training diagnostics

The denoisers learned the activation reconstruction task normally.

Gaussian denoiser validation relative MSE improvement:
`22.7% → 40.0% → 52.3% → 61.5% → 67.8%`.

Mixed denoiser validation relative MSE improvement:
`25.1% → 38.1% → 46.9% → 54.3% → 60.0%`.

There is no sign that the downstream failure is caused by a completely untrained network. Instead, the result exposes a mismatch between **activation-space MSE reconstruction** and **language-model behavioral quality**.

## What to show in the final report

Primary figures:

1. Pareto comparison restricted to additive vs Gaussian vs Gaussian DPAR.
2. Requested vs effective alpha: the clean mechanistic result that vanilla denoising cancels alpha and DPAR fixes it.

Secondary / appendix:

3. Correction cosine vs alpha.
4. Structured and norm-preserving controls.
5. Cross-corruption reconstruction table.

The original seven-line Pareto plot is useful for debugging but is too crowded for the main report.

## Hypothesis status

- **H1: Gaussian denoiser improves the frontier — not supported.**
- **H2: vanilla denoising cancels the protected direction and DPAR preserves it — strongly supported mechanistically.**
- **H3: structured corruption improves DPAR downstream — not supported.**

## Next experiments, in priority order

### E2A — Scaled DPAR (highest priority, no retraining)

The current experiment always applies 100% of the orthogonal denoiser correction. Since full DPAR preserves alpha but often worsens NLL, test

`h' = z + beta * Delta_perp`

with `beta ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}`.

Choose beta only on the already-separated calibration prompts, then evaluate once on the frozen test prompts. Also densify alpha around the useful transition (`1.0, 1.25, 1.5, 1.75, 2.0`). This is the fastest way to determine whether the geometric idea can produce a genuine Pareto win rather than the coarse-grid artifact seen here.

### E2B — Match denoiser training to inference states

The cache currently contains generic token-position activations, while repair is applied to the current response-token state during autoregressive generation. Train a second Gaussian denoiser on **final/current-token activations from random prefixes** and narrow the corruption-ratio distribution to the actual evaluation range (roughly 0.08–0.63 for alpha 0.5–4).

This tests whether the failure is mostly distribution mismatch.

### E2C — LM-aware denoising loss

If time remains, stop optimizing only hidden-state MSE. Add a behavior-preservation term on generic text, e.g. frozen-model next-token KL / CE after the repaired layer-6 state. The hypothesis is that reconstruction distance is a poor proxy for downstream fluency, which this experiment directly suggests.

### Validation after a positive result

Only after a method beats the frozen sentiment frontier should it be tested on a second held-out steering direction. This is necessary before claiming general steering repair rather than sentiment-specific tuning.

## Archived files

- `aggregate_compact.csv` — all 56 method/alpha rows with the metrics needed for the analysis.
- `frontier_summary.csv` — discrete Pareto threshold summary.
- `denoiser_cross_reconstruction.csv` — held-out reconstruction transfer.
- `denoiser_gaussian_history.json`, `denoiser_mixed_history.json` — training histories.
- `auto_hypothesis_report.md/json` — original automatic checks from the run.
- `config.yaml` — frozen experiment configuration.

The 1.3 MB raw generation CSV is intentionally not committed here; the full evaluator is deterministic with fixed seeds and can regenerate it. The aggregate scientific results are preserved in this directory.
