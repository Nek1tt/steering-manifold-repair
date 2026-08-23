# Experiment 007 results — Jacobian Residual Repair

Date: 2026-08-23

## Status

**Mechanistic hypothesis: strongly supported.**  
**Calibration oracle: strongly positive.**  
**Frozen-threshold held-out efficacy test: inconclusive because neither method reached the preregistered C80/C85/C90 thresholds under the manual oracle generation protocol.**  
**Held-out same-alpha evidence: JRR reliably improves NLL/fluency at some strong-steering settings, but concept preservation is seed-sensitive.**

The main scientific conclusion is therefore narrower, and more interesting, than the original JRR hypothesis:

> Strong steering creates an approximately second-order, mostly orthogonal downstream nonlinear remainder. Removing the whole orthogonal remainder can recover fluency, but the removed subspace is not purely harmful: it also contains useful nonlinear concept-carrying adaptation. First-order transported steering alone is not sufficient to preserve the semantic effect.

This means that the next method should be **selective residual repair**, not an amortized predictor of the entire `R_orth` target.

## 1. Mechanistic diagnostic

Source intervention:

```text
blocks.6.hook_resid_post
```

Calibration selected:

```text
blocks.7.hook_resid_post
```

For

```text
R_alpha = F(h + alpha v) - F(h) - alpha J_F(h)v
```

the selected layer produced:

| diagnostic | value |
|---|---:|
| log-log slope of `||R_alpha||` vs alpha | **1.9849** |
| mean orthogonal fraction of remainder | **0.9404** |
| rank corr `||R_orth||` vs NLL | **+0.8909** |
| rank corr `||R_orth||` vs fluency | **-0.8909** |
| JVP implementation | autograd |

The slope is extremely close to the Taylor prediction `O(alpha^2)`. This is the cleanest mechanistic result of the experiment.

The correlation result should be interpreted more cautiously because alpha itself drives both residual magnitude and behavioral degradation; the causal oracle stage was required to distinguish marker from mechanism.

### Strong-steering scale

On held-out autoregressive trajectories the nonlinear term becomes comparable to the first-order steering effect:

| alpha | `||R||` | `||R_orth||` | `||Jv||` | `||R|| / ||alpha Jv||` |
|---:|---:|---:|---:|---:|
| 1.00 | 3.77 | 3.60 | 13.63 | 0.276 |
| 1.50 | 8.40 | 8.05 | 13.67 | 0.409 |
| 2.00 | 16.02 | 15.59 | 13.63 | 0.587 |
| 2.25 | 21.28 | 20.88 | 13.56 | 0.698 |
| 2.50 | 28.36 | 27.96 | 13.49 | 0.841 |
| 3.00 | 40.14 | 39.58 | 13.56 | **0.986** |

At `alpha=3`, the nonlinear remainder is almost as large as the entire first-order displacement `alpha Jv`. Its orthogonal fraction is about 98.6%.

## 2. Calibration oracle

After correcting the first calibration protocol mismatch, the final calibration used the same 8 prompts, 32-token generation length, alpha grid, and seed 37 as the diagnostic behavior probe. No JRR parameter was retuned: target layer remained the diagnostic-selected layer and `beta=1` remained frozen.

Frozen calibration thresholds were C80/C85/C90.

| method | F@C80 | F@C85 | F@C90 |
|---|---:|---:|---:|
| additive | 45.49 | — | — |
| JRR orthogonal | **100.00** | **72.11** | — |

Calibration gain at C80 was **+54.51 fluency points**, so the preregistered gate opened the held-out evaluation.

This calibration result is real for that calibration sample, but the later seed sensitivity means it should not be presented as the expected held-out effect size.

## 3. Frozen held-out evaluation

Held-out protocol:

```text
20 prompts x seeds 11/23
alpha = 0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3
beta = 1
methods = additive, jrr_orth
```

### Preregistered frontier result

Under the manual oracle generator, neither method reached the preregistered C80/C85/C90 thresholds after averaging both seeds:

| method | maximum observed concept | F@C80 | F@C85 | F@C90 |
|---|---:|---:|---:|---:|
| additive | 77.30 | — | — | — |
| JRR orthogonal | 74.37 | — | — | — |

Therefore the **confirmatory C80/C85/C90 held-out test is not estimable**, rather than a numerical win or loss.

This oracle runner uses manual per-prompt autoregressive sampling, so its additive control should not be directly substituted for the earlier batched additive/DPAR held-out frontier. The earlier DPAR experiment reached the high-concept region under its frozen generation protocol; JRR must be compared to the additive control generated inside the same oracle runner.

## 4. Held-out same-alpha behavior

The strongest aggregate fluency improvements appear at high steering strength:

| alpha | additive C | JRR C | delta C | additive F | JRR F | delta F | delta NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 52.37 | 50.47 | -1.91 | 98.16 | 95.72 | -2.44 | +0.183 |
| 1.25 | 52.13 | 51.83 | -0.30 | 93.09 | 97.74 | +4.65 | -0.049 |
| 1.50 | 49.43 | 56.87 | +7.44 | 96.20 | 88.00 | -8.20 | +0.095 |
| 1.75 | 69.70 | 58.57 | -11.13 | 80.77 | 89.94 | +9.17 | -0.107 |
| 2.00 | 62.76 | 56.60 | -6.16 | 83.97 | 86.60 | +2.63 | -0.031 |
| 2.25 | 68.68 | 62.21 | -6.46 | 68.88 | **88.47** | **+19.59** | **-0.250** |
| 2.50 | 63.44 | 59.67 | -3.77 | 71.90 | 75.07 | +3.17 | -0.043 |
| 3.00 | 77.30 | 74.37 | -2.93 | 54.68 | **68.73** | **+14.05** | **-0.229** |

At `alpha=2.25` and `alpha=3.0`, paired prompt/seed bootstrap analysis gives a clearly negative NLL difference:

| alpha | mean delta NLL | paired bootstrap 95% interval |
|---:|---:|---:|
| 2.25 | **-0.250** | **[-0.394, -0.110]** |
| 3.00 | **-0.229** | **[-0.395, -0.071]** |

The corresponding concept-difference intervals include zero, so the semantic effect is much less stable than the NLL improvement.

These bootstrap intervals are post-hoc descriptive diagnostics, not a preregistered significance test.

## 5. Seed sensitivity

The key high-alpha result changes qualitatively across the two frozen generation seeds.

At `alpha=3`:

| seed | additive concept | JRR concept | delta concept | additive fluency | JRR fluency | delta fluency |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 73.55 | **82.32** | **+8.76** | 48.23 | **70.04** | **+21.81** |
| 23 | **81.04** | 66.41 | **-14.62** | 61.98 | **67.43** | **+5.45** |

At `alpha=2.25`, fluency improves by about 20 points in both seeds, but concept changes by -0.87 in seed 11 and -12.05 in seed 23.

So the held-out result is not a robust Pareto domination claim. It is stronger evidence for a **fluency-repair mechanism with an unstable semantic side effect**.

## 6. Exploratory within-support frontier

Because the preregistered high-concept thresholds were unreachable, a post-hoc frontier over thresholds inside the observed combined support is reported only as an exploratory diagnostic:

| method | F@C50 | F@C55 | F@C60 | F@C65 | F@C70 | F@C75 |
|---|---:|---:|---:|---:|---:|---:|
| additive | 98.42 | 91.96 | 88.15 | **84.35** | 63.75 | **57.53** |
| JRR orthogonal | 97.74 | 91.62 | **88.47** | 72.77 | **70.61** | — |

JRR is locally better around C60 and C70 but worse around C65 and does not reach C75 in the combined two-seed average. This nonuniformity matches the same-alpha and per-seed analyses.

Do **not** use this exploratory table as the primary success criterion.

## 7. What the causal test teaches us

The original strong JRR hypothesis was:

> The orthogonal nonlinear remainder is collateral damage, so removing it should preserve the first-order concept direction while restoring fluency.

The experiment supports only the first half.

1. Strong steering really does create a rapidly growing nonlinear downstream effect.
2. Almost all of that effect is orthogonal to the local first-order transported direction.
3. Removing it can substantially lower NLL and improve fluency.
4. But removing it can also reduce the target concept, despite preserving `Jv` by construction.

Therefore:

> **Orthogonal-to-`Jv` is not equivalent to irrelevant-to-concept.**

The nonlinear response appears to contain at least two functional components:

- harmful collateral distortion that damages fluency;
- useful nonlinear adaptation that helps realize the steered concept.

This explains why full `R_orth` removal can look excellent on one seed and trade concept for fluency on another.

## 8. Recommended next research direction

Do **not** train an adapter to reproduce the full `R_orth` yet. That would amortize a target we now know is semantically over-aggressive.

The next hypothesis should be **harmful-mode-selective nonlinear repair**: decompose `R_orth` further and remove only components causally associated with fluency degradation while retaining components that support the concept.

Candidate diagnostics include local logit sensitivity / Fisher-weighted residual geometry, low-rank harmful-mode discovery across prompts, and agreement across independent steering directions.

## Archived evidence

- `heldout_aggregate.csv` — combined held-out aggregate.
- `heldout_frontier_frozen_thresholds.csv` — preregistered C80/C85/C90 frontier (unreached).
- `heldout_same_alpha_deltas.csv` — additive vs JRR at identical alpha.
- `heldout_seed_deltas.csv` — seed-level same-alpha effects.
- `heldout_paired_bootstrap.csv` — paired prompt/seed bootstrap diagnostics.
- `heldout_exploratory_frontier.csv` — post-hoc within-support frontier; descriptive only.
- `heldout_exploratory_per_seed_frontier.csv` — seed sensitivity of exploratory frontier.
- `calibration_frontier.csv` — final valid calibration frontier.
- `diagnostic_target_summary.csv` — layer-selection diagnostic.

## Final claim

The defensible JRR contribution is:

> **Activation steering degradation has a measurable second-order downstream component: the nonlinear Taylor remainder grows approximately as alpha squared and is overwhelmingly orthogonal to the transported first-order steering direction. Causally deleting that orthogonal remainder can restore fluency at strong steering, but held-out evaluation shows that the same remainder can also carry useful concept information. This falsifies the naive “all orthogonal nonlinearity is damage” model and motivates selective nonlinear repair.**
