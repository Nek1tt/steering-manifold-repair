# Experiment 008 — KL-Selective JRR results

Date: 2026-08-24

## Outcome

The preregistered calibration gate **did not pass**, so the fresh held-out split (`data/selective_jrr_heldout_prompts.txt`, seeds `101/211`) was intentionally not opened.

This is a useful partial/negative result rather than a failed implementation. KL-Selective JRR does identify a small KL-increasing component inside the nonlinear remainder and often improves fluency, but the one-dimensional local KL selector is not sufficient to preserve concept reliably in the strongest steering regime.

## Frozen calibration protocol

- 8 calibration prompts
- seed `37`
- target `blocks.7.hook_resid_post`
- `beta=1`
- methods: additive, full JRR, KL-JRR
- strengths: `0, 1, 1.5, 2, 2.25, 3, 4`
- gate evaluated only at `alpha={2.25,3,4}`
- pass condition at one strong-alpha point:
  - fluency gain versus additive `>= +5`
  - concept loss versus additive `<= 5`

No thresholds, layer, beta, prompt, or seed was changed after observing the result.

## Calibration frontier

| method | F@C70 | F@C75 | F@C80 |
|---|---:|---:|---:|
| additive | 85.66 | 48.71 | 45.49 |
| full JRR | **100.00** | **100.00** | **100.00** |
| KL-JRR | **100.00** | 97.57 | — |

The frontier shows that the selector is promising in the mid-concept regime, but the primary decision criterion was the frozen same-alpha gate, not this interpolated frontier.

## Same-alpha result

| alpha | KL-JRR concept | additive concept | delta concept | KL-JRR fluency | additive fluency | delta fluency | delta NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 70.15 | 49.74 | **+20.41** | 100.00 | 98.01 | +1.99 | -0.074 |
| 1.50 | 75.00 | 74.15 | +0.85 | **97.57** | 83.13 | **+14.44** | **-0.151** |
| 2.00 | 53.36 | 39.36 | **+14.01** | 82.72 | 72.81 | **+9.92** | **-0.128** |
| 2.25 | 61.12 | 37.01 | **+24.11** | 91.94 | 87.90 | **+4.04** | -0.045 |
| 3.00 | 18.19 | 53.63 | **-35.45** | 87.88 | 62.48 | **+25.40** | **-0.332** |
| 4.00 | 51.65 | 83.50 | **-31.85** | 56.45 | 43.24 | **+13.21** | **-0.267** |

The closest strong-regime point was `alpha=2.25`: KL-JRR improved concept substantially and improved fluency by `+4.04`, but this was below the preregistered `+5` fluency gate. The gate therefore correctly remained closed.

At `alpha=3` and `4`, KL-JRR strongly improves fluency/NLL but loses large amounts of concept. This reproduces the core JRR lesson in a sharper form: even removing a small KL-sensitive part of the nonlinear response can alter the autoregressive semantic trajectory substantially.

## Mechanistic diagnostics

KL-JRR is highly selective. It removes only a small fraction of the full orthogonal nonlinear remainder:

| alpha | selected fraction of `R_orth` | KL before | KL after | relative KL reduction |
|---:|---:|---:|---:|---:|
| 1.00 | 5.17% | 0.0154 | 0.0130 | 15.9% |
| 1.50 | 6.55% | 0.0381 | 0.0290 | 23.9% |
| 2.00 | 6.90% | 0.0802 | 0.0546 | 32.0% |
| 2.25 | 7.06% | 0.1106 | 0.0688 | 37.8% |
| 3.00 | 8.99% | 0.2651 | 0.1503 | 43.3% |
| 4.00 | 7.73% | 0.5004 | 0.2812 | 43.8% |

Across the three preregistered strong strengths, the selected component averages only **7.93%** of `R_orth`, yet lowers local KL by **41.6%** on average.

The numerical protection of the transported first-order direction also worked: `sel_transport_dot_removed` stays at approximately `1e-7` scale.

Therefore the selector is not simply recreating full JRR. It finds a compact mode with a large local effect on clean-distribution KL.

## Interpretation

### Supported

1. **The harmful part of the nonlinear response is structured.** A one-dimensional locally KL-sensitive mode accounts for a disproportionate fraction of the local KL divergence while occupying less than 10% of the full orthogonal remainder norm.
2. **Selective removal can improve language-model behavior without weakening the transported first-order steering direction.** NLL and fluency improve at several strengths, and the removed component remains numerically orthogonal to `Jv`.
3. **The useful/harmful decomposition proposed after Experiment 007 is meaningful, but not solved by local KL alone.**

### Not supported

The strong claim that a single local KL-gradient mode cleanly separates fluency-damaging from concept-carrying nonlinear computation is **not supported**. In the strongest steering regime, concept can collapse even when only a small fraction of `R_orth` is removed.

A likely reason is autoregressive compounding: a locally small next-token correction can choose a different token, after which the subsequent activation trajectory and concept realization diverge. Local KL sensitivity is therefore not equivalent to long-horizon semantic irrelevance.

## Decision

The fresh held-out remains untouched because `go_to_new_heldout=false`.

With the submission deadline near, no further post-hoc threshold relaxation or beta/layer sweep should be run. The defensible research story is now stronger if Experiment 008 is retained as a principled negative/partial result:

> Experiment 007 showed that nonlinear steering propagation contains both fluency-damaging and concept-carrying computation. Experiment 008 then showed that a compact KL-sensitive harmful mode exists and can strongly reduce local KL, but local next-token sensitivity alone is insufficient to guarantee long-horizon concept preservation.

This motivates future sequence-aware or concept-aware harmful-mode selection, but it should not be tuned further on the current calibration split for the final submission.
