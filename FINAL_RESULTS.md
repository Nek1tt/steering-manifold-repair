# Final results

The project has one practical repair contribution and two linked mechanistic investigations:

1. **DPAR** — a practical geometric repair showing that vanilla activation denoising partly works by cancelling the requested steering direction. Removing this cancellation preserves alpha exactly and produces local held-out Pareto gains.
2. **JRR (Jacobian Residual Repair)** — a mechanistic experiment showing that strong steering creates a large, approximately second-order downstream nonlinear remainder. Causal removal restores fluency in the strong-steering regime, but the same nonlinear component can also carry useful concept information.
3. **KL-Selective JRR** — a preregistered follow-up testing whether a compact local KL-sensitive mode can separate harmful from useful nonlinear computation. It finds a highly structured harmful mode and reduces local KL strongly, but fails the frozen strong-regime concept/fluency gate; the fresh held-out was therefore not opened.

Complete archives:

- [`experiments/retrained_gaussian_followups/`](experiments/retrained_gaussian_followups/)
- [`experiments/jacobian_residual_repair/`](experiments/jacobian_residual_repair/)
- [`experiments/selective_jrr/`](experiments/selective_jrr/)

## 1. DPAR: validated denoiser geometry

Vanilla activation denoising has a clear failure mode: its correction increasingly points against the requested steering vector, so part of its apparent repair comes from weakening the intervention.

Direction-Preserving Activation Repair (DPAR) removes the denoiser correction parallel to the steering vector and preserves requested alpha to numerical precision.

A fresh Gaussian retrain reproduced the original training history exactly, ending at **67.8% held-out relative activation-MSE improvement**.

A dense held-out alpha sweep showed local rather than uniform gains. At the high-concept operating point `concept >= 90`:

| method | held-out fluency at C90 |
|---|---:|
| additive | 66.46 |
| full DPAR, beta=1.0 | **71.45** |

Descriptive gain: **+4.99 fluency points**.

The direction of this C90 result was consistent across both frozen evaluation seeds. Because the sentiment judge is noisy and non-monotonic across alpha, this is reported as local descriptive evidence rather than universal domination.

See [`experiments/retrained_gaussian_followups/README.md`](experiments/retrained_gaussian_followups/README.md).

## 2. JRR: nonlinear downstream propagation

JRR asks whether strong steering fails partly because a large displacement propagates nonlinearly through later Transformer blocks.

For downstream map `F`, clean state `h`, steering direction `v`, and strength `alpha`:

```text
y0      = F(h)
y_alpha = F(h + alpha v)
t       = J_F(h) v
R_alpha = y_alpha - y0 - alpha t
```

At the calibration-selected `blocks.7.hook_resid_post`:

| diagnostic | value |
|---|---:|
| log-log slope of `||R_alpha||` vs alpha | **1.9849** |
| mean orthogonal fraction of remainder | **0.9404** |
| rank correlation `||R_orth||` vs NLL | **+0.8909** |
| rank correlation `||R_orth||` vs fluency | **-0.8909** |

The slope is strikingly close to the second-order Taylor prediction `O(alpha^2)`.

On held-out autoregressive trajectories, at `alpha=3`:

- mean `||R|| = 40.14`;
- mean `||R_orth|| = 39.58`;
- mean `||Jv|| = 13.56`;
- `||R|| / ||alpha Jv|| = 0.986`.

Thus the nonlinear downstream response becomes almost as large as the entire first-order transported steering effect.

### Causal oracle result

Full JRR removes the exact orthogonal remainder:

```text
y_repaired = y_alpha - R_orth
```

Calibration was strongly positive (`F@C80=100.00` versus additive `45.49`) and opened frozen held-out evaluation.

The preregistered C80/C85/C90 held-out frontier itself was not estimable because the manual oracle generation regime did not reach C80 after combining both frozen seeds. Therefore the calibration `+54.51` must not be presented as a confirmed held-out effect size.

Within observed strong steering, however, the causal fluency repair is clear:

| alpha | delta concept | delta fluency | delta NLL |
|---:|---:|---:|---:|
| 2.25 | -6.46 | **+19.59** | **-0.250** |
| 3.00 | -2.93 | **+14.05** | **-0.229** |

Post-hoc paired prompt/seed bootstrap intervals for delta NLL remain below zero:

- `alpha=2.25`: `[-0.394, -0.110]`
- `alpha=3.00`: `[-0.395, -0.071]`

But seed sensitivity reveals an important limitation. At `alpha=3`:

| seed | delta concept | delta fluency |
|---:|---:|---:|
| 11 | **+8.76** | **+21.81** |
| 23 | **-14.62** | **+5.45** |

The mechanistic lesson is therefore:

> **Orthogonal-to-`Jv` is not equivalent to irrelevant-to-concept.**

The nonlinear response mixes fluency-damaging collateral computation with useful nonlinear concept realization.

See [`experiments/jacobian_residual_repair/RESULTS.md`](experiments/jacobian_residual_repair/RESULTS.md).

## 3. KL-Selective JRR: testing harmful-mode separation

Experiment 008 directly tests the lesson above. Rather than removing all `R_orth`, it computes the local clean-distribution KL gradient at the steered downstream state, projects that gradient orthogonal to `Jv`, and removes only the component of `R_orth` aligned with the KL-increasing direction.

No layer sweep or beta sweep was used. The new method was designed after Experiment 007, so a completely new held-out prompt set and new seeds `101/211` were frozen before calibration.

### Mechanistic result

The selector is genuinely compact. Across the preregistered strong strengths `alpha={2.25,3,4}` it removes on average only **7.93%** of `R_orth`, while lowering local clean-distribution KL by **41.6%** on average.

Examples:

| alpha | selected fraction | KL before | KL after | KL reduction |
|---:|---:|---:|---:|---:|
| 2.25 | 7.06% | 0.1106 | 0.0688 | 37.8% |
| 3.00 | 8.99% | 0.2651 | 0.1503 | 43.3% |
| 4.00 | 7.73% | 0.5004 | 0.2812 | 43.8% |

The correction remains numerically orthogonal to the transported first-order direction (`sel_transport_dot_removed` at approximately `1e-7` scale).

### Calibration result

KL-JRR is strong in the mid-strength regime:

| alpha | delta concept vs additive | delta fluency | delta NLL |
|---:|---:|---:|---:|
| 1.50 | +0.85 | **+14.44** | **-0.151** |
| 2.00 | **+14.01** | **+9.92** | **-0.128** |
| 2.25 | **+24.11** | +4.04 | -0.045 |

But the frozen gate required at least `+5` fluency with no more than `5` concept loss at one of `alpha={2.25,3,4}`. `alpha=2.25` missed the fluency threshold by about one point, while `alpha=3/4` improved fluency strongly but lost substantial concept.

Therefore:

```text
go_to_new_heldout = false
```

The fresh held-out was intentionally **not opened**. We do not relax the `+5` gate post hoc.

This result supports a narrower claim:

> A small KL-sensitive nonlinear mode explains a disproportionate share of local next-token divergence, but local next-token sensitivity alone is insufficient to guarantee long-horizon concept preservation under autoregressive generation.

See [`experiments/selective_jrr/RESULTS.md`](experiments/selective_jrr/RESULTS.md).

## 4. Combined scientific claim

The strongest defensible story is not that one repair universally solves steering degradation. Instead, the experiments isolate two distinct failure mechanisms and progressively falsify oversimplified fixes:

> Vanilla learned denoising has a steering-cancellation failure mode, which DPAR removes exactly. Separately, strong activation steering creates an approximately quadratic downstream nonlinear response that becomes comparable to the first-order steering effect. Causal intervention shows that this nonlinear response contains real fluency-damaging computation, but also concept-carrying nonlinear computation. A compact local KL-sensitive component captures much of the local divergence, yet sequence-level concept can still depend sensitively on that component. Coherence-preserving steering therefore requires preserving more than the original steering axis, its first-order transported image, or a purely local clean-distribution objective.

This combination of positive and negative causal evidence is the main mechanistic contribution.

## 5. What not to do before submission

With the final deadline near, do **not**:

- relax the Experiment 008 gate from `+5` to `+4` after seeing calibration;
- open its fresh held-out with `--force`;
- sweep beta or target layer post hoc;
- claim JRR or KL-JRR uniformly dominates additive steering;
- replace the frozen final claim with an unvalidated late experiment.

A future method should be sequence-aware or concept-aware rather than another local projection rule, but it should be evaluated on a genuinely new protocol rather than tuned further on the current calibration data.

## 6. Reproduction

DPAR fresh-runtime reproduction:

```text
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
```

JRR:

```text
notebooks/jrr_experiment_colab.ipynb
```

KL-Selective JRR, Windows/VS Code:

```text
notebooks/selective_jrr_vscode_windows.ipynb
```

Reference Gaussian validation relative MSE improvement:

```text
0.6781128741849923
```

## 7. Submission checkpoint

The best practical learned checkpoint is the deterministically reproduced Gaussian activation denoiser used with DPAR geometry.

**Public Hugging Face repository:** [Nek1tt/steering-repair-gpt2](https://huggingface.co/Nek1tt/steering-repair-gpt2)

The model repository contains `retrained_denoiser_gaussian.pt`, a model card, checkpoint metadata, the frozen training configuration, and the training history. DPAR is an inference-time projection of the denoiser correction and is therefore documented with the checkpoint rather than encoded into its weights.
