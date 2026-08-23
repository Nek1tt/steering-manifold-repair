# Final results

The project now has two complementary final contributions:

1. **DPAR** — a practical geometric repair result showing that vanilla activation denoising partly works by cancelling the requested steering direction, and that removing this cancellation preserves alpha exactly while producing local held-out Pareto gains.
2. **JRR (Jacobian Residual Repair)** — a new mechanistic experiment showing that strong steering creates a large, approximately second-order downstream nonlinear remainder. Causal removal of its orthogonal component can restore fluency, but held-out results show that the same nonlinear component can also carry useful concept information.

The most complete archives are:

- [`experiments/retrained_gaussian_followups/`](experiments/retrained_gaussian_followups/)
- [`experiments/jacobian_residual_repair/`](experiments/jacobian_residual_repair/)

## 1. DPAR: validated denoiser geometry

Vanilla activation denoising has a clear mechanistic failure mode: its correction increasingly points against the requested steering vector, so part of its apparent repair comes from weakening the intervention.

Direction-Preserving Activation Repair (DPAR) removes the denoiser correction parallel to the steering vector and preserves requested alpha to numerical precision.

A fresh Gaussian retrain reproduced the original training history exactly, ending at **67.8% held-out relative activation-MSE improvement**.

A dense held-out alpha sweep then showed that downstream gains are real but local rather than universal. At the high-concept operating point `concept >= 90`:

| method | held-out fluency at C90 |
|---|---:|
| additive | 66.46 |
| full DPAR, beta=1.0 | **71.45** |

Descriptive gain: **+4.99 fluency points**.

The direction of the C90 result was consistent across both frozen evaluation seeds, but the sentiment judge is noisy and non-monotonic, so the result should be described as local descriptive evidence rather than uniform domination.

See [`experiments/retrained_gaussian_followups/README.md`](experiments/retrained_gaussian_followups/README.md).

## 2. JRR: downstream nonlinear propagation

The JRR experiment asked a different question: perhaps strong steering fails not primarily because the intervention-layer state is off-manifold, but because a large displacement propagates nonlinearly through later Transformer blocks.

For a downstream map `F`, clean activation `h`, steering direction `v`, and strength `alpha`:

```text
y0      = F(h)
y_alpha = F(h + alpha v)
t       = J_F(h) v
R_alpha = y_alpha - y0 - alpha t
```

`R_alpha` is the exact nonlinear Taylor remainder.

### Mechanistic result

Calibration selected `blocks.7.hook_resid_post`. At that layer:

| diagnostic | value |
|---|---:|
| log-log slope of `||R_alpha||` vs alpha | **1.9849** |
| mean orthogonal fraction of remainder | **0.9404** |
| rank correlation `||R_orth||` vs NLL | **+0.8909** |
| rank correlation `||R_orth||` vs fluency | **-0.8909** |

The slope is strikingly close to the second-order Taylor prediction `O(alpha^2)`.

On held-out autoregressive trajectories the nonlinear effect becomes extremely large. At `alpha=3`:

- mean `||R|| = 40.14`;
- mean `||R_orth|| = 39.58`;
- mean `||Jv|| = 13.56`;
- therefore `||R|| / ||alpha Jv|| = 0.986`.

So by strong steering, the downstream nonlinear response is almost as large as the entire first-order transported steering effect.

### Causal oracle

JRR removes the component of the exact nonlinear remainder orthogonal to `Jv`:

```text
y_repaired = y_alpha - R_orth
```

with frozen `beta=1`.

Calibration was strongly positive: JRR reached **F@C80=100.00** versus additive **45.49**, a calibration gain of **+54.51** fluency points, and therefore opened the frozen held-out evaluation.

### Held-out result

The preregistered C80/C85/C90 held-out frontier was **not estimable** under the manual oracle generation protocol: after combining the two frozen seeds, additive reached maximum concept 77.30 and JRR reached 74.37.

Therefore the calibration +54.51 must **not** be presented as a confirmed held-out effect size.

Within the observed strong-steering regime, however, JRR shows a clear fluency/NLL repair signal:

| alpha | additive concept | JRR concept | delta concept | additive fluency | JRR fluency | delta fluency | delta NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.25 | 68.68 | 62.21 | -6.46 | 68.88 | **88.47** | **+19.59** | **-0.250** |
| 3.00 | 77.30 | 74.37 | -2.93 | 54.68 | **68.73** | **+14.05** | **-0.229** |

Paired prompt/seed bootstrap diagnostics give 95% intervals for delta NLL that remain below zero:

- `alpha=2.25`: **[-0.394, -0.110]**
- `alpha=3.00`: **[-0.395, -0.071]**

These bootstrap intervals are post-hoc descriptive diagnostics, not a preregistered significance test.

### Seed sensitivity reveals the mechanism

At `alpha=3`, the two frozen seeds behave very differently:

| seed | delta concept | delta fluency |
|---:|---:|---:|
| 11 | **+8.76** | **+21.81** |
| 23 | **-14.62** | **+5.45** |

Thus full orthogonal remainder removal is not a robust Pareto-dominating method.

But this negative practical result is mechanistically informative:

> **Orthogonal-to-`Jv` is not equivalent to irrelevant-to-concept.**

The nonlinear response appears to mix at least two components:

- harmful collateral distortion that damages fluency;
- useful nonlinear adaptation that helps realize the target concept.

Removing all of `R_orth` can recover fluency while also deleting useful semantic computation. Preserving the first-order transported direction alone is therefore insufficient to preserve the concept.

See [`experiments/jacobian_residual_repair/RESULTS.md`](experiments/jacobian_residual_repair/RESULTS.md) for the full analysis and archived compact evidence.

## 3. Final scientific claim

The strongest defensible combined claim is:

> Vanilla learned denoising has a geometric steering-cancellation failure mode, which DPAR removes exactly. Separately, strong activation steering creates a measurable downstream nonlinear Taylor remainder that grows approximately quadratically and becomes comparable to the first-order steering effect. Causal removal shows that this nonlinear remainder genuinely contains fluency-damaging components, but also reveals that orthogonal nonlinear computation can carry useful concept information. Consequently, coherence-preserving steering requires preserving more than the original steering axis or its first-order transported image.

This is stronger than claiming one universal repair algorithm: it identifies **two distinct failure modes** and shows why simple projection-based fixes are insufficient.

## 4. Recommended next method

Do not train an adapter to imitate the entire JRR `R_orth` target yet.

The next research direction is **harmful-mode-selective nonlinear repair**: decompose `R_orth` into components associated with fluency degradation versus components necessary for concept realization, and remove only the former.

Promising diagnostics include:

- local logit-sensitivity / Fisher-weighted residual geometry;
- low-rank harmful-mode discovery across prompts;
- cross-direction agreement of harmful residual modes;
- causal ablations inside the nonlinear residual subspace.

If such a selective oracle works, that target can then be amortized into the final lightweight adapter/checkpoint for Hugging Face.

## 5. Reproduction

DPAR fresh-runtime reproduction:

```text
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
```

JRR experiment:

```text
notebooks/jrr_experiment_colab.ipynb
```

Reference Gaussian validation relative MSE improvement:

```text
0.6781128741849923
```

The final Hugging Face adapter/checkpoint is still a packaging step to complete once the final method is frozen.
