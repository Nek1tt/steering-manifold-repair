# Experiment 007 — JRR first calibration attempt

Date: 2026-08-23

Status: **mechanistic diagnostic positive; first oracle calibration gate invalid rather than negative.**

## Stage A: nonlinear propagation diagnostic

The calibration-only diagnostic selected `blocks.7.hook_resid_post`.

Key values for the selected layer:

- log-log slope of `||R_alpha||` vs alpha: **1.984864**;
- mean orthogonal fraction of the nonlinear remainder: **0.940438**;
- rank correlation `||R_orth||` vs NLL: **+0.890909**;
- rank correlation `||R_orth||` vs fluency: **-0.890909**.

All candidate downstream layers showed a large, predominantly orthogonal nonlinear remainder. The near-quadratic slope at layer 7 is strong evidence that the intervention develops a second-order downstream component as steering strength grows.

Important qualification: the residual/behavior correlations are computed across alpha and are therefore partly confounded by steering strength itself. They motivated the causal oracle test but are not causal evidence by themselves.

## Stage B attempt 1: exact oracle repair

Frozen method:

```text
source: blocks.6.hook_resid_post
target: blocks.7.hook_resid_post
beta: 1.0
methods: additive, jrr_orth
```

The first oracle calibration used 6 prompts, 24 generated tokens and strengths `[0, 1, 1.5, 2, 2.5, 3]`.

The pre-registered frontier thresholds were C80/C85/C90. However, under this shorter/coarser sampling regime the additive control reached only **67.95** mean concept at its best point. Consequently neither method reached any frozen threshold and the entire frontier table was NaN:

```text
method      C80   C85   C90
additive    NaN   NaN   NaN
jrr_orth    NaN   NaN   NaN
```

Therefore `selected_method=None` and `go_to_heldout=false` do **not** constitute evidence that JRR failed the causal test. The gate was undefined because its control never entered the comparison region.

## Preliminary behavior inside the invalid calibration run

These values are descriptive only; they are not used to unlock held-out evaluation.

| alpha | Δ concept JRR-additive | Δ fluency JRR-additive | Δ NLL JRR-additive |
|---:|---:|---:|---:|
| 1.0 | +14.689 | +1.877 | -0.019 |
| 1.5 | -21.341 | -11.623 | +0.123 |
| 2.0 | -11.120 | -9.262 | +0.131 |
| 2.5 | +16.144 | -6.467 | +0.093 |
| 3.0 | +0.789 | -8.119 | +0.150 |

The only locally encouraging point is alpha=1.0; at stronger alphas full `beta=1` removal is usually worse in fluency. With only six prompts and a high-variance sentiment judge this must not be treated as a tuned operating point.

The exact oracle diagnostics also show that the correction becomes very large: mean `||R_orth||` grows from about 4.0 at alpha=1 to about 40.5 at alpha=3, while `||Jv||` stays around 14. This makes the causal question especially important: the nonlinear response is large, but it may contain useful task-relevant dynamics rather than pure collateral damage.

## Protocol correction before attempt 2

The Stage-A behavior probe used 8 calibration prompts, 32 generated tokens and the denser alpha grid `[0, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4]`. Under that exact calibration regime the additive control reached concept above 87 (and 99 at alpha=4), so the frozen C80/C85/C90 thresholds are meaningful.

Attempt 2 therefore changes **only the oracle calibration sampling regime** to match Stage A:

- 8 calibration prompts;
- 32 generated tokens;
- the same dense alpha grid including alpha=4.

The following remain frozen:

- JRR definition;
- target layer selected by Stage A;
- `beta=1`;
- methods;
- judge;
- C80/C85/C90 thresholds;
- +2 fluency-point gate;
- held-out prompts and seeds.

This is a protocol-validity correction, not post-hoc method tuning. Held-out evaluation remains locked until a valid calibration gate is obtained.
