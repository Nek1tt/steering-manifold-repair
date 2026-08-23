# Experiment 008 — KL-Selective Jacobian Residual Repair

Date: 2026-08-23

## Motivation

Experiment 007 (JRR) produced a useful surprise. Strong activation steering creates a downstream nonlinear Taylor remainder that grows almost exactly quadratically, and removing the component orthogonal to the transported first-order steering direction can recover fluency. However, full removal is not robustly concept-preserving: on one frozen seed it improved both concept and fluency, while on another it improved fluency but removed substantial concept signal.

The key mechanistic lesson is:

> **Orthogonal-to-`Jv` is not equivalent to irrelevant-to-concept.**

Therefore the next method should not remove all nonlinear computation. It should remove only nonlinear modes that look locally harmful to language-model behavior.

## Method: KL-Selective JRR

For clean source activation `h`, steering direction `v`, strength `alpha`, and a downstream map `F`:

```text
y0      = F(h)
y_alpha = F(h + alpha v)
t       = J_F(h) v
R       = y_alpha - y0 - alpha t
R_orth  = R - proj_t(R)
```

Full JRR uses `y_alpha - R_orth`.

KL-Selective JRR instead asks which part of `R_orth` is locally responsible for moving the model away from the clean next-token distribution.

Let

```text
p0 = p(next token | current prefix, clean model)
q(y) = p(next token | current prefix, target residual y)
L_KL(y) = KL(p0 || q(y))
g = d L_KL / d y evaluated at y_alpha
```

The clean distribution is used only as a local fluency reference on the *current generated prefix*. It does not provide a target continuation.

To preserve the first-order transported steering effect, first remove the component of the KL gradient parallel to `t`:

```text
g_perp = g - proj_t(g)
```

Then select only the component of `R_orth` aligned with the KL-increasing direction:

```text
c = max(0, <R_orth, g_perp> / ||g_perp||^2)
R_harm = c g_perp
```

Finally:

```text
y_repaired = y_alpha - beta R_harm
```

with frozen `beta=1`.

The `max(0, ...)` gate is important: if a nonlinear residual component locally moves the model *toward* the clean distribution, it is preserved rather than "repaired".

## Why this is a direct test of the JRR failure mode

Experiment 007 implicitly treated all `R_orth` as collateral damage. Experiment 008 tests a more selective hypothesis:

```text
R_orth = R_harmful + R_useful
```

where:

- `R_harmful` increases local divergence from the clean language model and damages fluency;
- `R_useful` is nonlinear computation needed to realize the steered concept.

KL-JRR is intentionally a one-dimensional adaptive selector at each token. It is not yet a learned adapter and has no trainable parameters.

## Important hold-out rule

The original 20-prompt held-out set from Experiment 007 is **not confirmatory for this method**, because the design of KL-JRR was motivated by those results.

Therefore Experiment 008 commits a new untouched split:

```text
data/selective_jrr_heldout_prompts.txt
```

with new seeds `101, 211`.

Do not modify those prompts, seeds, target layer, beta, or strong-alpha set after evaluation begins.

## Frozen protocol

Target layer is inherited from JRR calibration:

```text
blocks.7.hook_resid_post
```

No layer sweep and no beta sweep are performed.

### Calibration

```text
prompts: data/calibration_prompts.txt
8 prompts
seed: 37
methods: additive, full JRR, KL-JRR
alpha: 0, 1, 1.5, 2, 2.25, 3, 4
32 new tokens
```

Primary calibration gate is deliberately same-alpha rather than interpolated frontier. At one of `alpha={2.25,3,4}`, KL-JRR must simultaneously:

- improve fluency over additive by at least `+5` points;
- lose no more than `5` concept points versus additive.

This gate directly tests the intended improvement over full JRR: recover fluency without deleting large amounts of concept signal.

### Fresh held-out

Only if calibration passes:

```text
12 new prompts
seeds: 101, 211
methods: additive, full JRR, KL-JRR
alpha: 0, 2.25, 3, 4
32 new tokens
```

The held-out is intentionally focused on the strong-steering region where Experiment 007 found the nonlinear remainder comparable in norm to the first-order effect.

## Files

```text
configs/selective_jrr_gpt2.yaml
data/selective_jrr_heldout_prompts.txt
src/steering_repair/selective_jrr.py
scripts/preflight_selective_jrr.py
scripts/run_selective_jrr.py
tests/test_selective_jrr.py
notebooks/selective_jrr_experiment_colab.ipynb
```

Generated outputs:

```text
results/selective_jrr/calibration_samples.csv
results/selective_jrr/calibration_aggregate.csv
results/selective_jrr/calibration_frontier.csv
results/selective_jrr/calibration_same_alpha.csv
results/selective_jrr/calibration_summary.json
results/selective_jrr/calibration_pareto.png

# only after calibration passes:
results/selective_jrr/evaluation_samples.csv
results/selective_jrr/evaluation_aggregate.csv
results/selective_jrr/evaluation_frontier.csv
results/selective_jrr/evaluation_same_alpha.csv
results/selective_jrr/evaluation_summary.json
results/selective_jrr/evaluation_pareto.png
```

## Mechanistic diagnostics recorded

For KL-JRR every generated trajectory records token-averaged:

- `sel_remainder_norm`;
- `sel_orthogonal_norm`;
- `sel_selected_norm`;
- `sel_selected_fraction` — how much of full `R_orth` was actually removed;
- `sel_kl_gradient_norm` and `sel_kl_gradient_orth_norm`;
- `sel_alignment` between `R_orth` and the KL-increasing mode;
- `sel_kl_before` and `sel_kl_after`;
- `sel_transport_dot_removed` — numerical check that correction remains orthogonal to `Jv`.

A successful result should not only improve text metrics. It should show that KL-JRR removes a small/structured fraction of `R_orth`, lowers local KL, and preserves concept better than full JRR.

## Run

```bash
pip install -r requirements.txt
pip install -e .

pytest -q tests/test_selective_jrr.py tests/test_jrr.py

python scripts/preflight_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml

python scripts/run_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml \
  --phase calibration
```

Inspect `results/selective_jrr/calibration_summary.json`.

Only if `go_to_new_heldout` is true:

```bash
python scripts/run_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml \
  --phase evaluation
```

Do not use `--force` for the main reported result.

## Decision table

### Selective repair succeeds

KL-JRR retains the fluency improvement of full JRR while preserving more concept on the fresh held-out.

This supports the claim that downstream nonlinear response contains separable harmful and useful components. The next engineering step is to amortize the harmful-mode selector into a lightweight adapter/checkpoint.

### KL-JRR behaves like full JRR

The local KL gradient does not separate harmful from concept-carrying nonlinear computation. The decomposition hypothesis may still be correct, but a one-dimensional local clean-distribution sensitivity is insufficient.

### KL-JRR behaves like additive

The selected harmful component is too small or too conservative. This would imply that the fluency-repair signal from full JRR is distributed across multiple nonlinear modes rather than concentrated along the local KL gradient.
