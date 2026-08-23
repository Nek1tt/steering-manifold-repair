# Experiment 007 — Jacobian Residual Repair (JRR)

Date: 2026-08-23

## Research question

Strong activation steering is normally treated as a problem at the intervention layer itself. JRR tests a different hypothesis: the main fluency failure may be created **after** the intervention, when a large activation displacement propagates nonlinearly through later Transformer blocks.

Let `F` map the source residual state at layer 6 to a downstream residual state. For clean state `h`, steering direction `v`, and strength `alpha`:

```text
y0      = F(h)
y_alpha = F(h + alpha v)
t       = J_F(h) v
R_alpha = y_alpha - y0 - alpha t
```

`t` is the first-order transported steering direction. `R_alpha` is the exact nonlinear Taylor remainder.

The central mechanistic hypothesis is:

1. useful concept steering is carried substantially by the first-order term `alpha * t`;
2. fluency degradation grows with the nonlinear remainder;
3. the most harmful part is the component of the remainder orthogonal to `t`.

The proposed oracle intervention is therefore

```text
R_parallel = proj_t(R_alpha)
R_orth     = R_alpha - R_parallel

y_repaired = y_alpha - beta * R_orth
```

with `beta=1` in the first decisive test.

This differs from DPAR. DPAR constrains a learned denoiser correction at the steering layer. JRR directly measures the **downstream nonlinear effect caused by the model's own dynamics** and removes only the part collateral to the locally transported steering direction.

## Scientific protocol

The experiment has two hard-separated stages.

### Stage A — diagnostic, calibration prompts only

Run exact directional JVPs at downstream layers 7–11 and measure `R_alpha` over the alpha grid. Separately run short additive generations on the same calibration prompts.

We test three predictions:

- `||R_alpha||` should grow superlinearly; a log-log slope near 2 is especially supportive of a second-order regime;
- `||R_orth||` should correlate positively with NLL and/or negatively with fluency;
- a nontrivial fraction of the nonlinear remainder should be orthogonal to `Jv`.

The code selects a candidate downstream layer using only calibration data and writes an explicit go/no-go decision.

### Stage B — causal oracle repair

Only if Stage A is positive, generate with the exact per-token counterfactual remainder and subtract `R_orth` at the selected downstream layer.

Calibration compares:

```text
additive
jrr_orth
```

If JRR improves the interpolated concept/fluency frontier by at least 2 fluency points at one frozen concept threshold, the held-out evaluation unlocks.

Held-out evaluation then uses the original `data/prompts.txt` and seeds `11,23`. Do not tune after looking at those results.

## Why this starts as an oracle

The exact JRR intervention is intentionally expensive: every generated token needs clean/counterfactual model evaluations plus a directional derivative. That is a feature of this experiment, not a final deployment design.

The oracle answers the causal question first:

> If the nonlinear collateral remainder were known exactly, would removing it recover fluency while preserving steering?

Only after a positive oracle result should an amortized residual adapter be trained and published as the final Hugging Face checkpoint.

## Files

Implementation:

```text
configs/jrr_gpt2.yaml
src/steering_repair/jrr.py
src/steering_repair/jrr_diagnostic.py
src/steering_repair/jrr_oracle.py
scripts/run_jrr_diagnostic.py
scripts/run_jrr_oracle.py
tests/test_jrr.py
notebooks/jrr_experiment_colab.ipynb
```

Generated outputs under `results/jrr/`:

```text
diagnostic_samples.csv
diagnostic_aggregate.csv
behavior_samples.csv
behavior_aggregate.csv
target_layer_summary.csv
diagnostic_summary.json
DIAGNOSTIC.md
remainder_scaling.png
orthogonal_remainder_fraction.png
orthogonal_remainder_vs_fluency.png

oracle_calibration_samples.csv
oracle_calibration_aggregate.csv
oracle_calibration_frontier.csv
oracle_calibration_summary.json
ORACLE_CALIBRATION.md
oracle_calibration_pareto.png

# only after calibration passes:
oracle_evaluation_samples.csv
oracle_evaluation_aggregate.csv
oracle_evaluation_frontier.csv
oracle_evaluation_summary.json
oracle_evaluation_pareto.png
```

## Run instructions

### 0. Install and make sure the frozen sentiment direction exists

```bash
pip install -r requirements.txt
pip install -e .

python scripts/validate_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

If `results/sentiment_direction.pt` already exists from the frozen baseline, do not rebuild or retune it.

### 1. Unit tests

```bash
pytest -q tests/test_jrr.py tests/test_inference_followups.py tests/test_denoiser.py
```

The JRR tests verify:

- parallel/orthogonal decomposition;
- the protected transported component;
- the full-linearization ablation;
- exact `alpha^2` Taylor remainder on a synthetic quadratic system;
- agreement of autograd JVP with a central finite difference.

### 2. Run the cheap mechanistic diagnostic

```bash
python scripts/run_jrr_diagnostic.py --config configs/jrr_gpt2.yaml
```

Then inspect:

```text
results/jrr/DIAGNOSTIC.md
results/jrr/target_layer_summary.csv
results/jrr/remainder_scaling.png
results/jrr/orthogonal_remainder_vs_fluency.png
```

The most important fields are:

- `loglog_residual_slope` — near 2 supports second-order growth;
- `rank_corr_orthogonal_residual_vs_nll` — positive is supportive;
- `rank_corr_orthogonal_residual_vs_fluency` — negative is supportive;
- `mean_orthogonal_fraction` — tells us whether JRR has meaningful collateral distortion to remove;
- `oracle_recommended` — automatic compute gate.

If autograd JVP is unsupported by the installed TransformerLens/PyTorch combination, the experiment automatically falls back to central finite differences and records that fact in `jvp_modes_used`.

### 3. Run oracle calibration only if the diagnostic is promising

```bash
python scripts/run_jrr_oracle.py \
  --config configs/jrr_gpt2.yaml \
  --phase calibration
```

Inspect:

```text
results/jrr/ORACLE_CALIBRATION.md
results/jrr/oracle_calibration_pareto.png
```

The script refuses to run this expensive stage if the diagnostic gate is negative. `--force` exists only for a deliberate ablation and should not be used to manufacture a positive result.

### 4. Held-out evaluation — only after calibration passes

```bash
python scripts/run_jrr_oracle.py \
  --config configs/jrr_gpt2.yaml \
  --phase evaluation
```

This command is locked unless `oracle_calibration_summary.json` contains `go_to_heldout: true`.

## Decision table

### Strong positive

Diagnostic signal is clear and oracle JRR improves the frontier.

Next experiment: train a small adapter to predict `R_orth` from `(h, v, alpha)` across multiple training steering directions. This becomes the efficient method and Hugging Face checkpoint.

### Mechanistic positive, oracle negative

`R_orth` strongly tracks degradation but removing it does not restore fluency.

Conclusion: nonlinear collateral propagation is a marker of steering failure but not by itself the causal bottleneck. This is still a useful mechanistic result; do not train an adapter.

### Diagnostic negative

The remainder either stays small, is mostly transported-direction parallel, or fails to track behavioral degradation.

Stop JRR immediately. The hypothesis is falsified cheaply, without touching held-out evaluation.

## Important methodological constraint

Do not tune target layer, beta, thresholds, prompts, or alpha grid after seeing held-out results. The entire purpose of the diagnostic/calibration split is to make a positive result defensible rather than post-hoc.
