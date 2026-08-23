# Steering Manifold Repair

Research project for the interpretability task: reproduce activation-steering degradation, then test cheap learned and geometric ways to push the concept/fluency Pareto frontier up and to the right.

## Status

The required additive steering baseline has been reproduced successfully on GPT-2 Small at the midpoint layer.

The first SAE-feature attempt is preserved as a negative result in:

```text
experiments/failed_sae_profanity/
```

The successful control experiment is preserved in:

```text
experiments/successful_sentiment_baseline/
```

It uses a persona-style contrastive sentiment direction

\[
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}]
\]

at `blocks.6.hook_resid_post`, with literal response-token steering

\[
h' = h + \alpha v.
\]

The frozen baseline produced the expected trade-off: positive-sentiment concept score rises from about 28 to above 90 while stronger steering eventually collapses fluency. The main comparison region for repair methods is `alpha=0.5..4.0`.

## New research stage: repair hypotheses

The repository now implements the assignment-proposed denoiser plus two extensions intended to test something genuinely new.

### B2 — Gaussian activation denoiser

Train a small residual MLP on generic natural layer-6 activations only:

\[
\tilde h = h + \delta,\qquad D_\theta(\tilde h, r)\approx h,
\]

where corruption magnitude is parameterized by

\[
r = \|\delta\|/\|h\|.
\]

The network is conditioned on `r`, so one checkpoint can handle weak through severe steering.

### M1 — Direction-Preserving Activation Repair (DPAR)

For a steered activation `z = h + alpha*v`, let the denoiser propose

\[
\Delta = D(z)-z.
\]

A vanilla denoiser may recover fluency by simply cancelling the intended direction. DPAR removes the correction parallel to the steering vector:

\[
\Delta_\perp = \Delta - \operatorname{proj}_v(\Delta),
\]

\[
h_{\mathrm{DPAR}} = z + \Delta_\perp.
\]

This preserves the requested steering component by construction while still allowing orthogonal manifold repair.

### M2 — Structured corruption training

Steering perturbations are directional rather than isotropic. A second denoiser is therefore trained on a 50/50 mixture of:

- isotropic Gaussian directions;
- normalized random natural activation differences `h_j - h_k`.

The held-out sentiment steering direction is never used in denoiser training.

## Methods evaluated

The frozen suite compares:

```text
additive
norm_preserving
gaussian
gaussian_lambda05
gaussian_dpar
mixed
mixed_dpar
```

`gaussian_lambda05` is a partial direction-preservation ablation. `mixed_dpar` is the full structured-corruption + DPAR candidate method.

## Mechanistic analysis

The repair evaluator records more than final text scores. For every learned intervention it measures:

- requested vs effective alpha after repair;
- cosine between the denoiser correction and steering vector;
- fraction of correction norm parallel to the steering vector;
- correction norm relative to the steering perturbation;
- clean-model NLL, distinct-1/2/3 and 3-gram repetition;
- independent local sentiment concept score.

This directly tests whether a method actually repairs off-manifold damage or only weakens steering.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

A Hugging Face token is optional but recommended in Colab for higher download limits.

## Reproduce the successful additive baseline

```bash
python scripts/validate_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml

python scripts/run_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml

python scripts/plot_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

Once it has passed, do not retune prompts, seeds, vector data, judge, or the fluency definition.

## Run the full repair suite

The recommended Colab entry point is:

```text
notebooks/repair_experiments_colab.ipynb
```

Or run everything from the CLI:

```bash
python scripts/run_repair_suite.py \
  --config configs/repair_suite_gpt2.yaml
```

The full runner performs:

1. cache up to 80k generic WikiText-2 layer-6 activations;
2. train the Gaussian denoiser;
3. train the mixed structured denoiser;
4. evaluate all additive / learned / DPAR methods on the frozen sentiment baseline;
5. build Pareto and mechanistic plots;
6. write a descriptive hypothesis report.

Individual stages can be run separately:

```bash
python scripts/cache_activations.py --config configs/repair_suite_gpt2.yaml
python scripts/train_denoiser.py --config configs/repair_suite_gpt2.yaml --kind gaussian
python scripts/train_denoiser.py --config configs/repair_suite_gpt2.yaml --kind mixed
python scripts/eval_repairs.py --config configs/repair_suite_gpt2.yaml
python scripts/plot_repairs.py --config configs/repair_suite_gpt2.yaml
```

## Important outputs

```text
results/layer6_generic_activations.pt
checkpoints/denoiser_gaussian.pt
checkpoints/denoiser_mixed.pt
results/repair_suite_samples.csv
results/repair_suite/repair_pareto.png
results/repair_suite/effective_alpha.png
results/repair_suite/correction_geometry.png
results/repair_suite/frontier_summary.csv
results/repair_suite/hypothesis_report.md
```

Generated caches, checkpoints and raw results are gitignored. The best final adapter/checkpoint should later be uploaded to Hugging Face as required by the assignment.

## Repository layout

```text
configs/
  baseline_sentiment_gpt2.yaml
  repair_suite_gpt2.yaml

data/
  sentiment_positive.txt
  sentiment_negative.txt
  calibration_prompts.txt
  prompts.txt

experiments/
  failed_sae_profanity/
  successful_sentiment_baseline/
  repair_suite/

notebooks/
  baseline_colab.ipynb
  repair_experiments_colab.ipynb

scripts/
  validate_sentiment_baseline.py
  run_sentiment_baseline.py
  plot_sentiment_baseline.py
  cache_activations.py
  train_denoiser.py
  eval_repairs.py
  plot_repairs.py
  run_repair_suite.py

src/steering_repair/
  sentiment_baseline.py
  activation_cache.py
  denoiser.py
  train_denoiser.py
  repair_experiment.py
  steering.py
  generation.py
  metrics.py
```

## Reproducibility rules

- Keep the successful additive baseline frozen for all repair comparisons.
- Never expose the held-out evaluation steering vector to denoiser training.
- Compare methods by Pareto frontier, not by the same raw alpha only.
- Always compare a repaired point with smaller-alpha additive steering so cancellation cannot masquerade as repair.
- Preserve negative results and ablations.

## References

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- OpenAI sparse autoencoder: https://github.com/openai/sparse_autoencoder
- SAELens: https://github.com/decoderesearch/SAELens
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Generative Latent Prior: https://generative-latent-prior.github.io/
