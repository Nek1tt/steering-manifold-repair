# Steering Manifold Repair

Research scaffold for the interpretability task: first reproduce a valid activation-steering concept/fluency Pareto frontier, then study cheap ways to repair the fluency degradation.

## Current status

The first SAE-feature reproduction attempt is preserved as a **failed baseline-vector validation** in:

```text
experiments/failed_sae_profanity/
```

That run strongly changed GPT-2 behavior but never increased the intended generated-text concept, so it is not used as the control experiment.

The active reproduction baseline is now a **persona-style contrastive sentiment direction** on GPT-2 Small at the midpoint layer:

\[
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}],
\]

followed by the literal steering intervention

\[
h' = h + \alpha v.
\]

This is deliberately close to the task's recommended Persona Vectors evaluation philosophy while staying small, local, and fast.

## Active baseline

- LM: GPT-2 Small (`gpt2`)
- Intervention point: `blocks.6.hook_resid_post`
- Direction: mean activation difference from 30 matched positive/negative sentences
- Direction extraction: mean of the final 4 token activations for each sentence
- Steering positions: response/current token only during autoregressive generation
- Strength mode: literal `h + alpha * v`
- Concept judge: local `distilbert-base-uncased-finetuned-sst-2-english`
- Concept score: mean probability of **positive sentiment**, 0-100
- Fluency diagnostics: clean GPT-2 NLL, distinct-1/2/3, 3-gram repetition
- Pareto x-axis: a baseline-anchored composite of clean-NLL, distinct-3, and anti-repetition so degenerate low-diversity generations cannot look artificially fluent merely because their NLL decreases

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

A Hugging Face token is optional but recommended in Colab to avoid anonymous download rate limits.

## 1. Validate the direction first

Do **not** run the full sweep until this passes:

```bash
python scripts/validate_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

The validator:

1. extracts the positive-minus-negative direction at layer 6;
2. checks that positive examples project higher than negative examples;
3. uses held-out calibration prompts to choose the causal sign if necessary;
4. sweeps alpha on those calibration prompts;
5. requires a measurable increase in independent text-level sentiment score;
6. saves the calibrated direction to `results/sentiment_direction.pt`.

## 2. Run the full baseline

```bash
python scripts/run_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml

python scripts/plot_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

Expected qualitative result: as alpha grows, positive-sentiment score should rise; at sufficiently strong intervention, fluency should degrade. This gives the concept/fluency Pareto curve required before any denoiser or repair method is evaluated.

Outputs:

```text
results/sentiment_validation.json
results/sentiment_direction.pt
results/sentiment_baseline_samples.csv
results/sentiment_baseline_pareto.png
```

## Why the previous SAE baseline is archived

The failed OpenAI-SAE profanity run produced concept score `0.0` at every tested alpha while diversity eventually collapsed. This demonstrates that an interpretable SAE feature is not automatically a useful causal steering vector. The full aggregate table and exact configuration are preserved under `experiments/failed_sae_profanity/` rather than discarded.

## Repository layout

```text
configs/
  baseline_sentiment_gpt2.yaml    active reproduction baseline
  baseline_gpt2.yaml              legacy SAE experiment config

data/
  sentiment_positive.txt          direction construction data
  sentiment_negative.txt
  calibration_prompts.txt         held-out vector calibration prompts
  prompts.txt                     full evaluation prompts

experiments/
  failed_sae_profanity/           archived negative reproduction result

scripts/
  validate_sentiment_baseline.py
  run_sentiment_baseline.py
  plot_sentiment_baseline.py
  validate_baseline.py             legacy SAE validator
  run_baseline.py                  legacy SAE runner
  plot_baseline.py                 legacy SAE plotter

src/steering_repair/
  steering.py                     activation hooks / repair interface
  generation.py                   cached autoregressive generation
  sentiment_baseline.py           contrastive-vector baseline pipeline
  sae.py                           OpenAI SAE compatibility reader
  metrics.py                      text and likelihood metrics

tests/
```

## Reproducibility rules for later repair experiments

- Freeze the active baseline prompts, seeds, direction-construction data, and judge before comparing repair methods.
- Train future denoisers without using the held-out validation steering directions.
- Compare any repair method against a smaller steering coefficient; otherwise a denoiser can appear successful simply by cancelling the intervention.
- Keep text-level concept metrics separate from mechanistic activation diagnostics.
- Do not move to repair experiments until the additive baseline itself visibly reproduces the required trade-off.

## References

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- OpenAI sparse autoencoder: https://github.com/openai/sparse_autoencoder
- SAELens: https://github.com/decoderesearch/SAELens
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Generative Latent Prior: https://generative-latent-prior.github.io/
