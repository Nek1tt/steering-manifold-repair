# Failed baseline reproduction: OpenAI SAE profanity feature

Status: **failed as the required steering baseline**.

This experiment is intentionally preserved because it is informative: the intervention clearly perturbed GPT-2, but it did **not** measurably increase the intended generated-text concept. Therefore it must not be used as the control Pareto frontier for repair experiments.

## Setup

- Model: GPT-2 Small (`gpt2` / 12 layers)
- Intervention: response-token steering at `blocks.8.hook_resid_post`
- SAE: OpenAI GPT-2 Small v5 128k, `resid_post_mlp`, layer 8
- Feature: `64840`, interpreted as a predictor related to profanity
- Intervention: `h <- h + alpha * std(h) * v`
- Text concept score: percentage of completions containing a word matched by the local profanity lexicon
- Fluency diagnostics: clean-model NLL, distinct-3, 3-gram repetition

## Observed aggregate result

| alpha | NLL | concept score (%) | profanity rate | SAE mean | distinct-3 | repetition-3gram |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.895616 | 0.0 | 0.0 | 0.0 | 0.983509 | 0.016491 |
| 4 | 2.871503 | 0.0 | 0.0 | 0.0 | 0.987869 | 0.012131 |
| 8 | 2.945581 | 0.0 | 0.0 | 0.0 | 0.985419 | 0.014581 |
| 16 | 4.117668 | 0.0 | 0.0 | 0.0 | 0.997414 | 0.002586 |
| 24 | 3.990313 | 0.0 | 0.0 | 0.0 | 0.898077 | 0.001923 |
| 32 | 3.307565 | 0.0 | 0.0 | 0.0 | 0.400000 | 0.000000 |
| 48 | 3.006352 | 0.0 | 0.0 | 0.0 | 0.000000 | 0.000000 |
| 64 | 2.798054 | 0.0 | 0.0 | 0.0 | 0.000000 | 0.000000 |

The automatic baseline check returned:

```text
{'passed': False, 'concept_gain': 0.0, 'best_strength': 0.0,
 'nll_delta_at_best_concept': 0.0}
```

## Interpretation

The result is not evidence that activation steering itself failed. Increasing alpha changes model behavior strongly: NLL and diversity move substantially and the generation eventually degenerates. The missing piece is the desired causal concept effect.

A human-interpretable SAE feature is not automatically a good steering direction. This run moved from weak intervention into degeneration without producing a measurable rise in the intended text concept. That violates the assignment's required validation step: before training a repair method, naive `h + alpha v` must already yield a concept/fluency trade-off resembling the example Pareto curve.

We therefore keep this as a **negative vector-validation experiment** and switch the active reproduction baseline to a contrastive sentiment/persona-style direction at the midpoint layer.

See `configs/baseline_sentiment_gpt2.yaml` and `scripts/validate_sentiment_baseline.py`.
