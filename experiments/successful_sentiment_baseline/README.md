# Successful baseline reproduction: midpoint contrastive sentiment steering

Status: **PASS**

This experiment replaces the failed SAE profanity direction with a GPT-2 Small midpoint contrastive direction:

```text
v = mean(h | positive examples) - mean(h | negative examples)
```

The intervention is applied at `blocks.6.hook_resid_post` during response-token generation:

```text
h' = h + alpha * v
```

Concept strength is evaluated independently from the steering construction using `distilbert-base-uncased-finetuned-sst-2-english` as a local positive-sentiment judge. Fluency combines clean-model NLL, distinct-3, and anti-repetition, anchored to the unsteered baseline.

## Result

The run reproduces the qualitative Pareto trade-off required by the assignment: increasing steering strength first raises concept strength with little fluency cost, then stronger steering continues to increase/saturate the concept while fluency collapses.

Key points:

| alpha | fluency | concept | NLL |
|---:|---:|---:|---:|
| 0.00 | 100.00 | 27.92 | 2.896 |
| 0.25 | 99.88 | 48.09 | 2.758 |
| 0.50 | 100.00 | 66.15 | 2.836 |
| 0.75 | 99.56 | 76.63 | 2.900 |
| 1.00 | 88.74 | 84.18 | 3.015 |
| 2.00 | 63.35 | 93.59 | 3.352 |
| 4.00 | 18.01 | 95.31 | 4.610 |
| 16.00 | 6.67 | 99.91 | 5.572 |

The strongest useful region for later method comparisons is around `alpha=0.5..2.0`: it spans high-fluency / moderate-concept through lower-fluency / high-concept operating points without relying only on degenerate extreme steering.

The automatic baseline check returned:

```text
passed: True
concept_gain: 71.9979
best_strength: 16.0
best_concept_score: 99.9145
fluency_at_best_concept: 6.6658
max_fluency_drop_with_concept: 94.4190
```

The full aggregate table is stored in `aggregate.csv`.

## Interpretation

This is the baseline that should be frozen and used as the control curve for denoising / repair experiments. The earlier SAE profanity run remains archived separately as a failed vector-validation experiment.
