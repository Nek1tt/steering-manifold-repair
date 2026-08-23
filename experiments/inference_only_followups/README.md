# Inference-only follow-ups: scaled DPAR and dense alpha sweep

This experiment is designed to use the **already-trained** Gaussian and mixed
denoiser checkpoints from `experiments/repair_suite`. It must not update any
model weights.

## Motivation

The first repair suite established two facts:

1. vanilla Gaussian denoising increasingly cancels the intended steering
   direction as alpha grows;
2. DPAR fixes that cancellation almost exactly, but full-strength DPAR
   (`beta=1`) did not clearly improve the downstream concept/fluency frontier.

A plausible explanation is that the MSE-trained orthogonal correction is useful
in direction but too large in magnitude. We therefore introduce a second,
independent inference-time control:

```text
z = h + alpha v
raw = D(z) - z
filtered = raw - (1-lambda) proj_v(raw)
h_out = z + beta * filtered
```

- `lambda=1`: vanilla denoising geometry.
- `lambda=0`: DPAR geometry.
- `beta`: amount of the denoiser correction actually applied.

No retraining is required.

## Questions

### F1 — Was the previous repair simply too strong?

Sweep `beta` for Gaussian vanilla and Gaussian DPAR while keeping the denoiser
checkpoint frozen.

### F2 — Is DPAR geometry useful after correction magnitude is calibrated?

Gaussian vanilla and DPAR get **separate beta calibration**. This avoids
confounding direction preservation with correction magnitude.

### F3 — Can the structured checkpoint be rescued by a smaller DPAR correction?

A smaller beta sweep is also run for mixed-DPAR. If it remains worse after
magnitude calibration, the negative structured-corruption result is stronger.

### F4 — Was the apparent DPAR gain around concept ~= 90 a coarse-alpha artifact?

The held-out evaluation uses a dense alpha grid around the transition region and
reports a piecewise-linear interpolated frontier. This directly addresses the
ambiguity in the first repair suite.

## Protocol

1. Reuse `checkpoints/denoiser_gaussian.pt` and
   `checkpoints/denoiser_mixed.pt`.
2. Select beta **only on `data/calibration_prompts.txt`**.
3. Freeze the selected beta values.
4. Evaluate on `data/prompts.txt` with the original seeds `11, 23` and a dense
   alpha grid.
5. Keep the original sentiment judge and layer-6 steering vector unchanged.

## Files produced by the run

`results/inference_followups/` should contain:

- `calibration_samples.csv`
- `calibration_aggregate.csv`
- `calibration_beta_scores.csv`
- `selection.json`
- `heldout_samples.csv`
- `heldout_aggregate.csv`
- `heldout_interpolated_frontier.csv`
- `beta_calibration.png`
- `selected_dense_pareto.png`
- `selected_effective_alpha.png`
- `SUMMARY.md`

The scientific result should be interpreted only after the held-out phase. A
beta that looks good only on calibration prompts is not evidence of improvement.
