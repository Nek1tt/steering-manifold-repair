---
base_model: gpt2
library_name: pytorch
tags:
- mechanistic-interpretability
- activation-steering
- activation-denoising
- gpt2
---

# GPT-2 Gaussian Activation Denoiser for Direction-Preserving Activation Repair

This repository contains the final Gaussian activation-denoiser checkpoint from the `steering-manifold-repair` project.

The model is a residual MLP trained on generic GPT-2 Small activations from:

```text
blocks.6.hook_resid_post
```

It reconstructs clean residual-stream activations from Gaussian-corrupted activations and is intended to be used with **Direction-Preserving Activation Repair (DPAR)** at inference time.

## DPAR inference geometry

For steered state

```text
z = h + alpha * v
```

let the denoiser propose

```text
raw = D(z) - z
```

DPAR removes the component of that correction parallel to the steering direction:

```text
correction = raw - proj_v(raw)
output = z + correction
```

This preserves the requested steering component by construction while allowing an orthogonal denoising correction.

The DPAR projection is an inference-time operation and is **not encoded directly in the checkpoint weights**.

## Training

- base model: GPT-2 Small
- residual hook: `blocks.6.hook_resid_post`
- training corpus: WikiText-2 train stream
- cached activations: 80,000 total (72k train / 8k validation)
- d_model: 768
- hidden dimension: 1536
- corruption: isotropic Gaussian activation noise over a range of noise/severity ratios
- optimizer/training config: included as `training_config.yaml`
- exact training history: included as `training_history.json`

A fresh retrain reproduced the archived training history exactly under the frozen seeds/configuration. Final held-out relative activation-MSE improvement was approximately **67.8%**.

## Evaluation summary

The main mechanistic finding is that a vanilla denoiser correction increasingly points against the steering direction as steering strength grows. DPAR removes this steering-cancellation failure exactly and preserves effective alpha to numerical precision.

Downstream concept/fluency improvements are local rather than universal. In the dense held-out follow-up, full DPAR (`beta=1`) achieved:

```text
F@C90 = 71.45
```

versus additive steering:

```text
F@C90 = 66.46
```

for a descriptive gain of about `+4.99` fluency points. The sentiment score is noisy/non-monotonic, so this should not be interpreted as universal Pareto domination.

## Related mechanistic results

The GitHub repository also contains two later non-checkpoint oracle investigations:

- Jacobian Residual Repair (JRR): strong steering creates an approximately quadratic downstream nonlinear Taylor remainder whose magnitude becomes comparable to the first-order steering effect.
- KL-Selective JRR: a compact local KL-sensitive mode explains a large fraction of local next-token divergence, but does not robustly preserve long-horizon concept in the preregistered strong-steering calibration.

These experiments are included for mechanistic analysis; the checkpoint in this repository remains the best validated practical learned component.

## Code and reproduction

Full code, experiment reports, unit tests, notebooks, and archived compact results are available in the GitHub repository:

```text
https://github.com/Nek1tt/steering-manifold-repair
```

Primary reproduction notebook:

```text
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
```

## Limitations

- Results are on GPT-2 Small and a sentiment/persona steering direction.
- Concept evaluation is noisy and non-monotonic with steering strength.
- DPAR has a strong mechanistic guarantee (preserving the requested steering-axis component), but downstream text gains are threshold-dependent rather than universal.
- The checkpoint is not evidence that Gaussian denoising is optimal for every steering direction or model family.
