# Repair experiment suite

This stage starts only after the midpoint contrastive sentiment baseline passed.
The baseline is frozen; these experiments modify only the post-steering repair.

## Hypotheses

**H1 — Gaussian denoising.** A small residual MLP trained to reconstruct natural
layer-6 activations after isotropic corruption should recover some fluency after
`h + alpha*v`.

**H2 — Direction-Preserving Activation Repair (DPAR).** A vanilla denoiser may
improve fluency partly by subtracting the intended steering direction. Let
`delta = D(z)-z` for `z=h+alpha*v`. DPAR applies only

`delta_perp = delta - proj_v(delta)`

so the requested steering component is preserved by construction.

**H3 — Structured corruption.** Steering is directional, not isotropic. A second
denoiser is trained on a 50/50 mixture of Gaussian corruptions and normalized
natural activation-difference directions. The held-out sentiment direction is
never used during training.

## Controls / ablations

- additive steering
- norm-preserving rescaling
- Gaussian denoiser
- Gaussian denoiser with `lambda=0.5` parallel correction retention
- Gaussian DPAR (`lambda=0`)
- mixed structured denoiser
- mixed structured DPAR

## Mechanistic diagnostics

Every learned repair logs requested/effective alpha, cosine of the correction
with the steering vector, parallel correction fraction, and correction norm
relative to the steering perturbation. This tests *why* a method changes the
Pareto frontier instead of reporting text metrics only.
