# Steering Manifold Repair — report draft

## 1. Problem

Activation steering modifies a hidden state with

\[
\tilde h = h + \alpha v.
\]

Strong interventions can increase the desired concept while moving hidden states away from the model's natural activation distribution and degrading generation quality. The goal is to improve this concept/fluency Pareto frontier with a cheap repair step.

## 2. Baseline reproduction

Model: GPT-2 Small. Intervention point: `blocks.6.hook_resid_post`.

A persona-style contrastive sentiment direction is constructed from matched positive and negative examples:

\[
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}].
\]

The direction is applied only to the current response token state during autoregressive generation. Concept score is the probability of positive sentiment from an independent local SST-2 classifier. Fluency combines clean-GPT-2 NLL, distinct-3 and anti-repetition, anchored to the unsteered generation.

The baseline reproduces the required trade-off. Representative points:

| alpha | fluency | concept |
|---:|---:|---:|
| 0.00 | 100.0 | 27.9 |
| 0.50 | 100.0 | 66.2 |
| 0.75 | 99.6 | 76.6 |
| 1.00 | 88.7 | 84.2 |
| 2.00 | 63.4 | 93.6 |
| 4.00 | 18.0 | 95.3 |

The earlier OpenAI-SAE profanity-vector attempt is preserved as a failed vector-validation experiment rather than discarded.

## 3. Assignment-proposed learned baseline: Gaussian denoising

A residual MLP is trained on generic natural layer-6 activations, never on the held-out sentiment direction. For a relative corruption ratio

\[
r = \|\delta\|/\|h\|,
\]

we sample isotropic perturbations and train

\[
D_\theta(h+\delta,r) \approx h
\]

with MSE. The denoiser is explicitly conditioned on `r` and begins as the identity map.

## 4. Proposed method: Direction-Preserving Activation Repair (DPAR)

A vanilla denoiser can improve fluency simply by cancelling part of the intended steering direction. For

\[
z = h + \alpha v,
\qquad
\Delta = D(z)-z,
\]

we decompose the correction into parallel and orthogonal components:

\[
\Delta_\parallel = \operatorname{proj}_v(\Delta),
\qquad
\Delta_\perp = \Delta - \Delta_\parallel.
\]

DPAR applies

\[
h_{\text{DPAR}} = z + \Delta_\perp.
\]

A partial-preservation ablation uses

\[
h' = z + \Delta_\perp + \lambda\Delta_\parallel,
\]

where `lambda=1` is vanilla denoising and `lambda=0` is DPAR.

## 5. Proposed training change: structured corruption

Isotropic Gaussian noise may be a poor model of steering, which is a directional perturbation in representation space. A second denoiser is trained on a 50/50 mixture of Gaussian perturbations and normalized natural activation-difference directions

\[
u = \frac{h_j-h_k}{\|h_j-h_k\|},
\qquad
\tilde h = h_i + s u.
\]

The evaluation sentiment direction is never used to construct these perturbations.

## 6. Methods compared

- additive steering
- norm-preserving non-learned repair
- Gaussian denoiser
- Gaussian denoiser with `lambda=0.5`
- Gaussian DPAR
- mixed structured denoiser
- mixed structured DPAR

## 7. Mechanistic diagnostics

To distinguish true repair from weaker steering, we record:

1. effective steering amplitude after repair,
\[
\alpha_{\mathrm{eff}} = \frac{(h_{\mathrm{repair}}-h)^\top v}{\|v\|^2};
\]
2. cosine between the denoiser correction and the steering direction;
3. fraction of correction norm parallel to `v`;
4. correction norm relative to the original steering perturbation.

If vanilla denoising yields `alpha_eff < alpha` and negative correction cosine while DPAR keeps `alpha_eff ≈ alpha`, this directly demonstrates the cancellation failure mode.

## 8. Results

To be filled from `results/repair_suite/` after the Colab run.

Primary figures:

- `repair_pareto.png`
- `effective_alpha.png`
- `correction_geometry.png`
- `frontier_summary.csv`
- `denoiser_cross_reconstruction.csv`

## 9. Conclusions

To be filled after evaluating the frozen repair suite. Negative hypotheses will be retained and discussed rather than hidden.
