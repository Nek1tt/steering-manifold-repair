# Steering Manifold Repair

Mechanistic-interpretability project on **coherence-preserving activation steering** in GPT-2 Small.

The project begins with the standard failure mode: stronger activation steering increases the target concept but eventually damages language-model coherence. We reproduce that trade-off, evaluate learned activation denoisers, and then follow the failures of those methods into two new mechanistic hypotheses about why strong steering breaks downstream computation.

## Final results

The complete final report is in [`FINAL_RESULTS.md`](FINAL_RESULTS.md).

The main research contributions are:

1. **Direction-Preserving Activation Repair (DPAR).** Vanilla activation denoising increasingly cancels the requested steering direction. DPAR removes that parallel correction exactly, preserving requested alpha to numerical precision. Full DPAR retains a local held-out high-concept advantage (`F@C90=71.45` vs additive `66.46`).
2. **Jacobian Residual Repair (JRR).** Strong steering creates a downstream nonlinear Taylor remainder whose norm grows approximately as `alpha^1.9849`, almost exactly the second-order prediction. In the strong regime the nonlinear response becomes comparable in norm to the entire first-order transported steering effect. Causal removal recovers fluency but reveals that some orthogonal nonlinear computation also carries useful concept information.
3. **KL-Selective JRR.** A follow-up preregistered after JRR isolates a compact local KL-sensitive mode inside the nonlinear remainder. Across strong calibration strengths it removes only `7.93%` of `R_orth` on average while lowering local KL by `41.6%`, but it fails the frozen long-horizon concept/fluency gate. The newly frozen held-out was therefore intentionally not opened.

The strongest combined claim is mechanistic rather than universal-SOTA:

> Coherence loss under strong steering arises from at least two separable failures: learned denoisers can cancel the requested steering axis, while the model's own downstream nonlinear response becomes large enough to rival the first-order steering effect. That nonlinear response contains both fluency-damaging and concept-carrying computation, so preserving only the original steering direction—or even its locally transported image—is not sufficient.

## Experiment archive

| experiment | role | result |
|---|---|---|
| [`failed_sae_profanity/`](experiments/failed_sae_profanity/) | vector validation | negative: concept score did not move |
| [`successful_sentiment_baseline/`](experiments/successful_sentiment_baseline/) | frozen additive baseline | positive trade-off reproduced |
| [`repair_suite/`](experiments/repair_suite/) | denoiser + DPAR + structured corruption | DPAR mechanism validated; structured corruption negative |
| [`retrained_gaussian_followups/`](experiments/retrained_gaussian_followups/) | fresh retrain + dense held-out | deterministic retrain; local DPAR gains |
| [`jacobian_residual_repair/`](experiments/jacobian_residual_repair/) | new nonlinear-propagation hypothesis | strong mechanistic positive, mixed practical oracle |
| [`selective_jrr/`](experiments/selective_jrr/) | harmful-mode-selective follow-up | partial/negative; fresh held-out kept untouched |

Negative results and failed hypotheses are retained deliberately.

## Core setup

The successful baseline uses a persona-style sentiment direction

\[
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}]
\]

at

```text
blocks.6.hook_resid_post
```

with literal response-token steering

\[
h' = h + \alpha v.
\]

The concept score rises from roughly 28 to above 90 while sufficiently strong steering collapses fluency.

## DPAR

For steered activation

\[
z=h+\alpha v
\]

and denoiser correction

\[
\Delta=D(z)-z,
\]

DPAR applies only

\[
\Delta_\perp=\Delta-\operatorname{proj}_v(\Delta),
\]

so

\[
h_{\mathrm{DPAR}}=z+\Delta_\perp.
\]

This prevents the repair method from obtaining apparent fluency improvements merely by undoing steering.

## JRR

For downstream map `F`, JRR measures

\[
R_\alpha = F(h+\alpha v)-F(h)-\alpha J_F(h)v.
\]

The observed near-quadratic growth and causal interventions are documented in [`experiments/jacobian_residual_repair/RESULTS.md`](experiments/jacobian_residual_repair/RESULTS.md).

## Installation

### Linux / Colab

```bash
pip install -r requirements.txt
pip install -e .
```

### Windows + NVIDIA GPU + VS Code Jupyter

See [`LOCAL_WINDOWS_VSCODE.md`](LOCAL_WINDOWS_VSCODE.md).

Recommended local notebook for the latest experiment:

```text
notebooks/selective_jrr_vscode_windows.ipynb
```

## Reproduction entry points

Additive baseline:

```text
notebooks/baseline_colab.ipynb
```

Denoiser / DPAR suite:

```text
notebooks/repair_experiments_colab.ipynb
```

Fresh deterministic Gaussian retrain + dense DPAR follow-up:

```text
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
```

JRR:

```text
notebooks/jrr_experiment_colab.ipynb
```

KL-Selective JRR on Windows/VS Code:

```text
notebooks/selective_jrr_vscode_windows.ipynb
```

## Tests

```bash
pytest -q
```

Focused final-method tests:

```bash
pytest -q tests/test_denoiser.py tests/test_inference_followups.py tests/test_jrr.py tests/test_selective_jrr.py
```

## Hugging Face checkpoint

The best practical learned component is the deterministically reproduced Gaussian activation denoiser used with DPAR.

**Public checkpoint:** [Nek1tt/steering-repair-gpt2](https://huggingface.co/Nek1tt/steering-repair-gpt2)

The Hugging Face repository contains the trained checkpoint together with its model card, metadata, frozen training configuration, and training history. DPAR itself is inference-time geometry applied to the denoiser correction and is not encoded in the checkpoint weights.

Packaging files are retained in this repository:

```text
huggingface/MODEL_CARD.md
scripts/publish_best_checkpoint_hf.py
```

## Reproducibility rules

- Frozen baseline prompts/seeds are not retuned for repair methods.
- Evaluation steering vectors are not used to train denoisers.
- Pareto comparisons use dense/interpolated frontiers where appropriate.
- Repair geometry is inspected to rule out steering cancellation.
- New methods motivated by held-out results receive a newly frozen held-out split.
- Failed gates are not relaxed post hoc.
- Negative results and ablations remain in the repository.

## References

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- OpenAI sparse autoencoder: https://github.com/openai/sparse_autoencoder
- SAELens: https://github.com/decoderesearch/SAELens
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Generative Latent Prior: https://generative-latent-prior.github.io/
