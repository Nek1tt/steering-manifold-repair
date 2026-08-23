# Steering Manifold Repair

A compact research baseline for studying the fluency/concept trade-off of activation steering in GPT-2 Small, designed to grow into steering-aware manifold repair experiments.

## Research question

Naive activation steering adds a concept direction to a hidden state,

$$
\tilde h = h + \alpha v,
$$

which can increase the desired feature while moving the activation away from the distribution seen during language-model training. This repository first reproduces that trade-off with a fully local evaluation pipeline, then provides a clean interface for adding repair methods such as denoisers and direction-preserving corrections.

The baseline intentionally separates:

1. **intervention** — how the steering vector is added;
2. **repair** — optional post-processing of the steered activation;
3. **evaluation** — text fluency and concept strength;
4. **analysis** — Pareto curves and activation-level diagnostics.

## Baseline configuration

The default experiment uses:

- model: `gpt2-small` via TransformerLens;
- intervention location: `blocks.8.hook_resid_post`;
- SAE: OpenAI GPT-2 Small `v5_128k`, `resid_post_mlp`, layer 8;
- default feature: **56907**, described in OpenAI's SAE viewer source as "words in quotes";
- steering applied only to the last token state used to predict the next generated token;
- steering strength expressed as a ratio of the residual-stream norm, which makes the sweep easier to interpret than raw decoder-vector coefficients.

Layer 8 is used for the first reproducible baseline because OpenAI publicly lists human-interpretable features for that checkpoint. The code is configurable and can be switched to layer 6 for the assignment's midpoint-layer setting once a layer-6 validation feature set is fixed.

## Metrics

Each generated continuation is evaluated with the **clean, unsteered GPT-2** and the SAE:

- `nll`: mean negative log-likelihood of the generated continuation under clean GPT-2;
- `ppl`: perplexity from that NLL;
- `distinct_1/2/3`: lexical diversity;
- `repetition_3gram`: repeated 3-gram rate;
- `concept_sae_mean/max/firing_rate`: target SAE feature activation on continuation tokens;
- `quoted_span_rate`: a simple text-level corroborating metric for the default "words in quotes" feature.

The primary Pareto plot uses `-NLL` on x (higher is more fluent) and SAE feature activation on y (higher is more concept).

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
pip install -e .
```

The OpenAI SAE **weights** are downloaded from the official public blob storage at runtime. We intentionally do not install the historical `openai/sparse_autoencoder` package because its metadata pins `torch==2.1.0`, `transformer_lens==1.9.1`, and `blobfile==2.0.2`, which conflicts with modern Colab environments. A small inference-only compatibility reader lives in `src/steering_repair/sae.py`.

## Run the baseline

```bash
python scripts/run_baseline.py --config configs/baseline_gpt2.yaml
python scripts/plot_baseline.py \
  --input results/baseline_samples.csv \
  --output results/baseline_pareto.png
```

The generation script saves one row per `(prompt, seed, strength)` and shows progress with `tqdm`.

For a quick smoke run, reduce `max_new_tokens`, prompts, seeds, and strength grid in the YAML file.

## Expected experiment ladder

The baseline is deliberately structured for the research direction rather than only reproducing the assignment's suggested denoiser:

| Stage | Method | Purpose |
|---|---|---|
| B0 | no steering | reference |
| B1 | additive steering | required Pareto baseline |
| B2 | simple norm-preserving repair | cheap non-learned control |
| B3 | Gaussian denoiser | assignment-proposed learned baseline |
| M1 | structured-corruption denoiser | train on generic semantic perturbations, not validation vectors |
| M2 | direction-preserving repair | remove denoiser correction parallel to the intended steering direction |
| M3 | structured + direction-preserving | main candidate method |

A future direction-preserving repair can be written as

$$
\Delta = D(h + \alpha v) - (h + \alpha v),
$$

$$
\Delta_\perp = \Delta - \frac{\langle \Delta, v\rangle}{\|v\|^2}v,
$$

$$
h_{\text{repair}} = h + \alpha v + \beta\Delta_\perp.
$$

This directly tests whether a denoiser improves fluency by repairing off-direction/off-manifold components rather than merely cancelling the steering vector.

## Repository layout

```text
configs/                   experiment configuration
data/prompts.txt           fixed neutral continuation prompts
notebooks/                 Colab entry point
scripts/run_baseline.py    generation + evaluation
scripts/plot_baseline.py   aggregate Pareto plot
src/steering_repair/
  config.py                YAML dataclasses
  sae.py                   OpenAI SAE loading and decoder directions
  steering.py              intervention and repair hooks
  generation.py            deterministic autoregressive generation
  metrics.py               fluency + concept metrics
  experiment.py            end-to-end baseline loop
  plotting.py              plotting helpers
tests/                     local unit tests without model downloads
```

## Reproducibility rules

- Fix prompts and seeds before comparing repair methods.
- Validation/test SAE feature IDs must be split before denoiser training.
- A learned repair model must not train on the held-out steering vectors used for evaluation.
- Report both text-level metrics and internal activation diagnostics.
- Compare learned repair against simply using a smaller steering strength; otherwise a denoiser can appear successful by silently cancelling the intervention.

## References

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- OpenAI sparse autoencoder: https://github.com/openai/sparse_autoencoder
- SAELens: https://github.com/decoderesearch/SAELens
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Generative Latent Prior: https://generative-latent-prior.github.io/
