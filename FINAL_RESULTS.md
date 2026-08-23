# Final results

The final focused experiment is archived in [`experiments/retrained_gaussian_followups/`](experiments/retrained_gaussian_followups/).

## Bottom line

The project establishes a clear mechanistic failure mode of vanilla activation denoising: its correction increasingly points against the requested steering direction, so part of its apparent repair comes from cancelling steering. Direction-Preserving Activation Repair (DPAR) removes that parallel correction and preserves the requested alpha to numerical precision.

A fresh Gaussian retrain reproduced the original training history exactly, ending at **67.8% held-out relative activation-MSE improvement**. A subsequent beta calibration and dense held-out alpha sweep showed that correction scaling produces **local** Pareto improvements, but no single calibration-selected beta uniformly dominates additive steering.

At the high-concept operating point (`concept >= 90`), full DPAR retains a modest dense/interpolated advantage over additive steering:

- additive: **66.46** fluency
- DPAR beta=1.0: **71.45** fluency
- descriptive gain: **+4.99**

This direction is consistent across both frozen evaluation seeds in the per-seed analysis, but should still be presented as descriptive evidence because the sentiment concept score is noisy and non-monotonic across alpha.

## Reproduction

For a new Colab runtime, use:

`notebooks/retrain_gaussian_followups_fresh_colab.ipynb`

The notebook clones the repository, installs dependencies, runs the DPAR/denoiser unit tests, performs a real-data preflight, rebuilds the sentiment direction when needed, caches generic layer-6 activations, retrains the Gaussian denoiser, calibrates beta on separate prompts, and evaluates the frozen held-out prompts/seeds.

Reference final Gaussian validation relative MSE improvement: `0.6781128741849923`.

## Archived evidence

- `experiments/retrained_gaussian_followups/README.md` — protocol, interpretation, conclusions.
- `experiments/retrained_gaussian_followups/calibration_beta_scores.csv` — calibration sweep.
- `experiments/retrained_gaussian_followups/frontier_summary.csv` — dense held-out interpolated frontier.
- `experiments/retrained_gaussian_followups/heldout_per_seed_frontier.csv` — seed sensitivity.
- `experiments/retrained_gaussian_followups/retrained_denoiser_gaussian_history.json` — exact fresh-retrain history.

The scientific claim is intentionally narrow: **DPAR is a validated geometric repair mechanism with threshold-dependent downstream gains, not a universally dominating steering-repair method.**
