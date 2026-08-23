# KL-Selective JRR — deadline runbook

This runbook is optimized for the final few hours before submission.

## Preferred entry point

Use:

```text
notebooks/selective_jrr_experiment_colab.ipynb
```

Run cells top to bottom. The notebook will not open the fresh held-out if calibration fails.

## CLI equivalent

### 1. Update and install

```bash
git pull --ff-only
pip install -r requirements.txt
pip install -e .
```

If `results/sentiment_direction.pt` does not exist:

```bash
python scripts/validate_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

Do not rebuild the direction if it already exists.

### 2. Fast tests

```bash
pytest -q tests/test_selective_jrr.py tests/test_jrr.py
```

Expected: all tests pass.

### 3. Real-model numerical preflight

```bash
python scripts/preflight_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml
```

Required final line:

```text
KL-SELECTIVE JRR PREFLIGHT: PASS
```

If this fails, stop and send the full output. Do not run calibration on a numerically invalid gradient.

### 4. Calibration

```bash
python scripts/run_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml \
  --phase calibration
```

Inspect:

```text
results/selective_jrr/calibration_summary.json
results/selective_jrr/calibration_same_alpha.csv
results/selective_jrr/calibration_aggregate.csv
results/selective_jrr/calibration_pareto.png
```

The precommitted gate is:

```text
at alpha in {2.25, 3, 4}:
    delta fluency vs additive >= +5
    delta concept vs additive >= -5
```

If `go_to_new_heldout` is false, stop. That is a valid negative result.

### 5. Fresh held-out only after calibration passes

```bash
python scripts/run_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml \
  --phase evaluation
```

This evaluates only:

```text
alpha = 0, 2.25, 3, 4
methods = additive, full JRR, KL-JRR
prompts = 12 new prompts
seeds = 101, 211
```

Do not change these values after seeing results.

### 6. Package immediately

```python
import shutil
shutil.make_archive(
    "/content/selective_jrr_results",
    "zip",
    "results/selective_jrr",
)
```

Send `selective_jrr_results.zip` back for final analysis and GitHub result archival.

## What counts as success

The strongest result is not merely a higher concept/fluency point. We want all three:

1. KL-JRR removes substantially less than 100% of `R_orth` (`sel_selected_fraction`);
2. local `sel_kl_after < sel_kl_before` in the strong-steering regime;
3. relative to full JRR, KL-JRR preserves more concept while retaining a meaningful fraction of the fluency/NLL repair.

That combination would directly support the harmful-vs-useful nonlinear-mode decomposition.

## Time-saving rule

Do **not** run another layer sweep, beta sweep, or original 20-prompt held-out. The target layer and beta are already frozen, and the original held-out influenced this method's design.
