# JRR execution runbook

Use `notebooks/jrr_experiment_colab.ipynb` for the safest end-to-end run. The notebook contains the same gates as the CLI workflow below.

## 1. Fresh environment

```bash
pip install -r requirements.txt
pip install -e .
```

Restore the already-frozen sentiment steering direction only when it is absent:

```bash
python scripts/validate_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

Do not change the baseline prompts, judge, vector construction, held-out prompts, or evaluation seeds.

## 2. Mathematical regression tests

```bash
pytest -q \
  tests/test_jrr.py \
  tests/test_inference_followups.py \
  tests/test_denoiser.py
```

`tests/test_jrr.py` checks the decomposition, protected transported component, full-linearization ablation, exact quadratic remainder, and agreement between autograd JVP and central finite differences on a synthetic system.

## 3. Real-model numerical preflight

```bash
python scripts/preflight_jrr.py --config configs/jrr_gpt2.yaml
```

This is a numerical safety check, not experimental evidence. It loads the real GPT-2 model and frozen sentiment direction, verifies that every target hook is downstream of the source hook, and compares the primary JVP with an independent central finite difference on a real calibration prompt.

Expected final line:

```text
JRR PREFLIGHT: PASS
```

Do not run the mechanistic diagnostic if this check fails. A JVP disagreement would make the Taylor remainder uninterpretable.

## 4. Stage A — cheap mechanistic diagnostic

```bash
python scripts/run_jrr_diagnostic.py --config configs/jrr_gpt2.yaml
```

Read:

```text
results/jrr/DIAGNOSTIC.md
results/jrr/diagnostic_summary.json
results/jrr/target_layer_summary.csv
results/jrr/remainder_scaling.png
results/jrr/orthogonal_remainder_fraction.png
results/jrr/orthogonal_remainder_vs_fluency.png
```

Stage A uses only `data/calibration_prompts.txt`. It scans downstream layers 7–11 and asks whether the nonlinear remainder grows strongly and whether its orthogonal component tracks additive fluency degradation.

If `diagnostic_summary.json` says:

```json
{"oracle_recommended": false}
```

stop. Package `results/jrr/` and treat JRR as a falsified/negative hypothesis. Do not inspect held-out data and do not bypass the gate to obtain a prettier graph.

## 5. Stage B — causal oracle calibration

Only when `oracle_recommended` is true:

```bash
python scripts/run_jrr_oracle.py \
  --config configs/jrr_gpt2.yaml \
  --phase calibration
```

Read:

```text
results/jrr/ORACLE_CALIBRATION.md
results/jrr/oracle_calibration_summary.json
results/jrr/oracle_calibration_frontier.csv
results/jrr/oracle_calibration_pareto.png
```

The oracle recomputes the local downstream Taylor remainder at every generated token and removes the component orthogonal to the transported first-order direction. This stage is deliberately slow because it is a causal feasibility test, not the final efficient method.

If `go_to_heldout` is false, stop JRR and report the mechanistic result without training an adapter.

## 6. Frozen held-out evaluation

Only when oracle calibration passes:

```bash
python scripts/run_jrr_oracle.py \
  --config configs/jrr_gpt2.yaml \
  --phase evaluation
```

This uses the frozen held-out prompts and seeds `11,23`. After this command, do not tune JRR hyperparameters or target layer.

## 7. Package results

In Colab:

```python
import shutil
shutil.make_archive('/content/jrr_results', 'zip', 'results/jrr')
```

Send `jrr_results.zip` back for analysis.

## Decision after the run

- **Diagnostic negative:** stop JRR. The hypothesis failed cheaply.
- **Diagnostic positive, oracle negative:** nonlinear propagation is a marker of degradation but not its causal bottleneck. This is still a useful mechanistic negative result.
- **Oracle positive:** implement the next experiment — an amortized residual adapter trained to predict the useful oracle correction across multiple training steering directions. Only that efficient learned adapter should become a candidate Hugging Face checkpoint.
