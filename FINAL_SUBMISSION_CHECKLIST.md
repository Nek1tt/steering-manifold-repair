# Final submission checklist

Use this checklist for the final hours before submission. Do not open new research branches unless a critical bug is discovered.

## 1. Freeze research claims

- [x] DPAR practical result archived.
- [x] JRR mechanistic result archived.
- [x] KL-Selective JRR calibration archived with `go_to_new_heldout=false`.
- [x] Fresh Experiment-008 held-out remains untouched.
- [x] `FINAL_RESULTS.md` updated with narrow/defensible claims.

## 2. Publish the required Hugging Face checkpoint

The best practical learned checkpoint is the deterministically reproduced Gaussian activation denoiser used with DPAR.

Find the checkpoint file locally. Expected filename:

```text
retrained_denoiser_gaussian.pt
```

If necessary search from repository root in PowerShell:

```powershell
Get-ChildItem -Path . -Recurse -Filter retrained_denoiser_gaussian.pt
```

Authenticate to Hugging Face:

```powershell
.\.venv\Scripts\python.exe -m pip install -U huggingface_hub
.\.venv\Scripts\hf.exe auth login
```

Then publish, replacing the checkpoint path and repo id:

```powershell
.\.venv\Scripts\python.exe scripts\publish_best_checkpoint_hf.py `
  --checkpoint "C:\path\to\retrained_denoiser_gaussian.pt" `
  --repo-id "Nek1tt/steering-repair-gpt2"
```

Do **not** pass `--private`: the assignment asks for an open repository.

After upload:

- [ ] Open the Hugging Face URL in an incognito/private browser window.
- [ ] Confirm `retrained_denoiser_gaussian.pt`, `README.md`, `checkpoint_metadata.json`, `training_config.yaml`, and `training_history.json` are visible.
- [ ] Add the public Hugging Face URL to root `README.md` and `FINAL_RESULTS.md`.

## 3. Repository hygiene

- [ ] Root `README.md` gives a short project overview and links `FINAL_RESULTS.md`.
- [ ] `FINAL_RESULTS.md` clearly separates practical result from mechanistic oracle experiments.
- [ ] No claims of universal domination.
- [ ] Failed/negative experiments remain visible rather than deleted.
- [ ] Fresh Experiment-008 held-out remains unopened because calibration gate failed.
- [ ] `git status` is clean locally after final push.

## 4. Smoke tests

From the local Windows environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

If the full suite is too slow in the final minutes, at minimum:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_denoiser.py tests\test_inference_followups.py tests\test_jrr.py tests\test_selective_jrr.py
```

## 5. Final report structure

Recommended narrative:

1. reproduce steering degradation;
2. Gaussian denoiser baseline;
3. diagnose steering cancellation;
4. DPAR: exact geometric fix + local held-out gains;
5. JRR: discover approximately quadratic downstream nonlinear response;
6. causal JRR: nonlinear response contains both harmful and concept-carrying computation;
7. KL-Selective JRR: compact KL-sensitive mode exists but local sensitivity does not guarantee sequence-level concept preservation;
8. limitations and future work.

This structure emphasizes the assignment criterion that a strong method is only half of the work: the other half is explaining why it works and where simple mechanisms fail.

## 6. Do not spend final time on

- new beta/layer sweeps;
- relaxing preregistered Experiment-008 gates;
- forcing the untouched held-out after a failed gate;
- another model family;
- large architecture changes;
- cosmetic plots before the HF checkpoint and report links are complete.
