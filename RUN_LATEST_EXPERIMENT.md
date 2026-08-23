# Latest experiment: KL-Selective JRR

The current deadline-priority experiment is **Experiment 008 — KL-Selective Jacobian Residual Repair**.

Motivation and protocol:

```text
experiments/selective_jrr/README.md
```

Time-boxed execution instructions:

```text
experiments/selective_jrr/RUNBOOK.md
```

Recommended notebook:

```text
notebooks/selective_jrr_experiment_colab.ipynb
```

The method was designed after analyzing Experiment 007, so it uses a newly frozen held-out split (`data/selective_jrr_heldout_prompts.txt`, seeds `101/211`) rather than reusing the old JRR held-out as confirmatory evidence.
