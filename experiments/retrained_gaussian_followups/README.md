# Fresh retrain Gaussian denoiser + dense DPAR follow-up

Дата: 2026-08-23.

Это финальная practical-проверка DPAR: Gaussian denoiser воспроизводится с нуля в fresh runtime, затем отдельно калибруется масштаб correction `beta` и проводится плотный held-out sweep.

## Протокол

- GPT-2 Small, `blocks.6.hook_resid_post`;
- 80k WikiText-2 activations, 5 эпох, seed `2026`;
- calibration: 8 prompts, seed `37`;
- held-out: 20 prompts × seeds `11/23`;
- dense `alpha = {0, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3}`;
- DPAR `beta = {.10,.25,.50,.75,1.00}`;
- vanilla `beta = {.25,.50,.75,1.00}`.

## Воспроизводимость denoiser

Fresh retrain **точно воспроизвёл** исходную training history на всех пяти эпохах.

- final validation relative MSE improvement: **67.8%**;
- `val_denoised_mse = 2.822651`;
- `d_model=768`;
- `hidden_dim=1536`;
- 5,316,096 parameters.

Этот checkpoint опубликован на Hugging Face:

**https://huggingface.co/Nek1tt/steering-repair-gpt2**

## Calibration масштаба correction

Calibration objective выбрал:

- DPAR: `beta=0.25`, score `69.56`;
- vanilla: `beta=0.25`, score `72.61`.

У DPAR `beta=.25` лишь немного лучше `.50` на calibration (`69.56` vs `67.85`), поэтому exact optimum нельзя считать устойчивым.

## Dense held-out frontier

| метод | F@C80 | F@C85 | F@C90 | F@C95 |
|---|---:|---:|---:|---:|
| additive | **94.73** | 70.79 | 66.46 | — |
| DPAR `beta=.25` | 89.53 | 79.16 | 55.16 | **48.84** |
| DPAR `beta=1` | 89.47 | 75.03 | **71.45** | — |
| vanilla `beta=.25` | 89.11 | **81.06** | 47.65 | — |
| vanilla `beta=1` | 80.90 | 75.74 | 52.60 | — |

Относительно additive:

- DPAR `.25`: `-5.19 / +8.38 / -11.30` на C80/C85/C90;
- DPAR `1.0`: `-5.26 / +4.24 / +4.99`;
- vanilla `.25`: `-5.62 / +10.28 / -18.81`.

## Интерпретация

### Correction magnitude действительно важен

Небольшой `beta` создаёт хороший mid-concept region, особенно около C85. Но calibration-selected `.25` проигрывает на C80 и C90. Единого масштаба, который доминирует additive на всём frontier, нет.

### DPAR остаётся механистически полезным

DPAR сохраняет effective `alpha` по построению, тогда как vanilla correction продолжает вычитать часть steering direction. Однако на C85 vanilla `.25` даже лучше DPAR `.25`, поэтому downstream advantage нельзя приписывать одной только protected geometry.

### High-concept эффект переживает dense interpolation

Full DPAR `beta=1` даёт

\[
F@C90=71.45
\]

против

\[
F@C90=66.46
\]

у additive: **+4.99 fluency points**.

Это существенно меньше coarse-grid `+7.69`, но эффект не исчезает после densification/interpolation.

На обоих frozen seeds направление C90-эффекта совпадает:

| seed | additive F@C90 | DPAR `beta=1` F@C90 | delta |
|---:|---:|---:|---:|
| 11 | 61.23 | 71.76 | +10.53 |
| 23 | 69.04 | 89.75 | +20.71 |

Из-за шумного и немонотонного sentiment judge это **descriptive local evidence**, а не статистическое доказательство universal domination.

## Финальный practical вывод

> Vanilla denoising имеет реальный steering-cancellation failure mode, который DPAR устраняет точно. Dense held-out evaluation подтверждает локальные Pareto gains, включая `+4.99` fluency points на C90 для full DPAR, но ни один calibration-selected `beta` не доминирует additive во всех concept regions.

Сохранённые evidence-файлы:

- `calibration_beta_scores.csv`;
- `frontier_summary.csv`;
- `heldout_per_seed_frontier.csv`;
- `retrained_denoiser_gaussian_history.json`.
