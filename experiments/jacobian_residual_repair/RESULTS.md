# Experiment 007 — результаты Jacobian Residual Repair

Дата: 2026-08-23.

## Итоговый статус

- **Mechanistic hypothesis:** сильно поддержана.
- **Calibration oracle:** положительный.
- **Preregistered C80/C85/C90 held-out frontier:** не оценивается, потому что ни additive, ни JRR не достигли C80 после усреднения двух seeds в manual oracle protocol.
- **Same-alpha held-out:** JRR заметно улучшает fluency/NLL в части strong-steering regime, но сохранение concept чувствительно к seed.

Главный вывод:

> Сильный steering создаёт почти second-order downstream nonlinear remainder, большей частью ортогональный `Jv`. Удаление всего `R_orth` действительно может вернуть fluency, но этот подпространственный компонент не является чистым «мусором»: он также содержит полезную nonlinear concept adaptation.

## 1. Mechanistic diagnostic

Source hook: `blocks.6.hook_resid_post`.

Calibration выбрала target: `blocks.7.hook_resid_post`.

Для

$$
R_\alpha = F(h+\alpha v)-F(h)-\alpha J_F(h)v
$$

получено:

| диагностика | значение |
|---|---:|
| log-log slope нормы `R_alpha` vs `alpha` | **1.9849** |
| mean orthogonal fraction | **0.9404** |
| rank corr нормы `R_orth` vs NLL | +0.8909 |
| rank corr нормы `R_orth` vs fluency | -0.8909 |
| JVP | autograd |

Slope практически совпадает с `O(alpha^2)`. Correlation values сами по себе не являются causal evidence: `alpha` одновременно увеличивает и residual norm, и degradation. Поэтому решающим является oracle intervention.

### Масштаб нелинейности

| `alpha` | R norm | R_orth norm | Jv norm | R / (alpha Jv) |
|---:|---:|---:|---:|---:|
| 1.00 | 3.77 | 3.60 | 13.63 | 0.276 |
| 1.50 | 8.40 | 8.05 | 13.67 | 0.409 |
| 2.00 | 16.02 | 15.59 | 13.63 | 0.587 |
| 2.25 | 21.28 | 20.88 | 13.56 | 0.698 |
| 2.50 | 28.36 | 27.96 | 13.49 | 0.841 |
| 3.00 | 40.14 | 39.58 | 13.56 | **0.986** |

При `alpha=3` nonlinear remainder почти равен по норме `alpha Jv`; orthogonal fraction на trajectories уже около 98.6%.

## 2. Финальный calibration

После исправления несовпадения первого короткого protocol финальный calibration использовал те же 8 prompts, 32-token length, grid и seed `37`, что behavior probe. Target layer и `beta=1` не менялись.

| метод | F@C80 | F@C85 | F@C90 |
|---|---:|---:|---:|
| additive | 45.49 | — | — |
| JRR | **100.00** | **72.11** | — |

Calibration gain на C80 = **+54.51**, поэтому frozen held-out был открыт. Этот calibration effect **не интерпретируется как ожидаемый held-out effect size**.

## 3. Frozen held-out

Протокол:

```text
20 prompts × seeds 11/23
alpha = 0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3
beta = 1
methods = additive, jrr_orth
```

Максимальный усреднённый concept:

- additive: **77.30**;
- JRR: **74.37**.

Поэтому заранее заданные C80/C85/C90 не достигнуты и confirmatory frontier **не оценивается**. Это не численная победа и не поражение.

Также manual per-prompt oracle generator отличается от batched generation в DPAR experiment; additive curve из JRR нельзя напрямую смешивать с DPAR frontier.

## 4. Same-alpha causal behavior

| `alpha` | delta concept | delta fluency | delta NLL |
|---:|---:|---:|---:|
| 1.00 | -1.91 | -2.44 | +0.183 |
| 1.25 | -0.30 | +4.65 | -0.049 |
| 1.50 | +7.44 | -8.20 | +0.095 |
| 1.75 | -11.13 | +9.17 | -0.107 |
| 2.00 | -6.16 | +2.63 | -0.031 |
| 2.25 | -6.46 | **+19.59** | **-0.250** |
| 2.50 | -3.77 | +3.17 | -0.043 |
| 3.00 | -2.93 | **+14.05** | **-0.229** |

Для `alpha=2.25` и `3.0` post-hoc paired bootstrap по 40 prompt/seed units даёт 95% intervals delta NLL полностью ниже нуля:

- `2.25`: **[-0.394, -0.110]**;
- `3.00`: **[-0.395, -0.071]**.

Это descriptive robustness check, а не preregistered significance test. Concept-difference intervals включают zero.

## 5. Seed sensitivity

На `alpha=3`:

| seed | delta concept | delta fluency |
|---:|---:|---:|
| 11 | **+8.76** | **+21.81** |
| 23 | **-14.62** | **+5.45** |

На `alpha=2.25` fluency улучшается примерно на 20 points у обоих seeds, но concept меняется на `-0.87` и `-12.05`.

Это показывает, что full `R_orth` removal не является robust Pareto-dominating method.

## 6. Что мы узнали причинно

Исходная сильная гипотеза была: «`R_orth` — collateral damage, его можно удалить, сохранив first-order concept effect».

Поддерживается только часть этой гипотезы:

1. strong steering действительно создаёт быстро растущую downstream nonlinearity;
2. почти весь сильный remainder ортогонален `Jv`;
3. его удаление способно существенно снизить NLL и восстановить fluency;
4. но оно может одновременно уменьшить target concept при сохранённом `Jv`.

Иными словами:

> **Ортогонально `Jv` не означает семантически неважно.**

Нелинейный ответ содержит как минимум две функциональные составляющие: harmful collateral distortion и useful nonlinear concept realization.

## 7. Exploratory frontier

Поскольку frozen high-concept thresholds недостижимы, within-support thresholds C50–C75 рассчитаны только post-hoc. JRR локально лучше около C60/C70 и хуже около C65; это согласуется с non-uniform same-alpha/seed evidence.

Файлы `heldout_exploratory_frontier.csv` и `heldout_exploratory_per_seed_frontier.csv` сохранены как descriptive appendix и **не являются основным success criterion**.

## 8. Связь со следующим экспериментом

Именно этот failure mode мотивировал Experiment 008: вместо удаления всего `R_orth` выбрать только локально harmful component. Его результат: [`../selective_jrr/RESULTS.md`](../selective_jrr/RESULTS.md).

## Сохранённые evidence-файлы

- `diagnostic_target_summary.csv`;
- `calibration_frontier.csv`;
- `heldout_aggregate.csv`;
- `heldout_frontier_frozen_thresholds.csv`;
- `heldout_same_alpha_deltas.csv`;
- `heldout_seed_deltas.csv`;
- `heldout_paired_bootstrap.csv`;
- `heldout_exploratory_frontier.csv`;
- `heldout_exploratory_per_seed_frontier.csv`.
