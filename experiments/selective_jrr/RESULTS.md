# Experiment 008 — результаты KL-Selective JRR

Дата: 2026-08-24.

## Итог

Заранее зафиксированный calibration gate **не пройден**, поэтому новый held-out (`data/selective_jrr_heldout_prompts.txt`, seeds `101/211`) намеренно не открывался.

Это частично положительный mechanistic result, а не implementation failure. KL-JRR действительно находит небольшую компоненту нелинейного остатка, которая непропорционально сильно отвечает за локальный clean-distribution KL. Но one-dimensional local KL selector недостаточен для устойчивого сохранения concept в extreme steering regime.

## Frozen calibration

- 8 prompts;
- seed `37`;
- target `blocks.7.hook_resid_post`;
- `beta=1`;
- методы: additive, full JRR, KL-JRR;
- `alpha = 0, 1, 1.5, 2, 2.25, 3, 4`;
- gate только на `alpha={2.25,3,4}`;
- pass: `delta fluency >= +5` и `delta concept >= -5` хотя бы в одной strong point.

Ни threshold, ни layer, ни beta, ни prompts/seeds после просмотра результата не менялись.

## Frontier на calibration

| метод | F@C70 | F@C75 | F@C80 |
|---|---:|---:|---:|
| additive | 85.66 | 48.71 | 45.49 |
| full JRR | **100.00** | **100.00** | **100.00** |
| KL-JRR | **100.00** | 97.57 | — |

Эта таблица показывает promising mid-concept region, но основным decision criterion был заранее заданный same-alpha gate.

## Same-alpha result

| `alpha` | delta concept | delta fluency | delta NLL |
|---:|---:|---:|---:|
| 1.00 | **+20.41** | +1.99 | -0.074 |
| 1.50 | +0.85 | **+14.44** | **-0.151** |
| 2.00 | **+14.01** | **+9.92** | **-0.128** |
| 2.25 | **+24.11** | +4.04 | -0.045 |
| 3.00 | **-35.45** | **+25.40** | **-0.332** |
| 4.00 | **-31.85** | **+13.21** | **-0.267** |

Ближайшая к gate strong point — `alpha=2.25`: concept вырос на `+24.11`, fluency на `+4.04`, но preregistered threshold был `+5`. Ослаблять его post-hoc нельзя.

На `alpha=3/4` fluency/NLL улучшаются сильно, но concept падает на десятки points. Это настоящий extreme-regime failure, а не следствие слишком строгого `+5` threshold.

## Mechanistic diagnostics

Selector удаляет лишь малую долю полного `R_orth`:

| `alpha` | selected fraction | KL before | KL after | reduction |
|---:|---:|---:|---:|---:|
| 1.00 | 5.17% | 0.0154 | 0.0130 | 15.9% |
| 1.50 | 6.55% | 0.0381 | 0.0290 | 23.9% |
| 2.00 | 6.90% | 0.0802 | 0.0546 | 32.0% |
| 2.25 | 7.06% | 0.1106 | 0.0688 | 37.8% |
| 3.00 | 8.99% | 0.2651 | 0.1503 | 43.3% |
| 4.00 | 7.73% | 0.5004 | 0.2812 | 43.8% |

На preregistered strong strengths компонент составляет в среднем лишь **7.93%** `R_orth`, но уменьшает local KL в среднем на **41.6%**.

`sel_transport_dot_removed` остаётся порядка `1e-7`, то есть first-order transported direction действительно защищён численно.

## Что поддерживается

1. Harmful часть nonlinear response имеет структуру: очень маленькая KL-sensitive component объясняет непропорционально большую часть local divergence.
2. Selective correction может улучшать NLL/fluency без ослабления projection вдоль `Jv`.
3. Разложение JRR на useful/harmful nonlinear computation содержательно, но one-step KL gradient не является достаточным separator.

## Что не поддерживается

Не поддержана сильная гипотеза, что один local KL-gradient mode чисто отделяет fluency damage от concept-carrying computation в strong regime.

Вероятное объяснение — autoregressive compounding: даже небольшая local correction может изменить sampled next token, после чего вся дальнейшая trajectory и realization concept расходятся.

## Decision

```text
go_to_new_heldout = false
```

Fresh held-out оставлен нетронутым. Это сохраняет честность protocol и не позволяет превращать почти прошедшую точку `alpha=2.25` в post-hoc «победу» изменением threshold.

Наиболее точная формулировка:

> KL-JRR находит компактный локальный harmful mode и даёт сильные улучшения в moderate regime, но local next-token divergence недостаточна как гарантия long-horizon semantic preservation при extreme activation steering.
