# Repair suite: Gaussian denoiser, DPAR и structured corruption

## Что проверяется

На замороженном sentiment baseline сравниваются:

1. **Gaussian denoiser** — baseline из задания, residual MLP на generic layer-6 activations;
2. **DPAR** — удаление из denoiser correction компоненты вдоль steering direction;
3. **Structured corruption** — denoiser, обученный на смеси Gaussian noise и natural activation differences;
4. ablations: `norm_preserving` и частичная direction preservation `lambda=0.5`.

Validation steering-вектор никогда не используется при обучении denoiser.

## Протокол

- GPT-2 Small, `blocks.6.hook_resid_post`;
- 20 prompts × 2 frozen seeds;
- `alpha = {0, 0.5, 0.75, 1, 1.5, 2, 3, 4}`;
- 7 методов, всего 2240 completions;
- 80k generic activations;
- 5 эпох обучения.

## Главный mechanistic result: vanilla denoiser отменяет steering

Для

$$
z=h+\alpha v,\qquad \Delta=D(z)-z
$$

vanilla Gaussian correction всё сильнее направляется против `v` по мере роста `alpha`.

| метод | `alpha` | effective `alpha` | correction cosine | correction / steering norm |
|---|---:|---:|---:|---:|
| Gaussian | 1.5 | 1.364 | -0.257 | 0.341 |
| Gaussian | 2.0 | 1.680 | -0.361 | 0.435 |
| Gaussian | 4.0 | 2.358 | -0.615 | 0.665 |
| Gaussian DPAR | 1.5 | 1.500 | ~0 | 0.326 |
| Gaussian DPAR | 2.0 | 2.000 | ~0 | 0.407 |
| Gaussian DPAR | 4.0 | 4.000 | ~0 | 0.520 |

Средняя absolute alpha-preservation error:

- vanilla Gaussian: **0.4354**;
- Gaussian DPAR: **≈5.2e-8**.

DPAR использует

$$
\Delta_\perp=\Delta-\operatorname{proj}_v(\Delta),
\qquad
h_{out}=z+\Delta_\perp,
$$

и тем самым устраняет steering-cancellation confound по построению. Поправка при этом остаётся существенной: DPAR не превращается в identity.

## Discrete frontier

| метод | F@C70 | F@C80 | F@C90 | F@C95 |
|---|---:|---:|---:|---:|
| additive | 99.56 | 88.74 | 63.35 | 18.01 |
| Gaussian | 100.00 | 72.46 | 26.23 | — |
| Gaussian DPAR | 100.00 | 89.34 | 71.05 | — |
| Gaussian `lambda=0.5` | 100.00 | 71.24 | 53.57 | 9.65 |
| mixed | 70.15 | 70.15 | — | — |
| mixed DPAR | 83.14 | 73.01 | — | — |
| norm preserving | 97.89 | 86.67 | — | — |

На coarse grid Gaussian DPAR выглядит лучше additive на C90 на `+7.69`, однако это **не финальная оценка**: интерполяция coarse additive points показывает, что преимущество могло быть артефактом сетки. Поэтому был проведён отдельный dense follow-up, где итоговый conservative effect равен `+4.99` fluency points на C90.

## Structured corruption: полезный отрицательный результат

Оба denoiser действительно специализируются на своей corruption geometry:

| checkpoint | eval corruption | относительное улучшение MSE |
|---|---|---:|
| Gaussian | Gaussian | 67.9% |
| Gaussian | structured | 50.8% |
| Mixed | Gaussian | 49.9% |
| Mixed | structured | 68.7% |

Но `mixed`/`mixed_dpar` downstream хуже и не достигают C90 на frozen grid.

Вывод: **лучшее reconstruction выбранного perturbation family недостаточно для лучшего steering repair**. Activation-space MSE и behavioral quality заметно расходятся.

## Статус гипотез

- Gaussian denoiser сам по себе улучшает frontier — **не поддержано**.
- Vanilla denoiser частично отменяет steering, DPAR это устраняет — **сильно поддержано**.
- Structured corruption улучшает downstream repair — **не поддержано**.

## Сохранённые артефакты

- `aggregate_compact.csv` — ключевые metrics всех методов/alpha;
- `frontier_summary.csv` — discrete frontier;
- `denoiser_cross_reconstruction.csv` — cross-corruption evaluation;
- `denoiser_gaussian_history.json`, `denoiser_mixed_history.json` — training history;
- `effective_alpha.png`, `correction_geometry.png` — mechanistic plots;
- `config.yaml` — frozen config.

Следующий, уже выполненный этап с fresh retrain и dense interpolation: [`../retrained_gaussian_followups/`](../retrained_gaussian_followups/).
