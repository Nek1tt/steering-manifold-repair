# Успешный baseline: contrastive sentiment steering

**Статус:** валидация пройдена; этот baseline заморожен для дальнейших сравнений.

## Метод

Для GPT-2 Small строится midpoint contrastive direction:

$$
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}].
$$

Intervention применяется в `blocks.6.hook_resid_post`:

$$
h' = h + \alpha v.
$$

Concept strength измеряется независимо от построения вектора с помощью `distilbert-base-uncased-finetuned-sst-2-english`. Fluency объединяет clean-model NLL, `distinct-3` и anti-repetition относительно unsteered generation.

## Результат

Baseline воспроизводит требуемый trade-off: сначала sentiment усиливается почти без потери качества, затем при сильном steering fluency резко падает.

| `alpha` | fluency | concept | NLL |
|---:|---:|---:|---:|
| 0.00 | 100.00 | 27.92 | 2.896 |
| 0.25 | 99.88 | 48.09 | 2.758 |
| 0.50 | 100.00 | 66.15 | 2.836 |
| 0.75 | 99.56 | 76.63 | 2.900 |
| 1.00 | 88.74 | 84.18 | 3.015 |
| 2.00 | 63.35 | 93.59 | 3.352 |
| 4.00 | 18.01 | 95.31 | 4.610 |
| 16.00 | 6.67 | 99.91 | 5.572 |

Автоматическая проверка: `passed=True`, максимальный concept gain ≈ `72.0` points.

Практически наиболее полезный диапазон для repair — примерно `alpha=0.5..4`: он покрывает и связные умеренные вмешательства, и сильный high-concept regime.

## Вывод

Это контрольная additive curve для denoising/repair experiments. Prompt/seeds/judge/layer после её валидации не подстраиваются под результаты repair.

- Полная aggregate table: `aggregate.csv`.
- Pareto plot: `sentiment_baseline_pareto.png`.
- Следующий этап: [`../repair_suite/`](../repair_suite/).
