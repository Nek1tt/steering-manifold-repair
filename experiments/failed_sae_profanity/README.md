# Отрицательный контроль: SAE-вектор profanity

**Статус:** не прошёл обязательную валидацию steering-вектора.

## Зачем этот эксперимент сохранён

Первая попытка использовала интерпретируемую SAE-фичу как steering direction. Вмешательство заметно меняло поведение GPT-2 и при больших `alpha` приводило к деградации, но **не усиливало целевой текстовый concept**. Поэтому этот вектор нельзя использовать как baseline для оценки repair.

Это полезный отрицательный результат: интерпретируемая feature не обязана быть хорошим causal steering direction.

## Setup

- модель: GPT-2 Small;
- hook: `blocks.8.hook_resid_post`;
- SAE: OpenAI GPT-2 Small v5 128k, `resid_post_mlp`, layer 8;
- feature: `64840`, связанная с profanity;
- intervention: `h <- h + alpha * std(h) * v`;
- concept score: доля completions с совпадением по локальному profanity lexicon;
- fluency diagnostics: clean-model NLL, `distinct-3`, 3-gram repetition.

## Результат

| `alpha` | NLL | concept score, % | `distinct-3` |
|---:|---:|---:|---:|
| 0 | 2.896 | 0.0 | 0.984 |
| 4 | 2.872 | 0.0 | 0.988 |
| 8 | 2.946 | 0.0 | 0.985 |
| 16 | 4.118 | 0.0 | 0.997 |
| 24 | 3.990 | 0.0 | 0.898 |
| 32 | 3.308 | 0.0 | 0.400 |
| 48 | 3.006 | 0.0 | 0.000 |
| 64 | 2.798 | 0.0 | 0.000 |

Автоматическая проверка вернула `passed=False`, `concept_gain=0.0`.

## Вывод

Рост `alpha` меняет NLL/diversity и в итоге разрушает генерацию, но не создаёт измеримого profanity effect. Это нарушение шага задания «сначала провалидировать `h + alpha v`».

После этого baseline был заменён на contrastive sentiment direction в середине модели. Его результаты: [`../successful_sentiment_baseline/`](../successful_sentiment_baseline/).

Компактные данные этого run сохранены в `aggregate.csv`, исходный config — в `config.yaml`.
