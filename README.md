# Steering Manifold Repair

Исследовательский проект по механистической интерпретируемости: **как уменьшить потерю связности текста при сильном activation steering** в GPT-2 Small.

Вместо одного «лучшего трюка» работа последовательно разбирает два разных механизма деградации и проверяет гипотезы, возникающие из ошибок предыдущих методов.

## Главное

Полный отчёт: **[`report/README.md`](report/README.md)**.

Лучший обученный checkpoint опубликован открыто на Hugging Face:

**https://huggingface.co/Nek1tt/steering-repair-gpt2**

Основные результаты:

1. **DPAR (Direction-Preserving Activation Repair).** Обычный Gaussian-denoiser при сильном steering частично улучшает связность просто потому, что отменяет сам steering. DPAR удаляет из поправки компоненту вдоль steering-вектора и сохраняет запрошенный `alpha` с численной точностью. В финальном плотном held-out sweep полный DPAR дал локальный выигрыш на высоком concept: `F@C90 = 71.45` против `66.46` у additive steering (`+4.99`). Это локальный, а не универсальный Pareto-выигрыш.
2. **JRR (Jacobian Residual Repair).** Сильный steering создаёт большой downstream-нелинейный остаток Тейлора. Его норма растёт примерно как `alpha^1.9849`, а при `alpha=3` становится почти равной по масштабу всему first-order эффекту `alpha Jv`. Причинное удаление остатка улучшает fluency/NLL, но показывает, что часть нелинейной динамики одновременно несёт полезный concept-сигнал.
3. **KL-Selective JRR.** Следующая гипотеза удаляет только локально KL-вредную компоненту нелинейного остатка. В сильном режиме она составляет в среднем лишь **7.93%** нормы `R_orth`, но уменьшает локальный KL на **41.6%**. Однако заранее зафиксированный сильный calibration gate не пройден: локальная next-token геометрия не гарантирует сохранение долгосрочного concept. Новый held-out поэтому не открывался.

Главный механистический вывод:

> Деградация при сильном steering состоит как минимум из двух разных эффектов: обученный repair может отменять сам steering, а собственная downstream-динамика модели становится существенно нелинейной. При этом «ортогонально steering-направлению» не означает «семантически неважно». Поэтому качественный repair должен сохранять больше, чем исходную steering-ось или её локальный first-order образ.

## Экспериментальная схема

Успешный baseline использует contrastive sentiment direction в середине GPT-2 Small:

\[
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}],
\]

в точке `blocks.6.hook_resid_post` с интервенцией

\[
h' = h + \alpha v.
\]

Concept score — вероятность positive sentiment по независимому локальному SST-2 classifier. Fluency score объединяет clean-model NLL, `distinct-3` и штраф за повторение 3-грамм, нормированный относительно `alpha=0`.

Baseline воспроизводит требуемый trade-off: concept растёт примерно с `27.9` до `95+`, а fluency при сильном steering падает со `100` до значений порядка `18` и ниже.

> Тексты prompts и sentiment-примеров в `data/` оставлены на английском намеренно: GPT-2 и sentiment judge тестировались на английской генерации. Это экспериментальные данные, а не документация.

## Архив экспериментов

| Эксперимент | Роль | Итог |
|---|---|---|
| [`failed_sae_profanity`](experiments/failed_sae_profanity/) | первая проверка SAE-вектора | отрицательный контроль: concept не изменился |
| [`successful_sentiment_baseline`](experiments/successful_sentiment_baseline/) | валидированный additive baseline | требуемый Pareto trade-off воспроизведён |
| [`repair_suite`](experiments/repair_suite/) | Gaussian denoiser, DPAR, structured corruption | найдено steering cancellation; structured corruption не помог |
| [`retrained_gaussian_followups`](experiments/retrained_gaussian_followups/) | свежий retrain + dense sweep | детерминированный retrain; локальный DPAR-выигрыш `+4.99` на C90 |
| [`jacobian_residual_repair`](experiments/jacobian_residual_repair/) | новая гипотеза о downstream-нелинейности | сильный mechanistic result; практический oracle-эффект неоднороден |
| [`selective_jrr`](experiments/selective_jrr/) | selective repair нелинейного остатка | частично положительный механизм; strong-regime gate не пройден |

Отрицательные результаты сохранены намеренно: они показывают, какие интуитивные объяснения не выдержали причинной проверки.

## Воспроизведение

Установка:

```bash
pip install -r requirements.txt
pip install -e .
```

Тесты:

```bash
pytest -q
```

Основные notebooks, по одному на этап:

```text
notebooks/baseline_colab.ipynb
notebooks/repair_experiments_colab.ipynb
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
notebooks/jrr_experiment_colab.ipynb
notebooks/selective_jrr_experiment_colab.ipynb
```

Конфиги экспериментов находятся в `configs/`, реализация — в `src/steering_repair/`, компактные итоговые таблицы — рядом с соответствующими `experiments/*/README.md` и `RESULTS.md`.

## Hugging Face

Публичный репозиторий лучшего практического обученного компонента:

**https://huggingface.co/Nek1tt/steering-repair-gpt2**

Там сохранены:

- `retrained_denoiser_gaussian.pt` — веса Gaussian activation denoiser;
- `checkpoint_metadata.json` — метаданные архитектуры и обучения;
- `training_config.yaml` — конфиг воспроизведения;
- `training_history.json` — история обучения;
- `README.md` — model card.

DPAR — это inference-time геометрия поверх этого checkpoint, а не отдельные обученные веса.

## Ограничения

- Основная модель — GPT-2 Small.
- Основной валидированный steering-вектор — sentiment/persona-style direction.
- Sentiment judge шумный и немонотонный по `alpha`, поэтому Pareto-результаты интерпретируются локально и с оговорками.
- JRR и KL-JRR — дорогие oracle/diagnostic методы для причинного анализа, а не готовые deployment-алгоритмы.
- Работа не заявляет универсального доминирования additive steering; основной вклад — сочетание практического DPAR-результата и механистического анализа причин деградации.

## Структура

```text
report/        финальный отчёт
experiments/   компактные результаты и разбор отдельных экспериментов
notebooks/     воспроизводимые сценарии запуска
configs/       зафиксированные конфиги
src/           реализация методов
scripts/       CLI для обучения, evaluation и preflight
tests/         unit tests
huggingface/   шаблон model card и публикация checkpoint
```

## Ссылки

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- OpenAI Sparse Autoencoder: https://github.com/openai/sparse_autoencoder
- SAELens: https://github.com/decoderesearch/SAELens
- Persona Vectors: https://github.com/safety-research/persona_vectors
- Generative Latent Prior: https://generative-latent-prior.github.io/
