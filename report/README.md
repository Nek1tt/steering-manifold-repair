# Steering Manifold Repair — финальный отчёт

## 1. Задача

Activation steering меняет скрытое состояние языковой модели по правилу

\[
\tilde h = h + \alpha v,
\]

где `v` кодирует желаемое свойство, а `alpha` задаёт силу вмешательства. При достаточно больших `alpha` нужное свойство усиливается, но генерация деградирует: растёт perplexity/NLL, текст теряет связность и появляется вырождение.

Цель работы — сдвинуть Pareto frontier «качество текста ↔ сила концепта» вверх и вправо, а также понять **почему** отдельные методы repair работают или не работают.

В задании предложен дешёвый baseline: обучить denoiser на естественных активациях с искусственным шумом и применять его после steering. Мы воспроизводим эту идею, а затем последовательно проверяем более содержательные гипотезы.

## 2. Экспериментальный setup

### Модель и точка вмешательства

- LM: GPT-2 Small, 12 блоков.
- Основной hook: `blocks.6.hook_resid_post`, то есть середина модели.
- Steering применяется к текущему response-token state во время autoregressive generation.

### Steering-вектор

После неудачной попытки с SAE-фичей используется contrastive sentiment/persona-style direction:

\[
v = \mathbb{E}[h\mid\text{positive}] - \mathbb{E}[h\mid\text{negative}].
\]

Вектор строится из отдельных positive/negative примеров. Denoiser никогда не обучается на этом validation-векторе и не знает о его существовании.

### Concept score

Concept strength измеряется независимым локальным классификатором
`distilbert-base-uncased-finetuned-sst-2-english` как вероятность positive sentiment.

### Fluency

Используются несколько независимых диагностик:

- NLL продолжения под clean GPT-2;
- `distinct-1/2/3`;
- повторяемость 3-грамм;
- агрегированный fluency score, нормированный относительно `alpha=0`.

Для сравнения методов основным объектом остаётся Pareto frontier, а не качество в одной произвольно выбранной точке `alpha`.

## 3. Валидация baseline

### 3.1. Отрицательный контроль: SAE profanity feature

Первая попытка использовала OpenAI GPT-2 Small SAE feature `64840` на layer 8. Steering заметно менял поведение модели и при больших `alpha` ухудшал генерацию, но текстовый profanity concept score оставался равен нулю.

Это важный отрицательный результат: интерпретируемая SAE-фича не обязана быть хорошим **causal steering direction**. Такой вектор нельзя использовать как контрольный Pareto frontier, поэтому эксперимент был остановлен до обучения repair.

Подробности: [`experiments/failed_sae_profanity/`](../experiments/failed_sae_profanity/).

### 3.2. Успешный sentiment baseline

Contrastive midpoint direction проходит fail-fast validation и воспроизводит требуемый trade-off:

| `alpha` | fluency | concept | NLL |
|---:|---:|---:|---:|
| 0.00 | 100.00 | 27.92 | 2.896 |
| 0.50 | 100.00 | 66.15 | 2.836 |
| 0.75 | 99.56 | 76.63 | 2.900 |
| 1.00 | 88.74 | 84.18 | 3.015 |
| 2.00 | 63.35 | 93.59 | 3.352 |
| 4.00 | 18.01 | 95.31 | 4.610 |
| 16.00 | 6.67 | 99.91 | 5.572 |

То есть при росте steering нужное свойство сначала усиливается почти без потери fluency, а затем связность быстро разрушается.

Подробности: [`experiments/successful_sentiment_baseline/`](../experiments/successful_sentiment_baseline/).

## 4. Gaussian denoiser: baseline из задания

На 80 000 generic WikiText-2 активаций layer 6 обучается residual MLP:

\[
D_\theta(h+\delta, r) \approx h,
\qquad
r = \|\delta\|/\|h\|.
\]

Архитектура обусловлена относительной силой шума `r`; residual output инициализируется близко к identity map. Validation relative MSE improvement после пяти эпох достигает примерно **67.8%**, поэтому downstream-проблемы нельзя объяснить тем, что denoiser просто не научился reconstruction-задаче.

Однако vanilla Gaussian repair не улучшает high-concept frontier. Это приводит к первой механистической гипотезе.

## 5. DPAR: Direction-Preserving Activation Repair

### 5.1. Гипотеза

Пусть

\[
z = h + \alpha v,
\qquad
\Delta = D(z)-z.
\]

Denoiser может улучшить fluency самым тривиальным способом — направить `Delta` против `v` и частично отменить steering. Тогда apparent repair не является repair при сопоставимой силе концепта.

DPAR разлагает поправку:

\[
\Delta_\parallel = \operatorname{proj}_v(\Delta),
\qquad
\Delta_\perp = \Delta - \Delta_\parallel,
\]

и применяет только

\[
h_{\text{DPAR}} = z + \Delta_\perp.
\]

По построению компонент вдоль steering axis сохраняется.

### 5.2. Что происходит у vanilla denoiser

При росте `alpha` поправка Gaussian denoiser действительно всё сильнее направляется против steering-вектора:

| `alpha` | requested `alpha` | effective `alpha` | cosine(`Delta`, `v`) |
|---:|---:|---:|---:|
| 1.5 | 1.5 | 1.364 | -0.257 |
| 2.0 | 2.0 | 1.680 | -0.361 |
| 4.0 | 4.0 | 2.358 | -0.615 |

Средняя абсолютная ошибка сохранения `alpha` у vanilla Gaussian составляет **0.4354**.

У DPAR effective `alpha` совпадает с requested `alpha` до численной погрешности; средняя ошибка порядка **5.2e-8**. При этом поправка не исчезает: например, при `alpha=2` её ортогональная норма составляет около 41% нормы steering perturbation.

Это прямое подтверждение steering-cancellation failure mode.

### 5.3. Structured corruption

Дополнительно обучен denoiser на смеси Gaussian noise и нормированных natural activation differences `h_j-h_k`. Он действительно специализируется на своей corruption geometry:

| checkpoint | eval corruption | относительное улучшение MSE |
|---|---|---:|
| Gaussian | Gaussian | 67.9% |
| Gaussian | structured | 50.8% |
| Mixed | Gaussian | 49.9% |
| Mixed | structured | 68.7% |

Но downstream эта специализация не помогает: `mixed` и `mixed_dpar` не достигают concept 90 на frozen grid. Следовательно, лучшее reconstruction выбранного семейства perturbations **не гарантирует** лучший steering repair.

Подробности: [`experiments/repair_suite/`](../experiments/repair_suite/).

## 6. Dense follow-up: насколько DPAR практически полезен

Чтобы отделить геометрию от величины correction, введён независимый масштаб `beta`:

\[
h_{out}=z+\beta\,\Delta_{filtered}.
\]

Gaussian denoiser был заново обучен в fresh runtime. История обучения совпала с исходной **точно на каждой эпохе**, а финальное validation improvement снова составило **67.8%** (`val_denoised_mse = 2.822651`). Это сильная проверка воспроизводимости training pipeline.

Calibration выбирает `beta=0.25` и для DPAR, и для vanilla, но dense held-out sweep показывает, что оптимальный масштаб зависит от concept region.

Interpolated held-out frontier:

| метод | F@C80 | F@C85 | F@C90 | F@C95 |
|---|---:|---:|---:|---:|
| additive | **94.73** | 70.79 | 66.46 | — |
| DPAR `beta=0.25` | 89.53 | 79.16 | 55.16 | 48.84 |
| DPAR `beta=1.00` | 89.47 | 75.03 | **71.45** | — |
| vanilla `beta=0.25` | 89.11 | **81.06** | 47.65 | — |
| vanilla `beta=1.00` | 80.90 | 75.74 | 52.60 | — |

Главный practical результат:

\[
F@C90:\quad 71.45\;\text{(DPAR)}\;\;\text{vs}\;\;66.46\;\text{(additive)},
\]

то есть **+4.99 fluency points**.

На обоих frozen seeds направление эффекта на C90 совпадает: `+10.53` и `+20.71`, но concept judge заметно шумный и немонотонный, поэтому это **descriptive local evidence**, а не доказательство универсального доминирования.

Важно: в первом coarse grid казалось, что выигрыш DPAR равен `+7.69`. После плотной интерполяции этот эффект уменьшается до `+4.99`; в финальной работе используется именно более консервативная оценка.

Подробности: [`experiments/retrained_gaussian_followups/`](../experiments/retrained_gaussian_followups/).

## 7. JRR: нелинейная downstream-динамика steering

DPAR исправляет геометрию repair на source layer, но не объясняет, почему большой displacement ломает последующие вычисления Transformer. Поэтому следующая гипотеза рассматривает downstream map `F`.

Для clean state `h`:

\[
y_0 = F(h),
\qquad
y_\alpha = F(h+\alpha v),
\]

а first-order transported steering равен

\[
t = J_F(h)v.
\]

Определим точный nonlinear Taylor remainder:

\[
R_\alpha = y_\alpha-y_0-\alpha t.
\]

### 7.1. Диагностический результат

Calibration выбирает `blocks.7.hook_resid_post`. На этом слое:

| диагностика | значение |
|---|---:|
| log-log slope `||R_alpha||` по `alpha` | **1.9849** |
| средняя доля остатка, ортогональная `Jv` | **0.9404** |
| rank corr `||R_orth||` vs NLL | +0.8909 |
| rank corr `||R_orth||` vs fluency | -0.8909 |

Slope практически совпадает с second-order prediction `O(alpha^2)`. Корреляции интерпретируются осторожно, потому что их может частично объяснять общий рост `alpha`; поэтому далее проводится причинный oracle test.

На held-out autoregressive trajectories нелинейность становится очень большой:

| `alpha` | `||R||` | `||R_orth||` | `||Jv||` | `||R|| / ||alpha Jv||` |
|---:|---:|---:|---:|---:|
| 1.0 | 3.77 | 3.60 | 13.63 | 0.276 |
| 1.5 | 8.40 | 8.05 | 13.67 | 0.409 |
| 2.0 | 16.02 | 15.59 | 13.63 | 0.587 |
| 2.25 | 21.28 | 20.88 | 13.56 | 0.698 |
| 2.5 | 28.36 | 27.96 | 13.49 | 0.841 |
| 3.0 | 40.14 | 39.58 | 13.56 | **0.986** |

При `alpha=3` nonlinear remainder почти равен по норме всему first-order displacement `alpha Jv`.

### 7.2. Causal oracle

JRR удаляет компоненту остатка, ортогональную transported direction:

\[
y_{repair}=y_\alpha-R_\perp.
\]

Calibration дала `F@C80=100.00` против `45.49` у additive и открыла frozen held-out. Этот calibration gain **не переносится в отчёт как held-out effect size**.

В held-out manual oracle protocol обе кривые после усреднения seeds не достигли заранее выбранных C80/C85/C90; confirmatory frontier поэтому **не оценивается**, а не считается победой или поражением.

При одинаковом `alpha` causal effect на fluency/NLL всё же заметен:

| `alpha` | delta concept | delta fluency | delta NLL |
|---:|---:|---:|---:|
| 2.25 | -6.46 | **+19.59** | **-0.250** |
| 3.00 | -2.93 | **+14.05** | **-0.229** |

Post-hoc paired bootstrap по 40 prompt/seed units даёт интервалы delta NLL, полностью ниже нуля:

- `alpha=2.25`: `[-0.394, -0.110]`;
- `alpha=3.00`: `[-0.395, -0.071]`.

Это описательная post-hoc диагностика, не preregistered significance test.

### 7.3. Ключевой failure mode

На `alpha=3` seed sensitivity принципиальна:

| seed | delta concept | delta fluency |
|---:|---:|---:|
| 11 | **+8.76** | **+21.81** |
| 23 | **-14.62** | **+5.45** |

Полное удаление `R_orth` улучшает fluency, но иногда уничтожает concept-сигнал. Значит:

> **Ортогонально `Jv` не означает семантически неважно.**

Нелинейный ответ содержит смесь fluency-damaging collateral computation и useful nonlinear adaptation, необходимой для реализации concept.

Подробности: [`experiments/jacobian_residual_repair/`](../experiments/jacobian_residual_repair/).

## 8. KL-Selective JRR: проверка selective residual repair

JRR прямо мотивирует следующую гипотезу: не удалять весь `R_orth`, а выбрать только локально вредную часть.

В точке `y_alpha` вычисляется

\[
g = \nabla_y KL(p_{clean}\|p_y),
\]

затем gradient защищается от изменения first-order steering:

\[
g_\perp=g-\operatorname{proj}_{Jv}(g),
\]

и удаляется только положительная projection `R_orth` на `g_perp`.

Поскольку идея появилась **после** анализа JRR held-out, для неё заранее заморожен новый набор prompts и новые seeds `101/211`. Старый held-out не переиспользуется как confirmatory.

### 8.1. Что поддерживается

KL selector действительно очень компактный. На заранее выбранных сильных `alpha={2.25,3,4}` он удаляет в среднем только **7.93%** нормы `R_orth`, но уменьшает локальный clean-distribution KL на **41.6%**.

В умеренном режиме есть сильные Pareto-like улучшения при том же `alpha`:

| `alpha` | delta concept | delta fluency | delta NLL |
|---:|---:|---:|---:|
| 1.50 | +0.85 | **+14.44** | **-0.151** |
| 2.00 | **+14.01** | **+9.92** | **-0.128** |
| 2.25 | **+24.11** | +4.04 | -0.045 |

### 8.2. Что не поддерживается

Frozen gate требовал на одном из `alpha={2.25,3,4}` одновременно:

- gain fluency не меньше `+5`;
- loss concept не хуже `-5`.

`alpha=2.25` не добрал примерно один fluency point (`+4.04`), а на `alpha=3/4` fluency сильно вырос, но concept упал на `-35.45/-31.85`.

Поэтому

```text
go_to_new_heldout = false
```

и новый held-out **намеренно не открывался**. Порог не ослаблялся post-hoc.

Результат уточняет механизм: небольшая KL-sensitive component действительно концентрирует локальный harmful effect, но one-step next-token sensitivity не гарантирует long-horizon semantic preservation из-за autoregressive compounding.

Подробности: [`experiments/selective_jrr/`](../experiments/selective_jrr/).

## 9. Что именно нового даёт работа

Предложенный в задании Gaussian denoiser сам по себе не оказался универсальным решением. Основной вклад работы — в цепочке причинных проверок:

1. **Steering cancellation:** learned denoiser может улучшать текст просто ослабляя intervention. DPAR выявляет и устраняет этот confound точно по геометрии.
2. **Downstream nonlinearity:** сильный source-layer displacement порождает почти квадратичный nonlinear response, сравнимый по масштабу с first-order steering.
3. **Functional mixture:** causal ablation показывает, что nonlinear residual содержит и harmful, и useful computation.
4. **Selective-mode test:** локальный KL-gradient находит очень компактную harmful component, но её удаление не гарантирует сохранение sequence-level concept.

Таким образом, работа не ограничивается сравнением final scores: каждый новый эксперимент появляется из конкретного failure mode предыдущего и либо подтверждает, либо опровергает механизм.

## 10. Ограничения

- Проверена только GPT-2 Small.
- Основной валидированный concept — positive sentiment; generalization на другие steering directions не доказана.
- Sentiment classifier даёт шумный, немонотонный по `alpha` score, поэтому небольшие frontier differences нельзя переинтерпретировать как статистически строгие universal wins.
- JRR и KL-JRR вычислительно дороги и используются как causal oracle/diagnostic, а не deployment method.
- DPAR гарантирует сохранение projection вдоль `v`, но не гарантирует сохранение всей нелинейной семантики модели.
- Bootstrap для JRR был сделан post-hoc и используется только как описательная устойчивость NLL-сигнала.

## 11. Воспроизводимость

В репозитории сохранены:

- frozen configs;
- unit tests;
- notebooks для каждого основного этапа;
- компактные CSV с ключевыми aggregate/frontier результатами;
- negative experiments;
- exact Gaussian training history.

Основные notebooks:

```text
notebooks/baseline_colab.ipynb
notebooks/repair_experiments_colab.ipynb
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
notebooks/jrr_experiment_colab.ipynb
notebooks/selective_jrr_experiment_colab.ipynb
```

Лучший practical learned checkpoint опубликован открыто:

**https://huggingface.co/Nek1tt/steering-repair-gpt2**

Это Gaussian activation denoiser; DPAR применяется к его correction на inference-time.

## 12. Итог

Самая сильная формулировка результата:

> Vanilla activation denoising имеет измеримый steering-cancellation failure mode, который DPAR устраняет по построению и даёт локальный high-concept Pareto-выигрыш. Независимо от этого сильный steering создаёт приблизительно квадратичную downstream-нелинейность, которая в сильном режиме сопоставима с first-order steering. Причинные интервенции показывают, что эта нелинейность содержит как fluency-damaging, так и concept-carrying computation. Поэтому coherence-preserving steering требует сохранять больше, чем исходную steering-ось, её first-order transported image или локальный one-step KL objective.
