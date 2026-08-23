# Воспроизводимость

Основной результат можно проверить **без повторного обучения**: опубликованный Gaussian denoiser автоматически загружается с Hugging Face, после чего запускается frozen held-out сравнение additive steering, vanilla denoising и DPAR.

Публичный checkpoint:

https://huggingface.co/Nek1tt/steering-repair-gpt2

Исходный код:

https://github.com/Nek1tt/steering-manifold-repair

## 1. Установка

Клонируйте репозиторий и создайте отдельное окружение:

```bash
git clone https://github.com/Nek1tt/steering-manifold-repair.git
cd steering-manifold-repair

python -m venv .venv
source .venv/bin/activate
```

Обновите `pip`:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### CUDA PyTorch

Полный generation evaluation и fresh retrain рассчитаны на NVIDIA GPU.

В `requirements.txt` указана общая зависимость `torch>=2.2`, поэтому CUDA-сборку PyTorch нужно установить **до** остальных зависимостей.

Конфигурация, на которой выполнялся clean-room запуск:

```bash
python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

После этого установите зависимости проекта:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Проверьте, что PyTorch видит GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Для GPU-запуска ожидается:

```text
cuda: True
```

Если используется другая CUDA/PyTorch конфигурация, установите совместимую CUDA-сборку PyTorch вместо приведённой выше.

## 2. Быстрые проверки

Unit tests:

```bash
python -m pytest -q
```

Real-model preflight основного repair pipeline:

```bash
python scripts/preflight_repair_suite.py \
  --config configs/retrain_gaussian_followups_gpt2.yaml
```

Ожидаемая финальная строка:

```text
REPAIR SUITE PREFLIGHT: PASS
```

Preflight загружает GPT-2 и WikiText, извлекает небольшой activation cache и проверяет используемый TransformerLens hook и размерность residual stream.

## 3. Основной DPAR result с checkpoint из Hugging Face

Запустите:

```bash
python scripts/run_hf_dpar_evaluation.py
```

Скрипт автоматически:

1. строит `results/sentiment_direction.pt` из `data/sentiment_positive.txt` и `data/sentiment_negative.txt`, если direction ещё отсутствует;
2. загружает свежую копию `retrained_denoiser_gaussian.pt` из `Nek1tt/steering-repair-gpt2`;
3. использует frozen held-out prompts, seeds и alpha grid из `configs/retrain_gaussian_followups_gpt2.yaml`;
4. сравнивает additive steering, vanilla Gaussian denoising (`beta=1`) и full DPAR (`beta=1`);
5. считает aggregate fluency/concept curves и интерполированный frontier;
6. проверяет точное сохранение steering projection у DPAR.

Результаты сохраняются в:

```text
results/hf_dpar_reproduction/samples.csv
results/hf_dpar_reproduction/aggregate.csv
results/hf_dpar_reproduction/interpolated_frontier.csv
```

Frozen evaluation protocol:

```text
prompts: data/prompts.txt
seeds: 11, 23
alpha: 0, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3
max_new_tokens: 64
temperature: 0.9
top_p: 0.95
```

Reference interpolated frontier:

| Метод | F@C80 | F@C85 | F@C90 |
|---|---:|---:|---:|
| Additive | 94.73 | 70.79 | 66.46 |
| Vanilla Gaussian (`beta=1`) | 80.90 | 75.74 | 52.60 |
| Full DPAR (`beta=1`) | 89.47 | 75.03 | 71.45 |

Основной reported result:

```text
DPAR F@C90     = 71.45
Additive F@C90 = 66.46
Gain           = +4.99 fluency points
```

`F@C90` — максимальная fluency на интерполированной aggregate curve при concept score не ниже 90.

Full DPAR `beta=1` входил в held-out method set заранее как mandatory full-repair control и не выбирался по held-out результатам.

Точные generated continuations могут немного меняться между версиями PyTorch, Transformers и CUDA. Поэтому `+4.99` — reference descriptive effect, а не требование битового совпадения generation output.

При этом главный geometric invariant DPAR должен сохраняться с численной точностью:

```text
|effective alpha - requested alpha| ≈ 0
```

Скрипт завершает работу ошибкой, если максимальная ошибка сохранения `alpha` превышает `1e-4`.

Reference результаты этого эксперимента:

```text
experiments/retrained_gaussian_followups/
```

## 4. Воспроизведение sentiment baseline

Основной DPAR-скрипт сам строит steering direction при необходимости. Baseline можно запустить отдельно:

```bash
python scripts/validate_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

Ожидаемая финальная строка:

```text
SENTIMENT VECTOR VALIDATION: PASS
```

Полный baseline sweep:

```bash
python scripts/run_sentiment_baseline.py \
  --config configs/baseline_sentiment_gpt2.yaml
```

Reference results:

```text
experiments/successful_sentiment_baseline/
```

Качественный результат: positive-sentiment concept растёт с увеличением `alpha`, а strong steering ухудшает NLL и fluency.

## 5. Воспроизведение обучения Gaussian denoiser

Этот раздел не нужен для проверки опубликованного HF checkpoint. Он проверяет отдельно, что learned component можно заново получить из frozen config и публичных данных.

### 5.1. Activation cache

```bash
python scripts/cache_activations.py \
  --config configs/retrain_gaussian_followups_gpt2.yaml
```

Используется:

```text
base model: GPT-2 Small
dataset: Salesforce/wikitext
subset: wikitext-2-raw-v1
split: train
hook: blocks.6.hook_resid_post
max activations: 80,000
train / validation: 72k / 8k
seed: 1234
```

Результат:

```text
results/retrain_layer6_generic_activations.pt
```

### 5.2. Fresh retrain

```bash
python scripts/train_denoiser.py \
  --config configs/retrain_gaussian_followups_gpt2.yaml \
  --kind gaussian
```

Результаты:

```text
checkpoints/retrained_denoiser_gaussian.pt
results/retrained_denoiser_gaussian_history.json
```

Reference final epoch:

```text
val_noisy_mse                = 8.7690718587
val_denoised_mse             = 2.8226513367
val_relative_mse_improvement = 0.6781128742
```

Архивированная training history:

```text
experiments/retrained_gaussian_followups/retrained_denoiser_gaussian_history.json
```

### 5.3. Автоматическая сверка retrain

```bash
python scripts/check_reproducibility.py
```

Ожидаемая финальная строка:

```text
REPRODUCIBILITY CHECK: PASS
```

Checker не требует битовой идентичности между разными GPU/CUDA/PyTorch environments.

Проверяются:

- `train_mse` и `val_denoised_mse` по эпохам с относительным tolerance 5%;
- `val_relative_mse_improvement` с tolerance 2 percentage points;
- улучшение learning curve;
- `kind=gaussian`;
- `d_model=768`;
- `hidden_dim=1536`;
- структура checkpoint;
- согласованность `best_val_mse` с fresh history.

`val_noisy_mse` используется только как диагностика: validation Gaussian corruption семплируется заново через PyTorch RNG и поэтому сильнее зависит от runtime.

Контрольный clean-room retrain дал:

```text
archived val_denoised_mse = 2.822651
fresh    val_denoised_mse = 2.925425
difference                = 3.64%

archived improvement = 67.8113%
fresh    improvement = 68.1768%
difference           = +0.366 percentage points
```

Fresh retrain перезаписывает локальный `checkpoints/retrained_denoiser_gaussian.pt`, но не изменяет опубликованный Hugging Face repository.

Чтобы прогнать тот же held-out evaluation на fresh checkpoint без повторной загрузки HF weights:

```bash
python scripts/run_hf_dpar_evaluation.py --skip-download
```

## 6. JRR

JRR не использует обученный checkpoint.

Preflight:

```bash
python scripts/preflight_jrr.py \
  --config configs/jrr_gpt2.yaml
```

Ожидаемая финальная строка:

```text
JRR PREFLIGHT: PASS
```

Mechanistic diagnostic:

```bash
python scripts/run_jrr_diagnostic.py \
  --config configs/jrr_gpt2.yaml
```

Reference diagnostics:

```text
log-log slope ||R_alpha|| vs alpha ≈ 1.9849
mean orthogonal fraction          ≈ 0.9404
R / (alpha Jv) at alpha=3         ≈ 0.986
```

Causal oracle:

```bash
python scripts/run_jrr_oracle.py \
  --config configs/jrr_gpt2.yaml \
  --phase calibration

python scripts/run_jrr_oracle.py \
  --config configs/jrr_gpt2.yaml \
  --phase evaluation
```

Полные результаты:

```text
experiments/jacobian_residual_repair/
```

## 7. KL-Selective JRR

Preflight:

```bash
python scripts/preflight_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml
```

Ожидаемая финальная строка:

```text
KL-SELECTIVE JRR PREFLIGHT: PASS
```

Frozen calibration:

```bash
python scripts/run_selective_jrr.py \
  --config configs/selective_jrr_gpt2.yaml \
  --phase calibration
```

В reported experiment strong-steering gate не был пройден:

```text
go_to_new_heldout = false
```

Поэтому новый held-out намеренно не запускался. Для воспроизведения reported protocol `--force` использовать не нужно.

Reference mechanistic result:

```text
mean selected fraction of R_orth = 7.93%
mean local KL reduction          = 41.6%
```

Полные результаты:

```text
experiments/selective_jrr/
```
