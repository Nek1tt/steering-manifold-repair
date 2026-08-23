# Проверка воспроизводимости

Этот файл описывает, как проверить решение из чистого окружения, не полагаясь на локальные артефакты автора.

## 1. Чистый clone и окружение

На Windows + NVIDIA GPU:

```powershell
git clone https://github.com/Nek1tt/steering-manifold-repair.git steering-repro-check
cd steering-repro-check

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Проверка GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Быстрый обязательный smoke test

Сначала запускаются все unit tests, включая статическую проверку Markdown:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Затем real-model preflight для основного learned repair pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_repair_suite.py --config configs\repair_suite_gpt2.yaml
```

Ожидаемая финальная строка:

```text
REPAIR SUITE PREFLIGHT: PASS
```

Этот preflight реально загружает GPT-2 и WikiText, извлекает небольшой activation cache и проверяет правильный hook/размерность.

## 3. Воспроизведение steering baseline

Baseline полностью строит sentiment direction из файлов `data/sentiment_positive.txt` и `data/sentiment_negative.txt`; готовый steering-вектор не требуется.

Сначала fail-fast validation строит и сохраняет calibrated direction:

```powershell
.\.venv\Scripts\python.exe scripts\validate_sentiment_baseline.py --config configs\baseline_sentiment_gpt2.yaml
```

Ожидаемая финальная строка:

```text
SENTIMENT VECTOR VALIDATION: PASS
```

После этого запускается полный baseline sweep:

```powershell
.\.venv\Scripts\python.exe scripts\run_sentiment_baseline.py --config configs\baseline_sentiment_gpt2.yaml
```

После этих шагов должны появиться:

```text
results/sentiment_direction.pt
results/sentiment_baseline_samples.csv
```

Качественный критерий воспроизведения: positive-sentiment concept заметно растёт с `alpha`, а fluency/NLL деградируют при сильном steering. Архивированная reference-таблица находится в `experiments/successful_sentiment_baseline/aggregate.csv`.

## 4. Самая сильная проверка: fresh retrain Gaussian denoiser

Ключевой обученный checkpoint можно воспроизвести только из кода, конфига и публичных данных.

Сначала строится activation cache:

```powershell
.\.venv\Scripts\python.exe scripts\cache_activations.py --config configs\retrain_gaussian_followups_gpt2.yaml
```

Затем обучается только Gaussian denoiser:

```powershell
.\.venv\Scripts\python.exe scripts\train_denoiser.py --config configs\retrain_gaussian_followups_gpt2.yaml --kind gaussian
```

Ожидаемые выходы:

```text
checkpoints/retrained_denoiser_gaussian.pt
results/retrained_denoiser_gaussian_history.json
```

Архивированная reference history находится в:

```text
experiments/retrained_gaussian_followups/retrained_denoiser_gaussian_history.json
```

Reference final epoch:

```text
val_noisy_mse                = 8.7690718587
val_denoised_mse             = 2.8226513367
val_relative_mse_improvement = 0.6781128742
```

Автоматическая сверка:

```powershell
.\.venv\Scripts\python.exe scripts\check_reproducibility.py
```

Успех:

```text
REPRODUCIBILITY CHECK: PASS
```

Проверка намеренно не требует битовой идентичности разных GPU/CUDA/PyTorch сборок. Устойчивые критерии следующие:

- `train_mse` и `val_denoised_mse` на каждой эпохе должны отличаться от архивированной learning curve не более чем на 5% относительно;
- `val_relative_mse_improvement` должен отличаться не более чем на 2 percentage points абсолютно;
- learning curve должна реально улучшаться от первой к последней эпохе;
- `kind`, `d_model=768`, `hidden_dim=1536` и структура checkpoint проверяются точно;
- `best_val_mse` checkpoint должен совпадать с лучшей точкой fresh history и быть в пределах 5% от архивированного best value.

`val_noisy_mse` выводится как диагностика, но не является fail-критерием. Причина: текущий `evaluate_denoiser()` заново семплирует Gaussian validation corruption после каждой эпохи через torch RNG и не использует отдельный фиксированный validation generator. Поэтому эта величина закономерно сильнее плавает между CUDA/PyTorch builds, чем качество самого обученного denoiser.

Контрольный clean-room запуск из отдельного Windows clone воспроизвёл финальный результат:

```text
archived val_denoised_mse = 2.822651
fresh    val_denoised_mse = 2.925425   (3.64% difference)

archived improvement = 67.8113%
fresh    improvement = 68.1768%        (+0.366 percentage points)
```

Максимальное отличие `val_denoised_mse` по пяти эпохам в этом clean-room run составило около 4.10%; максимальный диагностический drift `val_noisy_mse` — около 4.83%.

## 5. Проверка mechanistic методов

После построения `results/sentiment_direction.pt` можно выполнить дешёвые numerical/real-model preflights:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_jrr.py --config configs\jrr_gpt2.yaml
.\.venv\Scripts\python.exe scripts\preflight_selective_jrr.py --config configs\selective_jrr_gpt2.yaml
```

Ожидаемые финальные строки:

```text
JRR PREFLIGHT: PASS
KL-SELECTIVE JRR PREFLIGHT: PASS
```

Они проверяют hook semantics, directional JVP и локальную KL-gradient geometry независимо от сохранённых итоговых CSV.

## 6. Полное воспроизведение результатов

Полные эксперименты существенно дороже smoke test. Для каждого этапа сохранён отдельный notebook:

```text
notebooks/baseline_colab.ipynb
notebooks/repair_experiments_colab.ipynb
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
notebooks/jrr_experiment_colab.ipynb
notebooks/selective_jrr_experiment_colab.ipynb
```

И соответствующие frozen configs в `configs/`.

Важно: для JRR/KL-JRR calibration/held-out протоколы нельзя менять после просмотра результатов. В частности, Experiment 008 не прошёл заранее заданный calibration gate, поэтому его новый held-out намеренно не запускался.

## 7. Проверка опубликованного checkpoint

Публичный checkpoint:

https://huggingface.co/Nek1tt/steering-repair-gpt2

В репозитории Hugging Face должны быть доступны без авторизации:

```text
retrained_denoiser_gaussian.pt
README.md
checkpoint_metadata.json
training_config.yaml
training_history.json
```

Checkpoint на Hugging Face является тем же типом Gaussian activation denoiser, который воспроизводится в разделе 4; DPAR — inference-time геометрия поверх его correction.

## Минимальный критерий воспроизводимости перед сдачей

Перед отправкой решения достаточно убедиться, что из отдельного чистого clone выполняются четыре пункта:

1. `pytest -q` проходит;
2. `preflight_repair_suite.py` заканчивается `PASS`;
3. `validate_sentiment_baseline.py` заканчивается `PASS`, а полный baseline строит expected trade-off;
4. fresh Gaussian retrain проходит `scripts/check_reproducibility.py`.

Это проверяет установку, данные, model hooks, обучение, формат checkpoint и основной экспериментальный pipeline независимо от старого рабочего окружения.
