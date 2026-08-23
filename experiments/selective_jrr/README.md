# Experiment 008 — KL-Selective JRR

Дата: 2026-08-24.

## Мотивация

JRR показал, что full removal `R_orth` может вернуть fluency, но одновременно удалить полезный concept-сигнал. Поэтому следующая гипотеза — **не удалять всю nonlinear response, а выбирать только локально вредную часть**.

## Метод

Как и в JRR:

\[
y_0=F(h),\qquad y_\alpha=F(h+\alpha v),
\]

\[
t=J_F(h)v,
\qquad
R=y_\alpha-y_0-\alpha t,
\]

\[
R_\perp=R-\operatorname{proj}_t(R).
\]

Далее на steered downstream state вычисляется локальный clean-distribution objective:

\[
L_{KL}(y)=KL(p_{clean}\|p_y),
\qquad
g=\nabla_yL_{KL}(y_\alpha).
\]

Чтобы не менять first-order transported steering, gradient ортогонализуется относительно `t`:

\[
g_\perp=g-\operatorname{proj}_t(g).
\]

Из `R_orth` выбирается только KL-increasing component:

\[
c=\max\left(0,\frac{\langle R_\perp,g_\perp\rangle}{\|g_\perp\|^2}\right),
\]

\[
R_{harm}=c\,g_\perp,
\qquad
y_{repair}=y_\alpha-R_{harm}.
\]

`beta=1` зафиксирован заранее; layer sweep и beta sweep не проводятся.

## Почему нужен новый held-out

KL-JRR был придуман после анализа JRR held-out, поэтому тот набор больше нельзя считать confirmatory для новой идеи.

Перед запуском Experiment 008 были заморожены:

- новый `data/selective_jrr_heldout_prompts.txt`;
- seeds `101/211`;
- target `blocks.7.hook_resid_post`;
- `beta=1`;
- strong strengths `alpha={2.25,3,4}`;
- calibration gate.

## Calibration protocol

```text
8 prompts
seed = 37
methods = additive, full JRR, KL-JRR
alpha = 0, 1, 1.5, 2, 2.25, 3, 4
32 new tokens
```

Gate считается только на `alpha={2.25,3,4}`. Для прохода на fresh held-out KL-JRR должен хотя бы в одной точке одновременно:

- улучшить fluency относительно additive минимум на `+5`;
- потерять не больше `5` concept points.

Этот gate не меняется после просмотра результата.

## Результат

Gate **не пройден**, поэтому fresh held-out не открывался. Это не означает, что метод бесполезен: selector находит компактную KL-sensitive component и даёт сильные улучшения на части moderate regime, но не обеспечивает semantic stability при extreme steering.

Полные результаты: [`RESULTS.md`](RESULTS.md).

## Воспроизведение

```bash
pip install -r requirements.txt
pip install -e .

pytest -q tests/test_selective_jrr.py tests/test_jrr.py
python scripts/preflight_selective_jrr.py --config configs/selective_jrr_gpt2.yaml
python scripts/run_selective_jrr.py --config configs/selective_jrr_gpt2.yaml --phase calibration
```

Evaluation запускается только если `calibration_summary.json` содержит `go_to_new_heldout=true`. Для reported result `--force` не используется.

Notebook: [`../../notebooks/selective_jrr_experiment_colab.ipynb`](../../notebooks/selective_jrr_experiment_colab.ipynb).

## Сохранённые evidence-файлы

- `calibration_frontier.csv`;
- `calibration_same_alpha.csv`.

Fresh held-out data остаётся в репозитории как frozen protocol, но его результаты отсутствуют намеренно: gate не позволил открыть evaluation.
