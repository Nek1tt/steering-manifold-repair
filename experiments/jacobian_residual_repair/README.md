# Experiment 007 — Jacobian Residual Repair (JRR)

Дата: 2026-08-23.

## Исследовательский вопрос

Предыдущие эксперименты анализировали repair в точке вмешательства. JRR проверяет другую гипотезу: сильный activation displacement может ломать fluency **после** source layer, потому что последующие Transformer blocks реагируют нелинейно.

Для downstream map `F`, clean state `h`, steering direction `v` и силы `alpha`:

\[
y_0=F(h),\qquad y_\alpha=F(h+\alpha v),
\]

\[
t=J_F(h)v,
\]

\[
R_\alpha=y_\alpha-y_0-\alpha t.
\]

`t` — first-order transported steering direction, `R_alpha` — точный nonlinear Taylor remainder.

Разлагаем

\[
R_\parallel=\operatorname{proj}_t(R_\alpha),
\qquad
R_\perp=R_\alpha-R_\parallel.
\]

Oracle JRR применяет

\[
y_{repair}=y_\alpha-\beta R_\perp,
\]

с заранее фиксированным `beta=1`.

Это принципиально отличается от DPAR: DPAR защищает steering axis в correction обученного denoiser на source layer, а JRR измеряет **нелинейный ответ самой модели downstream**.

## Протокол

Эксперимент жёстко разделён на стадии.

### A. Mechanistic diagnostic

Только calibration prompts. Для downstream layers 7–11 вычисляется directional JVP и измеряется `R_alpha`.

Проверяются предсказания:

- `||R_alpha||` растёт сверхлинейно; slope около 2 соответствует second-order regime;
- большая часть остатка не обязана быть parallel `Jv`;
- рост `R_orth` должен быть связан с деградацией fluency/NLL.

Target layer выбирается только по calibration data.

### B. Causal oracle

На выбранном layer exact per-token counterfactual remainder удаляется во время autoregressive generation.

Calibration сравнивает `additive` и `jrr_orth`. Held-out разрешается только после прохождения заранее заданного calibration gate. После просмотра held-out target layer, `beta`, thresholds, prompts и seeds не подстраиваются.

## Почему это oracle

Exact JRR дорог: для каждого generated token нужны clean/counterfactual forwards и directional derivative. Цель — не deployment, а причинный вопрос:

> Если бы nonlinear collateral response была известна точно, восстановило бы её удаление fluency без потери steering semantics?

Ответ оказался сложнее исходной гипотезы; результаты находятся в [`RESULTS.md`](RESULTS.md).

## Важная корректировка протокола

Первый короткий calibration attempt использовал несовпадающие с diagnostic behavior probe generation length/grid и не достигал даже требуемой concept support у additive control. Этот run был признан **невалидным для causal conclusion** до открытия held-out.

Финальный calibration был перезапущен с согласованными 8 prompts, 32-token generation, полным alpha grid и seed `37`; target layer и `beta=1` не менялись. В публичном архиве оставлены только итоговые evidence tables.

## Воспроизведение

```bash
pip install -r requirements.txt
pip install -e .

pytest -q tests/test_jrr.py tests/test_inference_followups.py tests/test_denoiser.py

python scripts/preflight_jrr.py --config configs/jrr_gpt2.yaml
python scripts/run_jrr_diagnostic.py --config configs/jrr_gpt2.yaml
python scripts/run_jrr_oracle.py --config configs/jrr_gpt2.yaml --phase calibration
# evaluation запускается только если calibration gate открыт
python scripts/run_jrr_oracle.py --config configs/jrr_gpt2.yaml --phase evaluation
```

Notebook: [`../../notebooks/jrr_experiment_colab.ipynb`](../../notebooks/jrr_experiment_colab.ipynb).

## Ключевые сохранённые файлы

- `diagnostic_target_summary.csv` — выбор downstream layer;
- `calibration_frontier.csv` — финальный валидный calibration;
- `heldout_aggregate.csv` — held-out aggregate;
- `heldout_frontier_frozen_thresholds.csv` — заранее заданные C80/C85/C90 thresholds;
- `heldout_same_alpha_deltas.csv` — causal same-alpha comparison;
- `heldout_seed_deltas.csv` — sensitivity по seeds;
- `heldout_paired_bootstrap.csv` — post-hoc descriptive bootstrap;
- `heldout_exploratory_frontier.csv` — явно exploratory within-support frontier.

Следующий selective test, уже выполненный после JRR: [`../selective_jrr/`](../selective_jrr/).
