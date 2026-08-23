---
base_model: gpt2
library_name: pytorch
tags:
- mechanistic-interpretability
- activation-steering
- activation-denoising
- gpt2
---

# Gaussian activation denoiser для GPT-2 и DPAR

Этот checkpoint — лучший обученный practical component проекта `steering-manifold-repair`.

Модель представляет собой residual MLP, обученный на generic активациях GPT-2 Small из

```text
blocks.6.hook_resid_post
```

для восстановления clean residual-stream activations после Gaussian corruption. На inference checkpoint используется вместе с **Direction-Preserving Activation Repair (DPAR)**.

## DPAR

Для steered state

$$
z=h+\alpha v
$$

denoiser предлагает correction

$$
\Delta=D(z)-z.
$$

DPAR удаляет из неё компоненту вдоль steering direction. В явном виде:

$$
\Delta_\perp = \Delta - \frac{\langle \Delta,v\rangle}{\langle v,v\rangle}v.
$$

$$
h_{out}=z+\Delta_\perp.
$$

Таким образом, requested steering component сохраняется по построению, а denoiser может корректировать ортогональные направления.

Важно: DPAR — **inference-time geometry** и не зашит непосредственно в weights checkpoint.

## Обучение

- base model: GPT-2 Small;
- hook: `blocks.6.hook_resid_post`;
- corpus: WikiText-2 train stream;
- 80 000 cached activations: 72k train / 8k validation;
- `d_model=768`;
- hidden dimension `1536`;
- corruption: isotropic Gaussian activation noise с диапазоном relative severity;
- optimizer/config: `training_config.yaml`;
- exact history: `training_history.json`.

Fresh retrain точно воспроизвёл archived training history при frozen seeds/config. Финальное held-out relative activation-MSE improvement — примерно **67.8%**.

## Evaluation

Главный mechanistic result: vanilla denoiser correction по мере усиления steering всё сильнее направляется против steering direction. DPAR устраняет это cancellation и сохраняет effective `alpha` до численной точности.

Downstream improvement локальный, а не universal. В dense held-out follow-up full DPAR (`beta=1`) дал:

```text
F@C90 = 71.45
```

против additive steering:

```text
F@C90 = 66.46
```

то есть descriptive gain **+4.99 fluency points**. Sentiment score шумный и немонотонный, поэтому этот результат не заявляется как universal Pareto domination.

## Связанные mechanistic experiments

В GitHub также находятся два более поздних causal/oracle исследования без отдельных learned weights:

- **JRR (Jacobian Residual Repair):** strong steering создаёт примерно квадратичный downstream nonlinear Taylor remainder, сопоставимый в сильном режиме с first-order steering effect.
- **KL-Selective JRR:** небольшая local KL-sensitive component объясняет большую долю next-token divergence, но не гарантирует long-horizon concept preservation.

Checkpoint в этом Hugging Face repo остаётся наиболее валидированным practical learned component.

## Код и воспроизведение

GitHub:

```text
https://github.com/Nek1tt/steering-manifold-repair
```

Основной notebook для fresh retrain и dense DPAR follow-up:

```text
notebooks/retrain_gaussian_followups_fresh_colab.ipynb
```

## Ограничения

- Evaluation проведён на GPT-2 Small и sentiment/persona-style steering direction.
- Concept judge шумный и немонотонный по steering strength.
- DPAR имеет точную геометрическую гарантию только для projection вдоль steering axis; downstream text gains остаются threshold-dependent.
- Checkpoint не доказывает оптимальность Gaussian denoising для других models/concepts.
