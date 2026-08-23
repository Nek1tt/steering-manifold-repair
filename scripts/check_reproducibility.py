from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ARCHIVED_HISTORY = Path(
    "experiments/retrained_gaussian_followups/retrained_denoiser_gaussian_history.json"
)
DEFAULT_HISTORY = Path("results/retrained_denoiser_gaussian_history.json")
DEFAULT_CHECKPOINT = Path("checkpoints/retrained_denoiser_gaussian.pt")


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сверить локальный fresh retrain с архивированным Gaussian denoiser run"
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--max-curve-relative-error",
        type=float,
        default=0.05,
        help=(
            "Допустимое относительное отличие train/denoised MSE по эпохам "
            "(по умолчанию 5%%)"
        ),
    )
    parser.add_argument(
        "--max-improvement-absolute-error",
        type=float,
        default=0.02,
        help=(
            "Допустимое абсолютное отличие relative MSE improvement "
            "(0.02 = 2 percentage points)"
        ),
    )
    args = parser.parse_args()

    if not ARCHIVED_HISTORY.exists():
        raise FileNotFoundError(f"Не найден архив: {ARCHIVED_HISTORY}")
    if not args.history.exists():
        raise FileNotFoundError(
            f"Не найдена новая история: {args.history}. Сначала выполните fresh retrain."
        )
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Не найден checkpoint: {args.checkpoint}")

    expected = json.loads(ARCHIVED_HISTORY.read_text(encoding="utf-8"))
    actual = json.loads(args.history.read_text(encoding="utf-8"))

    if len(actual) != len(expected):
        raise AssertionError(
            f"Число эпох отличается: actual={len(actual)}, expected={len(expected)}"
        )

    worst_curve = ("", 0, 0.0)
    worst_noisy = (0, 0.0)
    worst_improvement = (0, 0.0)

    for got, ref in zip(actual, expected):
        epoch = int(got["epoch"])
        if epoch != int(ref["epoch"]):
            raise AssertionError(
                f"Epoch mismatch: actual={got['epoch']}, expected={ref['epoch']}"
            )

        # Эти метрики отражают само обучение denoiser и должны оставаться
        # близкими между окружениями. Битовая идентичность разных CUDA/PyTorch
        # сборок не требуется.
        for key in ("train_mse", "val_denoised_mse"):
            err = _relative_error(float(got[key]), float(ref[key]))
            if err > worst_curve[2]:
                worst_curve = (key, epoch, err)
            if err > args.max_curve_relative_error:
                raise AssertionError(
                    f"{key} epoch {epoch}: actual={got[key]:.9g}, "
                    f"expected={ref[key]:.9g}, relative_error={err:.3%} > "
                    f"{args.max_curve_relative_error:.3%}"
                )

        # Relative improvement является основной нормированной reconstruction
        # метрикой. Для долей корректнее использовать абсолютное отличие, а не
        # относительную ошибку относительно числа около 0.5-0.7.
        imp_err = abs(
            float(got["val_relative_mse_improvement"])
            - float(ref["val_relative_mse_improvement"])
        )
        if imp_err > worst_improvement[1]:
            worst_improvement = (epoch, imp_err)
        if imp_err > args.max_improvement_absolute_error:
            raise AssertionError(
                f"val_relative_mse_improvement epoch {epoch}: "
                f"actual={float(got['val_relative_mse_improvement']):.6%}, "
                f"expected={float(ref['val_relative_mse_improvement']):.6%}, "
                f"absolute_error={imp_err:.3%} > "
                f"{args.max_improvement_absolute_error:.3%}"
            )

        # val_noisy_mse намеренно только диагностический. evaluate_denoiser()
        # заново семплирует Gaussian corruption через torch RNG после каждой
        # эпохи; отдельный фиксированный validation generator не используется.
        # Поэтому эта величина может сильнее плавать между CUDA/PyTorch builds.
        noisy_err = _relative_error(
            float(got["val_noisy_mse"]), float(ref["val_noisy_mse"])
        )
        if noisy_err > worst_noisy[1]:
            worst_noisy = (epoch, noisy_err)

    # Learning curve должна реально улучшаться, а не просто случайно попасть
    # в tolerance на финальной точке.
    if float(actual[-1]["val_denoised_mse"]) >= float(actual[0]["val_denoised_mse"]):
        raise AssertionError("val_denoised_mse не улучшился между первой и последней эпохой")
    if float(actual[-1]["val_relative_mse_improvement"]) <= float(
        actual[0]["val_relative_mse_improvement"]
    ):
        raise AssertionError(
            "val_relative_mse_improvement не вырос между первой и последней эпохой"
        )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "d_model",
        "hidden_dim",
        "kind",
        "train_config",
        "cache_metadata",
        "best_val_mse",
    }
    missing = required.difference(payload)
    if missing:
        raise AssertionError(f"В checkpoint отсутствуют поля: {sorted(missing)}")

    if payload["kind"] != "gaussian":
        raise AssertionError(f"Ожидался kind=gaussian, получено {payload['kind']!r}")
    if int(payload["d_model"]) != 768:
        raise AssertionError(f"Ожидался d_model=768, получено {payload['d_model']}")
    if int(payload["hidden_dim"]) != 1536:
        raise AssertionError(
            f"Ожидался hidden_dim=1536, получено {payload['hidden_dim']}"
        )

    actual_best = min(float(row["val_denoised_mse"]) for row in actual)
    checkpoint_mse = float(payload["best_val_mse"])
    checkpoint_history_err = _relative_error(checkpoint_mse, actual_best)
    if checkpoint_history_err > 1e-6:
        raise AssertionError(
            f"best_val_mse в checkpoint ({checkpoint_mse:.9g}) не совпадает с "
            f"лучшей точкой fresh history ({actual_best:.9g})"
        )

    archived_best = min(float(row["val_denoised_mse"]) for row in expected)
    archived_best_err = _relative_error(checkpoint_mse, archived_best)
    if archived_best_err > args.max_curve_relative_error:
        raise AssertionError(
            f"best_val_mse={checkpoint_mse:.9g}, archived_best={archived_best:.9g}, "
            f"relative_error={archived_best_err:.3%} > "
            f"{args.max_curve_relative_error:.3%}"
        )

    state_dict = payload["state_dict"]
    n_params = sum(t.numel() for t in state_dict.values())

    final_actual = actual[-1]
    final_expected = expected[-1]
    print("REPRODUCIBILITY CHECK: PASS")
    print(f"Эпох: {len(actual)}")
    print(
        "Финальный val_denoised_mse: "
        f"{float(final_actual['val_denoised_mse']):.9f} "
        f"(архив {float(final_expected['val_denoised_mse']):.9f}, "
        f"ошибка {_relative_error(float(final_actual['val_denoised_mse']), float(final_expected['val_denoised_mse'])):.3%})"
    )
    print(
        "Финальное relative improvement: "
        f"{float(final_actual['val_relative_mse_improvement']):.6%} "
        f"(архив {float(final_expected['val_relative_mse_improvement']):.6%})"
    )
    print(
        "Максимальная ошибка устойчивой learning curve: "
        f"{worst_curve[2]:.3%} ({worst_curve[0]}, epoch {worst_curve[1]})"
    )
    print(
        "Максимальное абсолютное отличие improvement: "
        f"{worst_improvement[1]:.3%} (epoch {worst_improvement[0]})"
    )
    print(
        "Диагностический max drift val_noisy_mse: "
        f"{worst_noisy[1]:.3%} (epoch {worst_noisy[0]}; не является fail-критерием)"
    )
    print(
        f"Checkpoint: kind={payload['kind']}, d_model={payload['d_model']}, "
        f"hidden_dim={payload['hidden_dim']}"
    )
    print(f"Параметров в state_dict: {n_params:,}")


if __name__ == "__main__":
    main()
