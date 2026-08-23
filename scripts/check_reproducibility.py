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
        "--max-relative-error",
        type=float,
        default=0.02,
        help="Допустимое относительное отличие training metrics (по умолчанию 2%%)",
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

    metric_keys = [
        "train_mse",
        "val_noisy_mse",
        "val_denoised_mse",
        "val_relative_mse_improvement",
    ]

    worst = ("", 0, 0.0)
    for got, ref in zip(actual, expected):
        if int(got["epoch"]) != int(ref["epoch"]):
            raise AssertionError(
                f"Epoch mismatch: actual={got['epoch']}, expected={ref['epoch']}"
            )
        for key in metric_keys:
            err = _relative_error(float(got[key]), float(ref[key]))
            if err > worst[2]:
                worst = (key, int(got["epoch"]), err)
            if err > args.max_relative_error:
                raise AssertionError(
                    f"{key} epoch {got['epoch']}: actual={got[key]:.9g}, "
                    f"expected={ref[key]:.9g}, relative_error={err:.3%} > "
                    f"{args.max_relative_error:.3%}"
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

    final_expected = float(expected[-1]["val_denoised_mse"])
    checkpoint_mse = float(payload["best_val_mse"])
    mse_err = _relative_error(checkpoint_mse, final_expected)
    if mse_err > args.max_relative_error:
        raise AssertionError(
            f"best_val_mse={checkpoint_mse:.9g}, expected={final_expected:.9g}, "
            f"relative_error={mse_err:.3%}"
        )

    state_dict = payload["state_dict"]
    n_params = sum(t.numel() for t in state_dict.values())

    print("REPRODUCIBILITY CHECK: PASS")
    print(f"Эпох: {len(actual)}")
    print(
        "Финальный val_denoised_mse: "
        f"{float(actual[-1]['val_denoised_mse']):.9f} "
        f"(архив {final_expected:.9f})"
    )
    print(
        "Финальное relative improvement: "
        f"{float(actual[-1]['val_relative_mse_improvement']):.6%}"
    )
    print(f"Максимальная относительная ошибка: {worst[2]:.4%} ({worst[0]}, epoch {worst[1]})")
    print(f"Checkpoint: kind={payload['kind']}, d_model={payload['d_model']}, hidden_dim={payload['hidden_dim']}")
    print(f"Параметров в state_dict: {n_params:,}")


if __name__ == "__main__":
    main()
