from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Опубликовать финальный Gaussian activation-denoiser checkpoint для DPAR"
    )
    parser.add_argument("--checkpoint", required=True, help="Путь к retrained_denoiser_gaussian.pt")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo, например Nek1tt/steering-repair-gpt2")
    parser.add_argument("--private", action="store_true", help="Создать private repo; для задания этот флаг использовать не нужно")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"state_dict", "d_model", "hidden_dim", "kind", "train_config", "cache_metadata", "best_val_mse"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"В checkpoint отсутствуют ожидаемые поля: {missing}")
    if payload.get("kind") != "gaussian":
        raise RuntimeError(f"Ожидался Gaussian denoiser checkpoint, получен kind={payload.get('kind')!r}")

    metadata = {
        "architecture": "ResidualActivationDenoiser",
        "base_model": "gpt2",
        "hook_name": payload["cache_metadata"].get("hook_name", "blocks.6.hook_resid_post"),
        "d_model": int(payload["d_model"]),
        "hidden_dim": int(payload["hidden_dim"]),
        "kind": payload["kind"],
        "best_val_mse": float(payload["best_val_mse"]),
        "train_config": payload["train_config"],
        "cache_metadata": payload["cache_metadata"],
        "recommended_inference": {
            "method": "Direction-Preserving Activation Repair (DPAR)",
            "formula": "z=h+alpha*v; raw=D(z)-z; correction=raw-proj_v(raw); output=z+correction",
            "beta": 1.0,
            "note": "DPAR — inference-time geometry; она не закодирована непосредственно в weights checkpoint.",
        },
    }

    card_template = Path("huggingface/MODEL_CARD.md")
    if not card_template.exists():
        raise FileNotFoundError("huggingface/MODEL_CARD.md")

    create_repo(args.repo_id, repo_type="model", private=bool(args.private), exist_ok=True)
    api = HfApi()

    with tempfile.TemporaryDirectory(prefix="steering_repair_hf_") as tmp:
        folder = Path(tmp)
        shutil.copy2(checkpoint_path, folder / "retrained_denoiser_gaussian.pt")
        shutil.copy2(card_template, folder / "README.md")
        (folder / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        shutil.copy2("configs/retrain_gaussian_followups_gpt2.yaml", folder / "training_config.yaml")
        shutil.copy2("experiments/retrained_gaussian_followups/retrained_denoiser_gaussian_history.json", folder / "training_history.json")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message="Обновить финальный DPAR Gaussian activation denoiser",
        )

    print(f"Опубликовано: https://huggingface.co/{args.repo_id}")
    print("Публичный URL должен быть указан в README.md и report/README.md.")


if __name__ == "__main__":
    main()
