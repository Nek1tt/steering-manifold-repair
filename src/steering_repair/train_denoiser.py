from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .activation_cache import load_activation_cache
from .denoiser import CorruptionConfig, ResidualActivationDenoiser, corrupt_batch


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 1536
    batch_size: int = 512
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ratio_min: float = 0.02
    ratio_max: float = 3.0
    structured_probability: float = 0.5
    identity_probability: float = 0.05
    seed: int = 2026
    num_workers: int = 0


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate_denoiser(
    model: ResidualActivationDenoiser,
    val: torch.Tensor,
    *,
    kind: str,
    corruption: CorruptionConfig,
    train_bank: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_noisy = 0.0
    total_denoised = 0.0
    total_n = 0
    with torch.no_grad():
        for start in range(0, val.shape[0], batch_size):
            clean = val[start : start + batch_size].to(device=device, dtype=torch.float32)
            bank = train_bank if kind != "gaussian" else None
            corrupted, ratio, _, _ = corrupt_batch(
                clean,
                kind=kind,
                config=corruption,
                bank=bank,
            )
            pred = model(corrupted, ratio)
            total_noisy += F.mse_loss(corrupted, clean, reduction="sum").item()
            total_denoised += F.mse_loss(pred, clean, reduction="sum").item()
            total_n += int(clean.numel())
    noisy_mse = total_noisy / max(1, total_n)
    denoised_mse = total_denoised / max(1, total_n)
    return {
        "noisy_mse": float(noisy_mse),
        "denoised_mse": float(denoised_mse),
        "relative_mse_improvement": float(1.0 - denoised_mse / max(noisy_mse, 1e-12)),
    }


def train_denoiser(
    *,
    cache_path: str | Path,
    checkpoint_path: str | Path,
    history_path: str | Path,
    kind: str,
    config: TrainConfig,
    device: str = "auto",
) -> dict:
    kind = kind.lower()
    if kind not in {"gaussian", "structured", "mixed"}:
        raise ValueError("kind must be gaussian, structured, or mixed")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    dev = _device(device)

    cache = load_activation_cache(cache_path)
    train = cache["train"].float()
    val = cache["val"].float()
    d_model = int(train.shape[-1])
    model = ResidualActivationDenoiser(d_model=d_model, hidden_dim=config.hidden_dim).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    corruption = CorruptionConfig(
        ratio_min=config.ratio_min,
        ratio_max=config.ratio_max,
        structured_probability=config.structured_probability,
        identity_probability=config.identity_probability,
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=config.num_workers,
        drop_last=False,
    )

    bank_cpu = train.contiguous()
    history: list[dict] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(loader, desc=f"train {kind} denoiser e{epoch}/{config.epochs}", unit="batch")
        for (clean_cpu,) in pbar:
            clean = clean_cpu.to(device=dev, dtype=torch.float32, non_blocking=True)
            bank = None
            if kind != "gaussian":
                idx = torch.randint(0, bank_cpu.shape[0], (max(2048, clean.shape[0] * 4),))
                bank = bank_cpu[idx].to(device=dev, dtype=torch.float32, non_blocking=True)

            corrupted, ratio, _, structured_mask = corrupt_batch(
                clean,
                kind=kind,
                config=corruption,
                bank=bank,
            )
            pred = model(corrupted, ratio)
            loss = F.mse_loss(pred, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            running += float(loss.item()) * clean.shape[0]
            seen += int(clean.shape[0])
            pbar.set_postfix(
                mse=f"{loss.item():.4g}",
                structured=f"{structured_mask.float().mean().item():.2f}",
            )

        val_bank = bank_cpu[: min(8192, bank_cpu.shape[0])].to(dev)
        val_metrics = evaluate_denoiser(
            model,
            val,
            kind=kind,
            corruption=corruption,
            train_bank=val_bank,
            device=dev,
            batch_size=config.batch_size,
        )
        epoch_row = {
            "epoch": epoch,
            "train_mse": running / max(1, seen),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_row)
        print(
            f"epoch={epoch} train_mse={epoch_row['train_mse']:.6f} "
            f"val_denoised_mse={val_metrics['denoised_mse']:.6f} "
            f"improvement={100*val_metrics['relative_mse_improvement']:.1f}%"
        )
        if val_metrics["denoised_mse"] < best_val:
            best_val = val_metrics["denoised_mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Denoiser training produced no checkpoint")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "d_model": d_model,
            "hidden_dim": config.hidden_dim,
            "kind": kind,
            "train_config": asdict(config),
            "cache_metadata": cache.get("metadata", {}),
            "best_val_mse": best_val,
        },
        checkpoint_path,
    )
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    return {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "best_val_mse": float(best_val),
        "history": history,
    }


def load_denoiser_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualActivationDenoiser(
        d_model=int(payload["d_model"]), hidden_dim=int(payload["hidden_dim"])
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
