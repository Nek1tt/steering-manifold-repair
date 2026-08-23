from __future__ import annotations

from typing import Literal

import torch


SAEWidth = Literal["32k", "128k"]


def load_openai_sae(
    *,
    location: str,
    layer: int,
    width: SAEWidth,
    device: torch.device | str,
):
    """Load one of OpenAI's released GPT-2 Small v5 SAEs."""
    import blobfile as bf
    import sparse_autoencoder

    path_fn = {
        "32k": sparse_autoencoder.paths.v5_32k,
        "128k": sparse_autoencoder.paths.v5_128k,
    }.get(width)
    if path_fn is None:
        raise ValueError(f"Unsupported OpenAI SAE width: {width!r}")

    path = path_fn(location, layer)
    with bf.BlobFile(path, mode="rb") as f:
        state_dict = torch.load(f, map_location="cpu", weights_only=False)
    # from_state_dict mutates its input by popping activation metadata.
    sae = sparse_autoencoder.Autoencoder.from_state_dict(dict(state_dict))
    sae.to(device)
    sae.eval()
    return sae


def decoder_direction(sae, feature_id: int, *, unit_norm: bool = True) -> torch.Tensor:
    """Return decoder column corresponding to one SAE latent."""
    weight = sae.decoder.weight
    if not (0 <= feature_id < weight.shape[1]):
        raise IndexError(f"feature_id={feature_id} outside [0, {weight.shape[1]})")
    v = weight[:, feature_id].detach().clone()
    if unit_norm:
        v = v / v.norm().clamp_min(1e-12)
    return v


@torch.no_grad()
def encode_feature(sae, activations: torch.Tensor, feature_id: int) -> torch.Tensor:
    """Encode activations and return one target latent with input shape preserved except d_model."""
    latents, _ = sae.encode(activations)
    return latents[..., feature_id]
