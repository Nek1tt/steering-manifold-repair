from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


SAEWidth = Literal["32k", "128k"]
_ALLOWED_LOCATIONS = {
    "resid_delta_attn",
    "resid_delta_mlp",
    "resid_post_attn",
    "resid_post_mlp",
}


def _openai_v5_path(location: str, layer: int, width: SAEWidth) -> str:
    if location not in _ALLOWED_LOCATIONS:
        raise ValueError(f"Unsupported OpenAI SAE location: {location!r}")
    if not 0 <= layer < 12:
        raise ValueError(f"GPT-2 Small layer must be in [0, 11], got {layer}")
    if width not in {"32k", "128k"}:
        raise ValueError(f"Unsupported OpenAI SAE width: {width!r}")
    return (
        "az://openaipublic/sparse-autoencoder/gpt2-small/"
        f"{location}_v5_{width}/autoencoders/{layer}.pt"
    )


class _TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


class OpenAIGPT2SAE(nn.Module):
    """Inference-compatible reader for OpenAI's released GPT-2 v5 SAEs."""

    def __init__(self, state_dict: dict) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.register_parameter(
            "weight", nn.Parameter(state_dict["encoder.weight"], requires_grad=False)
        )
        self.decoder = nn.Module()
        self.decoder.register_parameter(
            "weight", nn.Parameter(state_dict["decoder.weight"], requires_grad=False)
        )
        self.register_parameter(
            "pre_bias", nn.Parameter(state_dict["pre_bias"], requires_grad=False)
        )
        self.register_parameter(
            "latent_bias", nn.Parameter(state_dict["latent_bias"], requires_grad=False)
        )

        activation_name = state_dict.get("activation", "ReLU")
        if isinstance(activation_name, bytes):
            activation_name = activation_name.decode("utf-8")
        self.normalize = activation_name == "TopK"

        if activation_name == "TopK":
            activation_state = state_dict.get("activation_state_dict", {})
            k = activation_state.get("k", 32)
            if torch.is_tensor(k):
                k = int(k.item())
            self.activation: nn.Module = _TopK(int(k))
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        elif activation_name == "Identity":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported OpenAI SAE activation: {activation_name!r}")

    @staticmethod
    def _layer_norm(x: torch.Tensor, eps: float = 1e-5):
        # Matches openai/sparse_autoencoder/model.py (torch.std default is unbiased=True).
        mu = x.mean(dim=-1, keepdim=True)
        centered = x - mu
        std = centered.std(dim=-1, keepdim=True)
        return centered / (std + eps), mu, std

    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, {}
        x, mu, std = self._layer_norm(x)
        return x, {"mu": mu, "std": std}

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return F.linear(x, self.encoder.weight, self.latent_bias)

    def encode(self, x: torch.Tensor):
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict | None = None) -> torch.Tensor:
        out = F.linear(latents, self.decoder.weight) + self.pre_bias
        if self.normalize:
            if info is None:
                raise ValueError("Normalization metadata is required to decode TopK SAE latents")
            out = out * info["std"] + info["mu"]
        return out


def load_openai_sae(
    *,
    location: str,
    layer: int,
    width: SAEWidth,
    device: torch.device | str,
):
    """Load official OpenAI weights without installing their pinned old package."""
    import blobfile as bf

    path = _openai_v5_path(location, layer, width)
    with bf.BlobFile(path, mode="rb") as f:
        state_dict = torch.load(f, map_location="cpu", weights_only=False)
    sae = OpenAIGPT2SAE(dict(state_dict))
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
def feature_pre_activation(sae, activations: torch.Tensor, feature_id: int) -> torch.Tensor:
    """Target encoder pre-activation before TopK gating, for vector sanity checks."""
    x, _ = sae.preprocess(activations)
    x = x - sae.pre_bias
    weight = sae.encoder.weight[feature_id : feature_id + 1]
    bias = sae.latent_bias[feature_id : feature_id + 1]
    return F.linear(x, weight, bias).squeeze(-1)


@torch.no_grad()
def encode_feature(sae, activations: torch.Tensor, feature_id: int) -> torch.Tensor:
    latents, _ = sae.encode(activations)
    return latents[..., feature_id]
