import torch

from steering_repair.sae import (
    OpenAIGPT2SAE,
    _openai_v5_path,
    decoder_direction,
    feature_pre_activation,
)


def _fake_topk_sae() -> OpenAIGPT2SAE:
    state = {
        "pre_bias": torch.zeros(3),
        "encoder.weight": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        "latent_bias": torch.zeros(4),
        "decoder.weight": torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 3.0, 1.0],
            ]
        ),
        "activation": "TopK",
        "activation_state_dict": {"k": 2, "postact_fn": "ReLU"},
    }
    return OpenAIGPT2SAE(state)


def test_openai_v5_path_matches_release_layout():
    assert _openai_v5_path("resid_post_mlp", 8, "128k").endswith(
        "resid_post_mlp_v5_128k/autoencoders/8.pt"
    )


def test_local_sae_topk_encode_shape_and_sparsity():
    sae = _fake_topk_sae()
    x = torch.tensor([[[-1.0, 0.5, 2.0], [0.2, -0.1, 1.5]]])
    latents, _ = sae.encode(x)
    assert latents.shape == (1, 2, 4)
    assert (latents > 0).sum(dim=-1).max().item() <= 2


def test_decoder_direction_is_decoder_column_and_unit_norm():
    sae = _fake_topk_sae()
    v = decoder_direction(sae, 1, unit_norm=True)
    assert torch.allclose(v, torch.tensor([0.0, 1.0, 0.0]))


def test_feature_pre_activation_preserves_batch_shape():
    sae = _fake_topk_sae()
    x = torch.randn(5, 3)
    score = feature_pre_activation(sae, x, 1)
    assert score.shape == (5,)
