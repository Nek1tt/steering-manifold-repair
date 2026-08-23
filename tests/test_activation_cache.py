from pathlib import Path

import pytest
import torch

from steering_repair.activation_cache import (
    _stop_at_layer_for_hook,
    load_activation_cache,
    save_activation_cache,
)


def test_stop_at_layer_for_resid_post_hook():
    assert _stop_at_layer_for_hook("blocks.6.hook_resid_post") == 7
    assert _stop_at_layer_for_hook("blocks.0.hook_resid_post") == 1
    assert _stop_at_layer_for_hook("blocks.6.hook_resid_pre") is None
    assert _stop_at_layer_for_hook("not.a.real.hook") is None


def test_activation_cache_roundtrip(tmp_path: Path):
    path = tmp_path / "cache.pt"
    train = torch.randn(12, 8).half()
    val = torch.randn(3, 8).half()
    save_activation_cache(path, {"train": train, "val": val}, {"source": "test"})
    payload = load_activation_cache(path)
    assert torch.equal(payload["train"], train)
    assert torch.equal(payload["val"], val)
    assert payload["metadata"]["source"] == "test"
    assert not (tmp_path / "cache.pt.tmp").exists()


def test_activation_cache_rejects_mismatched_width(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"train": torch.randn(3, 8), "val": torch.randn(2, 7)}, path)
    with pytest.raises(ValueError, match="tensor shapes"):
        load_activation_cache(path)
