"""Tests for the projection-free, pre-LayerNorm LeWM variant."""

from types import SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.wm.lewm import LeWMPreLN


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, pixels, **kwargs):
        batch_size = pixels.size(0)
        pre_ln = torch.arange(
            batch_size * 4 * 3, dtype=pixels.dtype, device=pixels.device
        ).reshape(batch_size, 4, 3)
        post_ln = torch.full_like(pre_ln, -1)
        assert kwargs['output_hidden_states'] is True
        return SimpleNamespace(
            last_hidden_state=post_ln,
            hidden_states=(torch.zeros_like(pre_ln), pre_ln),
        )


def make_model():
    return LeWMPreLN(
        encoder=DummyEncoder(),
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
        projector=nn.Linear(3, 3),
        pred_proj=nn.Linear(3, 3),
    )


def test_encode_uses_pre_layernorm_cls_tokens():
    model = make_model()
    info = {'pixels': torch.zeros(2, 2, 3, 8, 8)}

    output = model.encode(info)

    expected = torch.arange(4 * 4 * 3).reshape(4, 4, 3)[:, 0]
    assert torch.equal(output['emb'], expected.reshape(2, 2, 3))
    assert 'patch_tokens' not in output


def test_encode_can_return_matching_pre_layernorm_patch_tokens():
    model = make_model()
    info = {'pixels': torch.zeros(1, 2, 3, 8, 8)}

    output = model.encode(info, return_patch_tokens=True)

    tokens = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    assert torch.equal(output['patch_tokens'], tokens[:, 1:].reshape(1, 2, 3, 3))
