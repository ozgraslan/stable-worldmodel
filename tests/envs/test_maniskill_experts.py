from types import SimpleNamespace

import pytest
import torch

from stable_worldmodel.envs.maniskill.experts import RGBExpert, StateExpert


@pytest.mark.parametrize('encoder', ['nature', 'spatial_softmax', 'flatten'])
def test_rgb_checkpoint_roundtrip(encoder):
    torch.set_num_threads(1)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(2,))
        )
    )
    obs = {'rgb': torch.randint(256, (2, 64, 64, 3), dtype=torch.uint8)}
    expert = RGBExpert(env, obs, encoder=encoder).eval()
    restored = RGBExpert(env, obs, encoder=encoder).eval()
    restored.load_state_dict(expert.state_dict(), strict=True)
    torch.testing.assert_close(
        expert.get_action(obs, deterministic=True),
        restored.get_action(obs, deterministic=True),
    )


def test_state_checkpoint_roundtrip():
    torch.set_num_threads(1)
    env = SimpleNamespace(
        single_action_space=SimpleNamespace(shape=(7,)),
        single_observation_space=SimpleNamespace(shape=(33,)),
    )
    expert = StateExpert(env).eval()
    restored = StateExpert(env).eval()
    restored.load_state_dict(expert.state_dict(), strict=True)
    obs = torch.randn(2, 33)
    torch.testing.assert_close(
        expert.get_action(obs, deterministic=True),
        restored.get_action(obs, deterministic=True),
    )
