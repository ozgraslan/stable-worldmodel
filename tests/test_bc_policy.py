"""State BC execution through the existing feed-forward policy."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel.data.normalization import ZScoreScaler
from stable_worldmodel.policy import FeedForwardPolicy


class ChunkModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.histories = []

    def get_action(self, info, horizon=1):
        state = info['state']
        self.histories.append(state.clone())
        actions = state[:, -1:, :] + torch.arange(horizon)[None, :, None]
        return actions[:, 0] if horizon == 1 else actions


def policy_and_model(n_envs=2, **kwargs):
    model = ChunkModel()
    policy = FeedForwardPolicy(
        model,
        history_len=3,
        action_chunk_size=5,
        history_keys=('state',),
        require_goal=False,
        **kwargs,
    )
    policy.set_env(
        SimpleNamespace(
            num_envs=n_envs,
            single_action_space=gym.spaces.Box(-1000.0, 1000.0, (1,)),
            action_space=gym.spaces.Box(-1000.0, 1000.0, (n_envs, 1)),
        )
    )
    return policy, model


def info(values, **kwargs):
    return {
        'state': np.asarray(values, dtype=np.float32)[:, None, None],
        **kwargs,
    }


def test_history_stride_and_sequential_chunk_execution():
    policy, model = policy_and_model()
    for step in range(11):
        action = policy.get_action(info([step, 100 + step]))
        np.testing.assert_allclose(action[:, 0], [step, 100 + step])
    assert len(model.histories) == 3
    torch.testing.assert_close(
        model.histories[0][0, :, 0], torch.tensor([0.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        model.histories[1][0, :, 0], torch.tensor([0.0, 0.0, 5.0])
    )
    torch.testing.assert_close(
        model.histories[2][0, :, 0], torch.tensor([0.0, 5.0, 10.0])
    )


def test_reset_flushes_only_affected_environment():
    policy, model = policy_and_model()
    policy.get_action(info([0, 100]))
    policy.get_action(info([1, 101]))
    raw = info([200, 102], _needs_flush=np.array([True, False]))
    action = policy.get_action(raw)
    np.testing.assert_allclose(action[:, 0], [200, 102])
    assert model.histories[-1].shape == (1, 3, 1)
    torch.testing.assert_close(
        model.histories[-1], torch.full((1, 3, 1), 200.0)
    )
    assert '_needs_flush' not in raw
    assert raw['state'].shape == (2, 1, 1)


def test_terminated_environment_is_not_predicted():
    policy, model = policy_and_model()
    actions = policy.get_action(
        info([0, 100], terminated=np.array([True, False]))
    )
    assert np.isnan(actions[0]).all()
    assert actions[1, 0] == 100
    assert model.histories[0].shape[0] == 1


def test_reuses_scalers_and_clips_after_denormalization():
    policy, model = policy_and_model(
        1,
        process={
            'state': ZScoreScaler(mean=[10.0], std=[2.0]),
            'action': ZScoreScaler(mean=[5.0], std=[3.0]),
        },
        clip_actions=True,
    )
    policy.env.action_space = gym.spaces.Box(-10.0, 10.0, (1, 1))
    action = policy.get_action(info([14]))
    torch.testing.assert_close(
        model.histories[0], torch.full((1, 3, 1), 2.0), check_dtype=False
    )
    assert action[0, 0] == 10  # normalized 2 -> raw 11 -> clipped 10


def test_history_stores_snapshots():
    policy, model = policy_and_model(1)
    raw = info([0])
    policy.get_action(raw)
    for step in range(1, 6):
        raw['state'][:] = step
        policy.get_action(raw)
    torch.testing.assert_close(
        model.histories[-1][0, :, 0], torch.tensor([0.0, 0.0, 5.0])
    )


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_rejects_invalid_history_length(value):
    with pytest.raises(ValueError, match='positive'):
        FeedForwardPolicy(ChunkModel(), history_len=value)
