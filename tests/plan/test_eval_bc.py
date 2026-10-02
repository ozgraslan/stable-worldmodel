"""State BC checkpoints use the native eval_bc / World evaluation path."""

import json
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import pytest
import torch

import stable_worldmodel as swm
from scripts.plan import eval_bc
from stable_worldmodel.policy import FeedForwardPolicy
from stable_worldmodel.wm.bc import StateBC
from stable_worldmodel.wm.utils import save_pretrained

ROOT = Path(__file__).resolve().parents[2]


class ToyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.observation_space = gym.spaces.Dict(
            {'state': gym.spaces.Box(-np.inf, np.inf, (1,))}
        )
        self.actions = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return {'state': np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        self.actions.append(float(action[0]))
        self.steps += 1
        return (
            {'state': np.array([self.steps], dtype=np.float32)},
            0.0,
            self.steps == 3,
            False,
            {},
        )

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def export(tmp_path, architecture='mlp'):
    cfg = {
        '_target_': 'stable_worldmodel.wm.bc.StateBC',
        'state_dim': 1,
        'action_dim': 1,
        'history_size': 3,
        'frameskip': 5,
        'hidden_dims': [],
    }
    options = {'architecture': architecture}
    if architecture == 'transformer':
        options['transformer'] = {
            'embed_dim': 16,
            'depth': 1,
            'heads': 2,
            'dim_head': 8,
            'mlp_dim': 32,
            'dropout': 0.0,
        }
    cfg.update(options)
    model = StateBC(1, 1, hidden_dims=(), **options)
    head = (
        model.network.action_head
        if architecture == 'transformer'
        else model.network[0]
    )
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.arange(5) * 0.2)
    save_pretrained(model, 'bc', config=cfg, cache_dir=str(tmp_path))
    directory = tmp_path / 'checkpoints/bc'
    (directory / 'normalization.json').write_text(
        json.dumps(
            {
                'state': {'mean': [0.0], 'std': [1.0], 'eps': 1e-8},
                'action': {'mean': [0.0], 'std': [1.0], 'eps': 1e-8},
            }
        )
    )
    return directory


def config():
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / 'scripts/plan/config')
    ):
        return hydra.compose(
            config_name='maniskill_bc_overhead',
            overrides=[
                'policy=bc/weights.pt',
                'device=cpu',
                'eval.num_eval=3',
                'world.num_envs=1',
                'eval.save_video=false',
                'eval.eval_budget=5',
                'subdir=bc_test',
            ],
        )


@pytest.mark.parametrize('architecture', ['mlp', 'transformer'])
def test_native_eval_uses_saved_stats_and_resets(
    monkeypatch, tmp_path, architecture
):
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    directory = export(tmp_path, architecture)
    env = ToyEnv()
    monkeypatch.setattr(gym, 'make', lambda *a, **kw: env)
    # Env-driven BC evaluation must work with only checkpoint artifacts.
    monkeypatch.setattr(
        eval_bc,
        'get_dataset',
        lambda *a: pytest.fail('Unexpected dataset load'),
    )
    cfg = config()
    assert cfg.world.expert_checkpoint is None
    assert cfg.world.state_goal is False
    metrics = eval_bc.run.__wrapped__(cfg)
    assert metrics['success_rate'] == 100
    np.testing.assert_allclose(env.actions, [0.0, 0.2, 0.4] * 3, atol=1e-6)
    assert env.closed
    assert (directory / 'evals/bc_test/bc_results.txt').exists()


def test_explicit_world_reset_flushes_queued_actions(monkeypatch):
    env = ToyEnv()
    monkeypatch.setattr(gym, 'make', lambda *a, **kw: env)
    model = StateBC(1, 1, hidden_dims=())
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.copy_(torch.arange(5) * 0.2)
    world = swm.World('unused', num_envs=1, image_shape=(8, 8))
    world.set_policy(
        FeedForwardPolicy(
            model,
            history_len=3,
            action_chunk_size=5,
            history_keys=('state',),
            require_goal=False,
        )
    )
    try:
        world.evaluate(episodes=1, seed=1)
        world.evaluate(episodes=1, seed=2)
        np.testing.assert_allclose(env.actions, [0.0, 0.2, 0.4] * 2, atol=1e-6)
    finally:
        world.close()


def test_missing_normalization_fails_without_refitting(monkeypatch, tmp_path):
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    directory = export(tmp_path)
    (directory / 'normalization.json').unlink()
    with pytest.raises(FileNotFoundError, match='normalization.json'):
        eval_bc.make_policy(config())
