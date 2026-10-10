"""Expert pair sampling, normalization-only data, and action-grid contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.plan import eval_energy_landscape as evaluation
from stable_worldmodel.planning.energy_landscape import (
    EndpointEnergy,
    evaluate_action_sequences,
    xy_action_grid,
)
from stable_worldmodel.wm.lewm.lewm import LeWM


class LinearDynamics(LeWM):
    def __init__(self, state=True):
        nn.Module.__init__(self)
        self.predictor = SimpleNamespace(num_frames=3)
        self.action_encoder = nn.Identity()
        self.weight = nn.Parameter(torch.ones(1))
        self.state_columns = ['obs'] if state else []

    def observation(self, info):
        if self.state_columns:
            return info['obs']
        return info['pixels'].mean((-2, -1))[..., :2]

    def encode(self, info):
        info['emb'] = self.observation(info)
        return info

    def predict(self, info):
        act_emb = info['act_emb']
        info['preds'] = info['emb'] + act_emb.reshape(
            *act_emb.shape[:-1], -1, 2
        ).sum(-2)
        return info


def test_chunking_and_minimum():
    axis, grid = xy_action_grid(3, -1.0, 1.0)
    torch.testing.assert_close(grid[0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(grid[2], torch.tensor([1.0, -1.0]))
    model = LinearDynamics()
    info = {
        'obs': torch.zeros(1, 1, 2),
        'goal_obs': torch.tensor([[[1.0, -1.0]]]),
    }
    a = evaluate_action_sequences(model, info, grid[:, None], chunk=2)
    b = evaluate_action_sequences(model, info, grid[:, None], chunk=20)
    torch.testing.assert_close(a, b)
    assert a.argmin() == 2
    assert a[2] == 0
    assert 'goal_emb' not in info
    assert axis.tolist() == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize('metric', ['l1', 'mse'])
def test_endpoint_energy(metric):
    info = {
        'predicted_emb': torch.tensor([[[[99.0, 99.0], [2.0, 4.0]]]]),
        'goal_emb': torch.tensor([[[0.0, 0.0]]]),
    }
    expected = 3.0 if metric == 'l1' else 10.0
    assert EndpointEnergy(metric)(info).item() == expected
    normalized = EndpointEnergy(metric, normalize=True)(info)
    assert torch.isfinite(normalized).all()


class StatsOnly:
    """Statistics source that rejects every attempt to read trajectory rows."""

    def __init__(self):
        self.data = {
            'action': np.array([[-1, -1], [1, 1]], dtype=np.float32),
            'obs': np.array([[-1, -1], [1, 1]], dtype=np.float32),
        }

    def get_col_data(self, key):
        return self.data[key]

    def get_row_data(self, rows):
        pytest.fail('Start and goal must never be read from dataset rows')


class ExpertEnvironment:
    """Successful expert case restored to a nonzero trajectory start."""

    def __init__(self, *, success=True, goal_index=4):
        self.closed = False
        self.success = success
        self.expert_goal_index = goal_index
        self.reset_calls = []
        self.comparisons = []
        self.expert_actions = np.array(
            [
                [0.1, 0.2],
                [0.3, -0.2],
                [-0.4, 0.1],
                [0.2, 0.3],
                [0.8, -0.9],
                [-0.7, 0.6],
            ],
            dtype=np.float32,
        )
        self.expert_frames = np.stack(
            [
                np.full((8, 8, 3), 30 + 7 * step, dtype=np.uint8)
                for step in range(len(self.expert_actions) + 1)
            ]
        )
        self.start = np.array([0.4, -0.2], dtype=np.float32)
        self.goal_observation = {
            'obs': self.start + self.expert_actions[:goal_index].sum(0) * 0.1
        }

    def reset(self, *, seed, options):
        self.reset_calls.append((seed, options))
        return {'obs': self.start.copy()}, {
            'goal': self.expert_frames[self.expert_goal_index],
            'expert_start_step': 7,
            'expert_seed': 123,
            'expert_task_success': self.success,
            'expert_rollout_attempts': 2,
            'goal_step_distance': self.expert_goal_index,
        }

    def planning_action_comparison(self, sequences, output):
        self.comparisons.append((sequences, output))
        return {label: {'final_goal_success': True} for label in sequences}

    def close(self):
        self.closed = True


def config(tmp_path):
    return OmegaConf.create(
        {
            'seed': 42,
            'expert_checkpoint': 'unused-expert.pt',
            'expert_policy_type': 'rgb',
            'expert_encoder': 'spatial_softmax',
            'expert_max_steps': 100,
            'expert_max_attempts': 20,
            'expert_min_object_displacement': 0.02,
            'expert_min_eef_displacement': 0.02,
            'start_from_beginning': False,
            'use_expert_action_mean': False,
            'sim_backend': 'physx_cpu',
            'action_block': 2,
            'horizon': 2,
            'img_size': 8,
            'policy': 'unused',
            'dataset_name': 'normalization-only',
            'cache_dir': None,
            'device': 'cpu',
            'normalize_reps': False,
            'samples': 3,
            'grid_size': 0.075,
            'controller_scale': 0.1,
            'goal_offset_steps': 4,
            'pose_column': 'obs',
            'chunk_size': 2,
            'output_dir': str(tmp_path),
        }
    )


def mock_inputs(monkeypatch, model, stats=None, env=None):
    stats = StatsOnly() if stats is None else stats
    env = ExpertEnvironment() if env is None else env
    loads = []

    def load_stats(name, **kwargs):
        assert name == 'normalization-only'
        assert 'pixels' not in kwargs['keys_to_load']
        loads.append(kwargs)
        return stats

    monkeypatch.setattr(evaluation.swm.data, 'load_dataset', load_stats)
    monkeypatch.setattr(evaluation, 'make_goal_environment', lambda cfg: env)
    if model is not None:
        monkeypatch.setattr(
            evaluation, 'load_pretrained', lambda *a, **kw: model
        )
    return env, stats, loads


@pytest.mark.parametrize('state', [True, False])
@pytest.mark.parametrize('expert_mean', [False, True])
def test_complete_expert_evaluation(tmp_path, monkeypatch, state, expert_mean):
    model = LinearDynamics(state)
    env, _stats, loads = mock_inputs(monkeypatch, model)
    cfg = config(tmp_path)
    cfg.use_expert_action_mean = expert_mean
    result = evaluation.evaluate(cfg)
    assert env.closed
    assert env.reset_calls == [(42, {'start_from_beginning': False})]
    assert len(loads) == 1
    assert loads[0]['keys_to_load'] == (
        ['action', 'obs'] if state else ['action']
    )
    for filename in (
        'landscape.png',
        'landscape.pdf',
        'landscape_3d.png',
        'landscape_3d.pdf',
        'trajectory.png',
        'start.png',
        'goal.png',
    ):
        assert (tmp_path / filename).stat().st_size > 0
    assert json.loads((tmp_path / 'results.json').read_text()) == result
    assert result['goal_source'] == 'expert'
    assert result['context_step'] == 7
    assert result['goal_step'] == 11
    assert result['expert']['expert_seed'] == 123
    assert result['reference_action_label'] == 'Expert actions'
    assert result['comparison_video'] == 'best_action/env_0.mp4'
    assert set(result['comparison_results']) == {
        'Grid minimum',
        'Expert actions',
    }
    assert not any(key.startswith('cem_') for key in result)

    with np.load(tmp_path / 'landscape.npz') as data:
        assert not any(key.startswith('cem_') for key in data.files)
        raw = env.expert_actions[:4].reshape(2, 2, 2)
        np.testing.assert_array_equal(
            data['ground_truth_controller_action_sequence'], raw
        )
        np.testing.assert_allclose(
            data['ground_truth_action_total_delta'], raw.sum((0, 1)) * 0.1
        )
        np.testing.assert_array_equal(
            data['clip_frames'], env.expert_frames[[0, 2, 4]]
        )
        assert data['model_action_sequences'].shape == (9, 2, 4)
        assert data['controller_action_sequences'].shape == (9, 2, 2, 2)
        np.testing.assert_allclose(
            data['total_deltas'],
            data['controller_action_sequences'].sum((1, 2)) * 0.1,
        )
        assert np.isfinite(data['energy']).all()
        expected_center = raw if expert_mean else np.zeros_like(raw)
        np.testing.assert_array_equal(
            data['controller_action_sequences'][4], expected_center
        )
        np.testing.assert_allclose(
            result['grid_center_total_delta'],
            expected_center.sum((0, 1)) * 0.1,
        )
        if expert_mean:
            assert result['ground_truth_action_energy'] == pytest.approx(
                data['energy'].ravel()[4]
            )
        best = data['energy'].argmin()
        sequences = env.comparisons[0][0]
        assert set(sequences) == {'Grid minimum', 'Expert actions'}
        np.testing.assert_array_equal(
            sequences['Expert actions'], env.expert_actions[:4]
        )
        sequence = sequences['Grid minimum']
        np.testing.assert_array_equal(
            sequence, data['controller_action_sequences'][best].reshape(-1, 2)
        )
        assert env.comparisons[0][1] == tmp_path / 'best_action'


@pytest.mark.parametrize('state', [True, False])
def test_expert_endpoints_use_training_normalization(tmp_path, state):
    cfg = config(tmp_path)
    stats = StatsOnly()
    stats.data['action'] = np.array(
        [[0.1, -0.2], [0.9, 0.4]], dtype=np.float32
    )
    stats.data['obs'] = np.array([[0, 0], [2, 4]], dtype=np.float32)
    env = ExpertEnvironment()
    info, scaler, preview, pose_delta, raw, _metadata = (
        evaluation.prepare_expert_case(
            env,
            stats,
            LinearDynamics(state),
            cfg,
        )
    )
    assert info['action_history'].shape == (1, 0, 4)
    np.testing.assert_allclose(scaler.mean, [[0.5, 0.1]], atol=1e-7)
    np.testing.assert_array_equal(raw.flatten(0, 1), env.expert_actions[:4])
    np.testing.assert_array_equal(preview, env.expert_frames[[0, 2, 4]])
    np.testing.assert_allclose(
        pose_delta, env.goal_observation['obs'] - env.start
    )
    if state:
        np.testing.assert_allclose(
            info['obs'][0, 0], (env.start - [1, 2]) / [1, 2]
        )
        np.testing.assert_allclose(
            info['goal_obs'][0, 0],
            (env.goal_observation['obs'] - [1, 2]) / [1, 2],
        )
    else:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(
            info['pixels'][0, 0, :, 0, 0], (30 / 255 - mean) / std
        )
        torch.testing.assert_close(
            info['goal'][0, 0, :, 0, 0], (58 / 255 - mean) / std
        )


def test_grid_preserves_expert_sequence_at_controller_bounds(tmp_path):
    from stable_worldmodel.data.normalization import ZScoreScaler

    cfg = config(tmp_path)
    cfg.use_expert_action_mean = True
    raw = torch.tensor(
        [[[1.0, -1.0], [-1.0, 1.0]], [[0.2, -0.3], [-0.4, 0.5]]]
    )
    scaler = ZScoreScaler(mean=[[0.2, -0.1]], std=[[0.4, 0.3]])
    axes, commands, candidates, clipped = evaluation.build_action_grid(
        cfg, scaler, raw
    )
    torch.testing.assert_close(commands[4], raw)
    assert clipped > 0
    assert (commands.abs() <= 1).all()
    recovered = scaler.inverse_transform(candidates.reshape(9, 2, 2, 2))
    torch.testing.assert_close(recovered, commands)
    totals = commands.sum((1, 2)) * cfg.controller_scale
    np.testing.assert_allclose(axes[0], totals[:3, 0])
    np.testing.assert_allclose(axes[1], totals[::3, 1])


@pytest.mark.parametrize(
    'success,goal_index,message',
    [
        (False, 4, 'did not solve'),
        (True, 3, 'pair spans 3 steps'),
    ],
)
def test_invalid_expert_case_closes_environment(
    tmp_path, monkeypatch, success, goal_index, message
):
    env = ExpertEnvironment(success=success, goal_index=goal_index)
    mock_inputs(monkeypatch, LinearDynamics(), env=env)
    with pytest.raises((RuntimeError, ValueError), match=message):
        evaluation.evaluate(config(tmp_path))
    assert env.closed
    assert not env.comparisons


def test_expert_pair_requires_matching_offset(tmp_path):
    cfg = config(tmp_path)
    cfg.goal_offset_steps = 3
    env = ExpertEnvironment()
    with pytest.raises(ValueError, match=r'horizon \* action_block'):
        evaluation.prepare_expert_case(env, StatsOnly(), LinearDynamics(), cfg)
    assert not env.reset_calls


@pytest.mark.parametrize('invalid', [float('nan'), 2.0])
def test_invalid_expert_actions_are_rejected(tmp_path, invalid):
    env = ExpertEnvironment()
    env.expert_actions[0, 0] = invalid
    with pytest.raises(ValueError, match='finite|bounds'):
        evaluation.prepare_expert_case(
            env, StatsOnly(), LinearDynamics(), config(tmp_path)
        )


def test_native_state_checkpoint(tmp_path, monkeypatch):
    """Exercise export/load, native goal encoding, and LeWM's history rollout."""
    import hydra

    from stable_worldmodel.wm.utils import save_pretrained

    torch.manual_seed(0)
    model_config = {
        '_target_': 'stable_worldmodel.wm.lewm.LeWMState',
        'state_columns': ['obs'],
        'encoder': {
            '_target_': 'torch.nn.Linear',
            'in_features': 2,
            'out_features': 8,
        },
        'action_encoder': {
            '_target_': 'torch.nn.Linear',
            'in_features': 4,
            'out_features': 8,
        },
        'predictor': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Predictor',
            'num_frames': 3,
            'depth': 1,
            'heads': 2,
            'mlp_dim': 16,
            'input_dim': 8,
            'hidden_dim': 8,
            'dim_head': 4,
        },
    }
    model = hydra.utils.instantiate(model_config)
    for module in model.modules():
        if hasattr(module, 'adaLN_modulation'):
            nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.02)
    save_pretrained(model, 'native', config=model_config, cache_dir=tmp_path)
    cfg = config(tmp_path / 'result')
    cfg.normalize_reps = True
    cfg.policy = str(tmp_path / 'checkpoints/native/weights.pt')
    cfg.cache_dir = str(tmp_path)
    stats = StatsOnly()
    mock_inputs(monkeypatch, model=None, stats=stats)
    result = evaluation.evaluate(cfg)
    assert np.isfinite(result['ground_truth_action_energy'])
    saved = np.load(tmp_path / 'result/landscape.npz')
    assert np.ptp(saved['energy']) > 0


@pytest.mark.parametrize('normalize', [True, False])
def test_reference_normalization_feedback(normalize):
    """Compare with the notebook's encode/norm/predict/norm loop."""
    from torch.nn import functional as F

    from stable_worldmodel.planning.energy_landscape import NotebookDynamics

    native = LinearDynamics()
    model = NotebookDynamics(native, normalize=normalize)
    info = {
        'obs': torch.tensor([[[2.0, 5.0]]]),
        'goal_obs': torch.tensor([[[4.0, 1.0]]]),
    }
    actions = torch.tensor(
        [[[2.5, 0.0], [-0.3, 0.1]], [[0.4, -0.2], [0.8, 0.1]]]
    )
    actual = evaluate_action_sequences(model, info, actions, chunk=1)
    context = info['obs'][:, 0].expand(2, -1)
    target = info['goal_obs'][:, 0]
    if normalize:
        context = F.layer_norm(context, (2,))
        target = F.layer_norm(target, (2,))
    for step in range(2):
        context = context + actions[:, step]
        if normalize:
            context = F.layer_norm(context, (2,))
    expected = (context - target).abs().mean(-1)
    torch.testing.assert_close(actual, expected)
    if normalize:
        # Scoring-only normalization is deliberately different.
        old = evaluate_action_sequences(
            native, info, actions, EndpointEnergy(normalize=True)
        )
        assert not torch.allclose(actual, old)
    else:
        # Disabled adapter must preserve native model rollout and energy.
        original = evaluate_action_sequences(native, info, actions)
        torch.testing.assert_close(actual, original)


def test_delta_chunk_conservation():
    from stable_worldmodel.data.normalization import ZScoreScaler
    from stable_worldmodel.planning.energy_landscape import (
        pack_delta_sequences,
    )

    scaler = ZScoreScaler(mean=[[0.2, -0.1]], std=[[0.4, 0.3]])
    deltas = torch.tensor([[[0.075, -0.05], [-0.02, 0.01]]])
    packed = pack_delta_sequences(deltas, scaler, 5, 0.1)
    raw = scaler.inverse_transform(packed.reshape(1, 2, 5, 2))
    torch.testing.assert_close(raw.sum(-2) * 0.1, deltas)
    torch.testing.assert_close(raw[:, :, 0], raw[:, :, -1])


def test_replot_preserves_values_without_model(tmp_path, monkeypatch):
    axis = np.linspace(-0.075, 0.075, 5)
    energy = np.arange(25, dtype=np.float32).reshape(5, 5) / 25
    histogram = energy.copy()
    source = tmp_path / 'landscape.npz'
    np.savez(
        source,
        axis=axis,
        energy=energy,
        histogram=histogram,
        xedges=axis,
        yedges=axis,
        ground_truth_delta=[0.01, -0.02],
        ground_truth_action_total_delta=[0.02, -0.03],
        ground_truth_action_energy=0.123,
    )
    original = source.read_bytes()
    calls = []

    def capture(axis_value, energy_value, histogram_value, *args):
        np.testing.assert_array_equal(axis_value, axis)
        np.testing.assert_array_equal(energy_value, energy)
        np.testing.assert_array_equal(histogram_value, histogram)
        assert args[3]['ground_truth_action_energy'] == pytest.approx(0.123)
        np.testing.assert_allclose(
            args[3]['ground_truth_action_total_delta'], [0.02, -0.03]
        )
        calls.append(True)

    def no_model(*args, **kwargs):
        pytest.fail('Saved-data rendering must not load a checkpoint')

    monkeypatch.setattr(evaluation, 'plot_landscape', capture)
    monkeypatch.setattr(evaluation, 'load_pretrained', no_model)
    cfg = OmegaConf.create(
        {'landscape_file': str(source), 'output_dir': str(tmp_path / 'plots')}
    )
    evaluation.run.__wrapped__(cfg)
    assert calls == [True]
    assert source.read_bytes() == original


def test_expert_config_defaults():
    from pathlib import Path

    import hydra

    root = Path(__file__).resolve().parents[2]
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(root / 'scripts/plan/config')
    ):
        cfg = hydra.compose(config_name='energy_landscape_overhead')
        native = hydra.compose(config_name='maniskill_overhead')
    assert cfg.expert_policy_type == native.world.expert_policy_type
    assert cfg.expert_max_attempts == native.world.expert_max_attempts
    assert not cfg.start_from_beginning
    assert not cfg.use_expert_action_mean
    assert 'solver' not in cfg
    assert 'plan_config' not in cfg
    assert 'goal_source' not in cfg
    assert 'goal_action_index' not in cfg
    cfg.horizon = 3
    assert cfg.goal_offset_steps == 15


def test_expert_environment_uses_shared_adapter(tmp_path, monkeypatch):
    pytest.importorskip('mani_skill')
    from stable_worldmodel.envs.maniskill import pusht

    cfg = config(tmp_path)
    env = ExpertEnvironment()
    settings = {}

    def create(**kwargs):
        settings.update(kwargs)
        return env

    monkeypatch.setattr(pusht, 'PushTSWMEnv', create)
    assert evaluation.make_goal_environment(cfg) is env
    assert settings['env_id'] == 'OverheadPushT-v1'
    assert settings['control_mode'] == 'pd_ee_delta_xy'
    assert settings['camera_name'] == 'overhead_camera'
    assert settings['expert_checkpoint'] == cfg.expert_checkpoint
    assert settings['expert_policy_type'] == cfg.expert_policy_type
    assert settings['expert_max_attempts'] == cfg.expert_max_attempts
    assert settings['goal_step_distance'] == cfg.horizon * cfg.action_block
    assert settings['state_goal']
