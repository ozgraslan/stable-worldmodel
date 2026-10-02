"""Check XY orientation, endpoint scoring, and executed block alignment."""

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

    def predict(self, emb, act_emb):
        return emb + act_emb.reshape(*act_emb.shape[:-1], -1, 2).sum(-2)


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


class Dataset:
    column_names = ('action', 'obs', 'pixels')
    _schema_names = ('episode_idx', 'step_idx')

    def __init__(self):
        self.data = {
            'episode_idx': np.array([7] * 12 + [8] * 12),
            'step_idx': np.tile(np.arange(12), 2),
            'action': np.arange(48, dtype=np.float32).reshape(24, 2) / 100,
            'obs': np.arange(48, dtype=np.float32).reshape(24, 2),
            'pixels': np.zeros((24, 8, 8, 3), dtype=np.uint8),
        }

    def get_col_data(self, key):
        return self.data[key]

    def get_dim(self, key):
        return self.data[key].shape[-1]

    def get_row_data(self, rows):
        return {k: v[rows] for k, v in self.data.items()}


def config(tmp_path):
    return OmegaConf.create(
        {
            'seed': 42,
            'goal_source': 'dataset',
            'episode': 7,
            'start_step': 4,
            'history_len': 1,
            'action_block': 2,
            'horizon': 2,
            'img_size': 8,
            'policy': 'unused',
            'dataset_name': 'unused',
            'stats_dataset': None,
            'cache_dir': None,
            'device': 'cpu',
            'normalize_reps': False,
            'samples': 3,
            'grid_size': 0.075,
            'controller_scale': 0.1,
            'goal_offset_steps': 4,
            'play_in_reverse': False,
            'pose_column': 'obs',
            'plan_config': {
                'horizon': 2,
                'receding_horizon': 1,
                'action_block': 2,
                'history_len': 1,
            },
            'solver': {
                '_target_': 'stable_worldmodel.planning.solver.CEMSolver',
                'num_samples': 25,
                'n_steps': 2,
                'topk': 10,
                'var_scale': 1.0,
                'device': 'cpu',
                'seed': 42,
            },
            'chunk_size': 2,
            'output_dir': str(tmp_path),
        }
    )


def test_window_and_action_packing(tmp_path):
    ds = Dataset()
    cfg = config(tmp_path)
    info, _scaler, _, delta = evaluation.prepare_case(
        ds, ds, LinearDynamics(), cfg
    )
    assert info['action_history'].shape == (1, 0, 4)
    assert info['obs'].shape == (1, 1, 2)
    np.testing.assert_allclose(delta, ds.data['obs'][6] - ds.data['obs'][4])
    with pytest.raises(ValueError, match='contiguous'):
        evaluation.trajectory_rows(ds, 7, 10, 3, 2, 2)
    with pytest.raises(ValueError, match='contiguous'):
        evaluation.trajectory_rows(ds, 7, 0, 3, 2, 1)


@pytest.mark.parametrize('state', [True, False])
def test_complete_evaluation(tmp_path, monkeypatch, state):
    ds = Dataset()
    model = LinearDynamics(state)
    monkeypatch.setattr(evaluation, 'load_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(
        evaluation.swm.data, 'load_dataset', lambda *a, **kw: ds
    )
    result = evaluation.evaluate(config(tmp_path))
    assert (tmp_path / 'landscape.png').stat().st_size > 0
    assert (tmp_path / 'landscape_3d.png').stat().st_size > 0
    assert (tmp_path / 'landscape.pdf').stat().st_size > 0
    assert (tmp_path / 'landscape_3d.pdf').stat().st_size > 0
    if not state:
        assert (tmp_path / 'trajectory.png').stat().st_size > 0
    data = np.load(tmp_path / 'landscape.npz')
    assert data['energy'].shape == (3, 3)
    assert data['model_action_sequences'].shape == (9, 2, 4)
    assert data['cem_model_action_sequence'].shape == (2, 4)
    raw = data['cem_controller_action_sequence']
    np.testing.assert_allclose(data['cem_delta_sequence'], raw.sum(1) * 0.1)
    assert np.isfinite(data['energy']).all()
    expected_raw = ds.data['action'][4:8].reshape(2, 2, 2)
    np.testing.assert_allclose(
        data['ground_truth_controller_action_sequence'], expected_raw
    )
    np.testing.assert_allclose(
        data['ground_truth_action_total_delta'], expected_raw.sum((0, 1)) * 0.1
    )
    center = expected_raw.sum((0, 1)) * 0.1
    np.testing.assert_allclose(data['axis'].mean(1), center, atol=1e-7)
    np.testing.assert_allclose(data['axis'][:, 1], center, atol=1e-7)
    np.testing.assert_allclose(data['total_deltas'][4], center, atol=1e-7)
    np.testing.assert_allclose(result['grid_center_total_delta'], center)
    assert result['grid_center_source'] == 'recorded_action'
    np.testing.assert_allclose(
        data['axis'][:, -1] - data['axis'][:, 0], [0.3, 0.3]
    )
    from stable_worldmodel.data.normalization import get_scaler

    action_scaler = get_scaler('zscore').fit(ds.data['action'])
    expected_packed = action_scaler.transform(expected_raw).reshape(2, 4)
    np.testing.assert_allclose(
        data['ground_truth_model_action_sequence'], expected_packed
    )
    expected_gap = (
        result['ground_truth_action_energy'] - result['best_grid_energy']
    )
    assert result['ground_truth_action_minus_grid_min'] == pytest.approx(
        expected_gap
    )
    assert result['grid_actions_with_lower_energy'] == int(
        (data['energy'] < result['ground_truth_action_energy'] - 1e-6).sum()
    )
    assert json.loads((tmp_path / 'results.json').read_text()) == result


def test_native_state_checkpoint(tmp_path, monkeypatch):
    """Exercise export/load, native goal encoding, and LeWM's history rollout."""
    import hydra

    from stable_worldmodel.wm.utils import save_pretrained

    torch.manual_seed(0)
    model_config = {
        '_target_': 'stable_worldmodel.wm.lewm.StateLeWM',
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
    ds = Dataset()
    monkeypatch.setattr(
        evaluation.swm.data, 'load_dataset', lambda *a, **kw: ds
    )
    result = evaluation.evaluate(cfg)
    assert np.isfinite(result['cem_energy'])
    saved = np.load(tmp_path / 'result/landscape.npz')
    assert np.ptp(saved['energy']) > 0


@pytest.mark.parametrize('representation', ['jpeg', 'objects', 'dense'])
def test_rgb_row_representations(tmp_path, representation):
    import io

    from PIL import Image

    ds = Dataset()
    frames = np.full((24, 8, 8, 3), 128, dtype=np.uint8)
    if representation == 'jpeg':
        blobs = []
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format='JPEG')
            blobs.append(buffer.getvalue())
        ds.data['pixels'] = np.asarray(blobs, dtype=object)
    elif representation == 'objects':
        objects = np.empty(24, dtype=object)
        objects[:] = list(frames)
        ds.data['pixels'] = objects
    else:
        ds.data['pixels'] = frames
    info, _, preview, _ = evaluation.prepare_case(
        ds, ds, LinearDynamics(state=False), config(tmp_path)
    )
    assert preview.dtype == np.uint8
    assert preview.shape == (3, 8, 8, 3)
    assert np.all(preview == 128)
    expected = (
        128 / 255 - torch.tensor([0.485, 0.456, 0.406])
    ) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(info['pixels'][0, 0, :, 0, 0], expected)
    torch.testing.assert_close(info['goal'][0, 0, :, 0, 0], expected)


def test_rgb_lance_rows(tmp_path, monkeypatch):
    pytest.importorskip('lancedb')
    from stable_worldmodel.data.formats.lance import LanceWriter

    ds = Dataset()
    ds.data['pixels'][:] = 128
    path = tmp_path / 'rgb.lance'
    with LanceWriter(path) as writer:
        writer.write_episodes(
            [
                {
                    key: list(ds.data[key][:12])
                    for key in ('action', 'pixels', 'obs')
                }
            ]
        )
    cfg = config(tmp_path / 'result')
    cfg.episode = 0
    cfg.dataset_name = str(path)
    cfg.cache_dir = str(tmp_path)
    monkeypatch.setattr(
        evaluation, 'load_pretrained', lambda *a, **kw: LinearDynamics(False)
    )
    result = evaluation.evaluate(cfg)
    assert np.isfinite(result['cem_energy'])
    assert (tmp_path / 'result/landscape.png').stat().st_size > 0


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


def test_reverse_pose_reference(tmp_path):
    ds = Dataset()
    cfg = config(tmp_path)
    forward, _, _, delta = evaluation.prepare_case(
        ds, ds, LinearDynamics(), cfg
    )
    cfg.play_in_reverse = True
    backward, _, _, reverse_delta = evaluation.prepare_case(
        ds, ds, LinearDynamics(), cfg
    )
    torch.testing.assert_close(backward['obs'], forward['goal_obs'])
    torch.testing.assert_close(backward['goal_obs'], forward['obs'])
    np.testing.assert_allclose(reverse_delta, -delta)


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


def test_shared_cem_defaults():
    from pathlib import Path

    import hydra

    root = Path(__file__).resolve().parents[2]
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(root / 'scripts/plan/config')
    ):
        cfg = hydra.compose(config_name='energy_landscape_overhead')
        native = hydra.compose(config_name='maniskill_overhead')
    for key in ('_target_', 'num_samples', 'n_steps', 'topk', 'var_scale'):
        assert cfg.solver[key] == native.solver[key]
    assert cfg.plan_config.horizon == native.plan_config.horizon
    assert cfg.plan_config.action_block == native.plan_config.action_block
    assert cfg.samples == 41
    assert cfg.goal_source == 'simulator'
    assert cfg.goal_action_index is None
    assert cfg.goal_offset_steps == 5
    cfg.horizon = 3
    assert cfg.goal_offset_steps == 15
    cfg.action_block = 2
    assert cfg.goal_offset_steps == 6
    assert 'cem' not in cfg


def test_cem_matches_native_solver(tmp_path):
    import hydra
    from gymnasium.spaces import Box

    from stable_worldmodel.data.normalization import ZScoreScaler
    from stable_worldmodel.planning import ShootingCostEvaluator
    from stable_worldmodel.policy import PlanConfig

    cfg = config(tmp_path)
    model = LinearDynamics()
    info = {
        'obs': torch.tensor([[[0.1, 0.3]]]),
        'goal_obs': torch.tensor([[[0.8, -0.5]]]),
    }
    scaler = ZScoreScaler(mean=[[0.2, -0.1]], std=[[0.4, 0.3]])
    objective = EndpointEnergy()
    actual, raw, delta, _ = evaluation.plan_with_cem(
        model, info, scaler, cfg, objective
    )
    solver = hydra.utils.instantiate(
        cfg.solver, cost=ShootingCostEvaluator(model, objective)
    )
    solver.configure(
        action_space=Box(-1.0, 1.0, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(**OmegaConf.to_container(cfg.plan_config)),
    )
    expected = solver.solve(info)['actions'][0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        raw, scaler.inverse_transform(expected.reshape(2, 2, 2))
    )
    torch.testing.assert_close(delta, raw.sum(1) * cfg.controller_scale)


def test_recorded_actions_reverse_and_episode_boundary(tmp_path):
    from stable_worldmodel.data.normalization import ZScoreScaler

    ds = Dataset()
    cfg = config(tmp_path)
    cfg.play_in_reverse = True
    assert evaluation.recorded_action_sequence(ds, ZScoreScaler(), cfg) == (
        None,
        None,
    )
    cfg.play_in_reverse = False
    cfg.start_step = 10
    with pytest.raises(ValueError, match='contiguous'):
        evaluation.recorded_action_sequence(ds, ZScoreScaler(), cfg)


def test_recorded_action_energy_against_goal(tmp_path):
    from stable_worldmodel.data.normalization import get_scaler

    ds = Dataset()
    cfg = config(tmp_path)
    model = LinearDynamics()
    info, action_scaler, _, _ = evaluation.prepare_case(ds, ds, model, cfg)
    actions, _ = evaluation.recorded_action_sequence(ds, action_scaler, cfg)
    actual = evaluate_action_sequences(model, info, actions.unsqueeze(0))[
        0
    ].item()
    state_scaler = get_scaler('zscore').fit(ds.data['obs'])
    start, goal = state_scaler.transform(ds.data['obs'][[4, 8]])
    increments = action_scaler.transform(ds.data['action'][4:8]).sum(0)
    expected = np.abs(start + increments - goal).mean()
    assert actual == pytest.approx(expected)


class GoalSimulator:
    """Deterministic simulator double with observable action execution."""

    def __init__(self, truncate=False):
        self.actions = []
        self.closed = False
        self.truncate = truncate

    def reset(self, *, seed):
        self.seed = seed
        self.position = np.array([0.4, -0.2], dtype=np.float32)
        return {'obs': self.position.copy()}, {}

    def step(self, action):
        self.actions.append(action.copy())
        self.position += action * 0.1
        return {'obs': self.position.copy()}, 0, False, self.truncate, {}

    def render(self):
        return np.full((8, 8, 3), 30 + len(self.actions), dtype=np.uint8)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('state', [True, False])
def test_simulator_goal_is_selected_grid_endpoint(
    tmp_path, monkeypatch, state
):
    ds = Dataset()
    model = LinearDynamics(state)
    env = GoalSimulator()
    cfg = config(tmp_path)
    cfg.goal_source = 'simulator'
    cfg.goal_action_index = 7
    monkeypatch.setattr(evaluation, 'make_goal_environment', lambda cfg: env)
    monkeypatch.setattr(evaluation, 'load_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(
        evaluation.swm.data, 'load_dataset', lambda *a, **kw: ds
    )

    def no_expert_rows(*args, **kwargs):
        pytest.fail('Simulator goals must not read expert trajectory rows')

    monkeypatch.setattr(ds, 'get_row_data', no_expert_rows)
    result = evaluation.evaluate(cfg)
    assert env.closed
    assert env.seed == cfg.seed
    assert len(env.actions) == cfg.horizon * cfg.action_block
    expected_command = np.array([0, 0.375], dtype=np.float32)
    np.testing.assert_allclose(env.actions, np.tile(expected_command, (4, 1)))
    data = np.load(tmp_path / 'landscape.npz')
    np.testing.assert_allclose(
        data['ground_truth_model_action_sequence'],
        data['model_action_sequences'][7],
    )
    np.testing.assert_allclose(
        data['ground_truth_action_total_delta'], data['total_deltas'][7]
    )
    assert result['ground_truth_action_energy'] == pytest.approx(
        data['energy'].ravel()[7], abs=1e-6
    )
    assert result['goal_source'] == 'simulator'
    assert result['goal_action_index'] == 7
    assert result['reference_action_label'] == 'Executed grid action'
    assert result['context_step'] == 0
    assert result['goal_step'] == 4
    assert data['clip_frames'][-1, 0, 0, 0] == 34
    assert (tmp_path / 'goal.png').is_file()
    assert (tmp_path / 'start.png').is_file()


@pytest.mark.parametrize('state', [True, False])
def test_simulator_goal_preprocessing(tmp_path, monkeypatch, state):
    ds, model = Dataset(), LinearDynamics(state)
    cfg = config(tmp_path)
    cfg.goal_action_index = 7
    env = GoalSimulator()
    monkeypatch.setattr(evaluation, 'make_goal_environment', lambda cfg: env)
    _, grid = xy_action_grid(3, -0.075, 0.075)
    info, _, _, _, _, _, _ = evaluation.prepare_simulator_case(
        ds, model, cfg, grid
    )
    if state:
        from stable_worldmodel.data.normalization import get_scaler

        scaler = get_scaler('zscore').fit(ds.data['obs'])
        np.testing.assert_allclose(
            info['goal_obs'][0, 0],
            scaler.transform(env.position).reshape(-1),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            info['obs'][0, 0],
            scaler.transform(np.array([0.4, -0.2], dtype=np.float32)).reshape(
                -1
            ),
        )
    else:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(
            info['goal'][0, 0, :, 0, 0], (34 / 255 - mean) / std
        )
        torch.testing.assert_close(
            info['pixels'][0, 0, :, 0, 0], (30 / 255 - mean) / std
        )


def test_simulator_goal_validation_and_cleanup(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    ds, model = Dataset(), LinearDynamics()
    _, grid = xy_action_grid(3, -0.075, 0.075)
    env = GoalSimulator(truncate=True)
    monkeypatch.setattr(evaluation, 'make_goal_environment', lambda cfg: env)
    cfg.goal_action_index = 7
    with pytest.raises(RuntimeError, match='before the goal horizon'):
        evaluation.prepare_simulator_case(ds, model, cfg, grid)
    assert env.closed
    cfg.goal_action_index = 9
    with pytest.raises(ValueError, match='must index'):
        evaluation.prepare_simulator_case(ds, model, cfg, grid)
    cfg.play_in_reverse = True
    with pytest.raises(ValueError, match='forward playback'):
        evaluation.prepare_simulator_case(ds, model, cfg, grid)


def test_simulator_goal_seeded_selection(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.goal_action_index = None
    _, grid = xy_action_grid(3, -0.075, 0.075)
    environments = []

    def make_env(cfg):
        env = GoalSimulator()
        environments.append(env)
        return env

    monkeypatch.setattr(evaluation, 'make_goal_environment', make_env)
    results = [
        evaluation.prepare_simulator_case(
            Dataset(), LinearDynamics(), cfg, grid
        )
        for _ in range(2)
    ]
    expected = int(np.random.default_rng(cfg.seed).integers(len(grid)))
    assert results[0][-1] == results[1][-1] == expected
    np.testing.assert_array_equal(
        environments[0].actions, environments[1].actions
    )
    torch.testing.assert_close(
        results[0][0]['goal_obs'], results[1][0]['goal_obs']
    )
