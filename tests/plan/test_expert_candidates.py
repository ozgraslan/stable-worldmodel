"""Expert injection preserves batching and chooses by model cost."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts.plan.eval_wm_expert_candidates import (
    ExpertCandidateCEM,
    expert_sequences,
)


class QuadraticCost:
    def get_cost(self, info, candidates):
        return (candidates - info['target']).square().sum((2, 3))


def make_solver(cost, expert):
    solver = ExpertCandidateCEM(
        cost,
        batch_size=2,
        num_samples=4,
        n_steps=2,
        topk=2,
    )
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(len(expert), 1)),
        n_envs=len(expert),
        config=SimpleNamespace(horizon=2, action_block=1),
    )
    solver.set_expert(expert)
    return solver


def test_expert_selected_across_solver_batches():
    expert = torch.tensor([0.7, -0.8, 0.3])[:, None, None].expand(-1, 2, 1)
    solver = make_solver(QuadraticCost(), expert)
    outputs = solver({'target': expert})
    torch.testing.assert_close(outputs['actions'], expert)
    assert solver.diagnostics['expert_selection_rate'] == 1
    records = solver.diagnostics['iterations']
    assert len(records) == 6
    assert all(row['expert_rank'] == 1 for row in records)
    assert all(row['num_candidates'] == 4 for row in records)
    assert all(
        row['expert_rank'] == 1 and row['num_candidates'] == 5
        for row in solver.diagnostics['selections']
    )
    parameters = solver.diagnostics['planning_parameters']
    assert parameters['solver']['num_samples'] == 4
    assert parameters['solver']['n_steps'] == 2
    assert parameters['solver']['topk'] == 2
    assert parameters['horizon_env_steps'] == 2
    assert {row['env_index'] for row in records} == {0, 1, 2}
    for row in solver.diagnostics['selections']:
        index = row['env_index']
        assert row['initial_expert_cost'] == 0
        assert row['initial_expert_rank'] == 1
        assert row['last_iteration_expert_rank'] == 1
        assert row['initial_cem_mean_cost'] == pytest.approx(
            float(expert[index].square().sum())
        )
    assert outputs['costs'] == [0, 0, 0]


def test_expert_is_not_forced_when_model_prefers_other_actions():
    expert = torch.full((3, 2, 1), 100.0)
    solver = make_solver(QuadraticCost(), expert)
    solver({'target': torch.zeros_like(expert)})
    assert solver.diagnostics['expert_selection_rate'] == 0
    assert all(
        row['selected_cost'] < row['expert_cost']
        for row in solver.diagnostics['selections']
    )


def test_cost_restored_on_scoring_failure():
    class FailingCost:
        def get_cost(self, info, candidates):
            raise RuntimeError('failed scoring')

    cost = FailingCost()
    expert = torch.zeros(3, 2, 1)
    solver = make_solver(cost, expert)
    with pytest.raises(RuntimeError, match='failed scoring'):
        solver({'target': expert})
    assert solver.cost is cost


def test_expert_actions_normalized_then_blocked():
    class Dataset:
        def load_chunk(self, episodes, starts, ends):
            np.testing.assert_array_equal(episodes, [7])
            np.testing.assert_array_equal(starts, [3])
            np.testing.assert_array_equal(ends, [7])
            return [{'action': np.arange(8).reshape(4, 2)}]

    class Processor:
        def transform(self, actions):
            return (actions - 2) / 2

    expert = expert_sequences(
        Dataset(),
        {'episodes_idx': [7], 'start_steps': [3]},
        SimpleNamespace(plan_len=4, horizon=2),
        Processor(),
    )
    torch.testing.assert_close(
        expert,
        torch.tensor([[[-1, -0.5, 0, 0.5], [1, 1.5, 2, 2.5]]]),
    )


def fake_expert_envs():
    envs = [
        SimpleNamespace(
            expert_actions=np.arange(12, dtype=np.float32).reshape(6, 2),
            expert_goal_index=4,
            replay_report={'passed': True, 'pixel_max_error': 0},
        )
        for _ in range(2)
    ]
    return SimpleNamespace(
        envs=[SimpleNamespace(unwrapped=env) for env in envs],
        seeds=np.array([42, 43]),
        num_envs=2,
    )


def test_environment_expert_matches_goal_and_preserves_reset_seeds():
    from scripts.plan.eval_wm_expert_candidates import (
        environment_expert_sequences,
    )

    config = SimpleNamespace(plan_len=4, horizon=2)
    expert, metadata = environment_expert_sequences(
        fake_expert_envs(),
        config,
        None,
    )
    assert expert.shape == (2, 2, 4)
    torch.testing.assert_close(
        expert[0],
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )
    assert [row['seed'] for row in metadata] == [42, 43]
    assert all(row['expert_goal_index'] == 4 for row in metadata)


@pytest.mark.parametrize(
    ('attribute', 'value', 'message'),
    [
        ('expert_actions', None, 'world.expert_checkpoint'),
        ('expert_goal_index', 2, 'terminated early'),
        ('replay_report', {'passed': False}, 'replay failed'),
        ('expert_actions', np.zeros((2, 2)), 'requires 4'),
    ],
)
def test_invalid_environment_witness_rejected(attribute, value, message):
    from scripts.plan.eval_wm_expert_candidates import (
        environment_expert_sequences,
    )

    envs = fake_expert_envs()
    setattr(envs.envs[0].unwrapped, attribute, value)
    with pytest.raises(ValueError, match=message):
        environment_expert_sequences(
            envs,
            SimpleNamespace(plan_len=4, horizon=2),
            None,
        )


def test_environment_expert_refreshed_only_after_reset(monkeypatch):
    from collections import deque

    from scripts.plan.eval_wm_expert_candidates import (
        EnvironmentExpertPolicy,
    )
    from stable_worldmodel.policy import WorldModelPolicy

    captured = []
    solver = SimpleNamespace(set_expert=lambda expert: captured.append(expert))
    policy = EnvironmentExpertPolicy(
        solver,
        config=SimpleNamespace(plan_len=4, horizon=2),
    )
    policy.env = fake_expert_envs()
    policy._action_buffer = [deque([torch.ones(2)]) for _ in range(2)]
    monkeypatch.setattr(WorldModelPolicy, 'get_action', lambda *a, **k: None)
    policy.reset()
    assert all(not buffer for buffer in policy._action_buffer)
    policy.get_action({})
    policy.get_action({})
    assert len(captured) == 1
    policy.env.envs[0].unwrapped.expert_actions += 10
    policy.env.seeds[0] = 99
    policy.reset()
    policy.get_action({})
    assert len(captured) == 2
    torch.testing.assert_close(captured[1][0], captured[0][0] + 10)
    assert policy.expert_metadata[0]['seed'] == 99


def test_mpc_provider_refreshes_subset_and_keeps_all_replans():
    expert = torch.full((3, 2, 1), 0.7)
    solver = make_solver(QuadraticCost(), expert)
    solver.diagnostic_history = []
    calls = []

    def provider(indices):
        calls.append(indices)
        return expert[indices], [
            {'env_step': 5 * (len(calls) - 1)} for _ in indices
        ]

    solver.expert_provider = provider
    solver(
        {
            'target': expert[[2, 0]],
            '_expert_env_index': torch.tensor([[2], [0]]),
        }
    )
    solver(
        {
            'target': expert[[0]],
            '_expert_env_index': torch.tensor([[0]]),
        }
    )
    assert calls == [[2, 0], [0]]
    assert len(solver.diagnostic_history) == 2
    first, second = solver.diagnostic_history
    assert [row['env_index'] for row in first['selections']] == [2, 0]
    assert second['selections'][0]['env_index'] == 0
    assert second['selections'][0]['env_step'] == 5
    assert second['selections'][0]['replan_index'] == 1
    assert first['selections'][0]['replan_index'] == 0


def test_native_cem_return_rule_preserved():
    expert = torch.full((3, 2, 1), 0.7)
    solver = make_solver(QuadraticCost(), expert)
    solver.return_best = False
    outputs = solver({'target': expert})
    torch.testing.assert_close(outputs['actions'], outputs['mean'][0])
    assert (
        solver.diagnostics['planning_parameters']['final_selection']
        == 'native CEM mean'
    )
    assert all(
        row['selected_cost'] == row['native_cem_mean_cost']
        for row in solver.diagnostics['selections']
    )


def test_comparison_renderer_receives_final_selected_actions():
    expert = torch.full((3, 2, 1), 100.0)
    solver = make_solver(QuadraticCost(), expert)
    captured = []
    solver.comparison_renderer = lambda expert, selected, rows: (
        captured.append((expert.clone(), selected.clone(), rows)) or []
    )
    outputs = solver({'target': torch.zeros_like(expert)})
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], expert)
    torch.testing.assert_close(captured[0][1], outputs['actions'])
    assert not torch.equal(captured[0][1], expert)
    assert solver.diagnostics['comparison_videos'] == []


def test_comparison_renderer_denormalizes_unblocks_and_uses_global_env(
    tmp_path,
):
    from scripts.plan.eval_wm_expert_candidates import ExpertComparisonRenderer

    calls = []

    def render(sequences, output):
        calls.append((sequences, output))
        return {'expert': {'final_goal_success': True}}

    envs = SimpleNamespace(
        envs=[
            SimpleNamespace(
                unwrapped=SimpleNamespace(
                    planning_action_comparison=render, _executed_actions=[]
                )
            )
            for _ in range(3)
        ]
    )
    processor = SimpleNamespace(
        inverse_transform=lambda actions: actions * 2 + 3
    )
    renderer = ExpertComparisonRenderer(
        envs,
        SimpleNamespace(plan_len=4),
        processor,
        tmp_path,
        max_comparisons=1,
    )
    expert = torch.arange(8).float().reshape(1, 2, 4)
    selected = expert + 10
    records = renderer(
        expert, selected, [{'env_index': 2, 'env_step': 5, 'replan_index': 1}]
    )
    np.testing.assert_array_equal(
        calls[0][0]['expert'], np.arange(8).reshape(4, 2) * 2 + 3
    )
    np.testing.assert_array_equal(
        calls[0][0]['selected'], (np.arange(8).reshape(4, 2) + 10) * 2 + 3
    )
    assert records[0]['env_index'] == 2
    assert records[0]['env_step'] == 5
    assert records[0]['replan_index'] == 1
    assert records[0]['video'] == 'comparisons/comparison_000/env_0.mp4'
    assert renderer(expert, selected, [{'env_index': 0}]) == []
    assert len(calls) == 1
