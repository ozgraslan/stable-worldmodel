"""Contracts for state behavior cloning."""

import pytest
import torch

from stable_worldmodel.protocols import Actionable
from stable_worldmodel.wm.bc import StateBC
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained


def inputs(batch_size=2):
    return {
        'obs_a': torch.randn(batch_size, 3, 1),
        'obs_b': torch.randn(batch_size, 3, 2),
    }


def model(**kwargs):
    return StateBC(
        state_dim=3, action_dim=2, state_columns=['obs_a', 'obs_b'], **kwargs
    )


def test_column_time_and_chunk_order():
    policy = model(hidden_dims=())
    info = {
        'obs_b': torch.tensor([[[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]]]),
        'obs_a': torch.tensor([[[1.0], [4.0], [7.0]]]),
        'action': torch.full((1, 5, 2), float('nan')),
    }
    original = {key: value.clone() for key, value in info.items()}
    # Explicit wiring checks chronological flattening, column order,
    # and reshaping into consecutive two-dimensional actions.
    with torch.no_grad():
        policy.network[0].weight.zero_()
        policy.network[0].bias.zero_()
        policy.network[0].weight[:9] = torch.eye(9)
        policy.network[0].bias[9] = 10
    expected = torch.arange(1.0, 11.0).reshape(1, 5, 2)
    torch.testing.assert_close(policy(info), expected)
    for key, value in info.items():
        torch.testing.assert_close(value, original[key], equal_nan=True)


def test_goal_fields_are_ignored():
    policy = model()
    info = inputs()
    expected = policy(info)
    with_goals = {
        **info,
        'goal_obs_a': torch.full((2, 1, 1), float('nan')),
        'goal_obs_b': torch.randn(2, 1, 2),
    }
    torch.testing.assert_close(policy(with_goals), expected)


def test_actionable_shapes_and_gradients():
    policy = model(hidden_dims=(16, 16))
    info = inputs()
    assert isinstance(policy, Actionable)
    chunk = policy(info)
    assert chunk.shape == (2, 5, 2)
    torch.testing.assert_close(policy.get_action(info), chunk[:, 0])
    torch.testing.assert_close(policy.get_action(info, horizon=5), chunk)
    torch.testing.assert_close(
        policy.get_action(info, horizon=2), chunk[:, :2]
    )
    chunk.square().mean().backward()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in policy.parameters()
    )


def test_overhead_defaults():
    policy = StateBC(state_dim=31, action_dim=2)
    assert policy.network[0].in_features == 93
    assert policy.network[-1].out_features == 10
    assert policy(
        {
            'state': torch.zeros(2, 3, 31),
        }
    ).shape == (2, 5, 2)


def test_checkpoint_round_trip(tmp_path):
    config = {
        '_target_': 'stable_worldmodel.wm.bc.StateBC',
        'state_dim': 3,
        'action_dim': 2,
        'state_columns': ['obs_a', 'obs_b'],
        'history_size': 3,
        'frameskip': 5,
        'hidden_dims': [16],
    }
    policy = model(hidden_dims=(16,))
    info = inputs()
    save_pretrained(policy, 'bc', config=config, cache_dir=str(tmp_path))
    restored = load_pretrained('bc', cache_dir=str(tmp_path))
    torch.testing.assert_close(restored(info), policy(info))
    assert restored.state_columns == policy.state_columns
    assert restored.history_size == 3
    assert restored.frameskip == 5


@pytest.mark.parametrize('horizon', [0, 6, 1.5, True])
def test_unsupported_horizon(horizon):
    with pytest.raises(ValueError, match='horizon'):
        model().get_action(inputs(), horizon=horizon)


def test_rejects_prefix_actions():
    with pytest.raises(ValueError, match='prefix_actions'):
        model().get_action(inputs(), prefix_actions=torch.zeros(2, 1, 2))


@pytest.mark.parametrize(
    'key,value',
    [
        ('obs_a', torch.zeros(2, 2, 1)),
        ('obs_a', torch.zeros(2, 1)),
        ('obs_b', torch.zeros(1, 3, 2)),
    ],
)
def test_rejects_malformed_inputs(key, value):
    info = inputs()
    info[key] = value
    with pytest.raises(ValueError):
        model()(info)


def test_missing_state_and_wrong_feature_count():
    info = inputs()
    del info['obs_a']
    with pytest.raises(KeyError, match='obs_a'):
        model()(info)
    with pytest.raises(ValueError, match='Expected 4 state features'):
        StateBC(4, 2)({'state': torch.zeros(2, 3, 3)})


@pytest.mark.parametrize(
    'kwargs',
    [
        {'history_size': 0},
        {'frameskip': 1.5},
        {'state_dim': 0},
        {'action_dim': -1},
        {'hidden_dims': [0]},
        {'state_columns': []},
        {'state_columns': ['state', 'state']},
    ],
)
def test_invalid_configuration(kwargs):
    config = {'state_dim': 3, 'action_dim': 2, **kwargs}
    with pytest.raises(ValueError):
        StateBC(**config)


def transformer_model(**kwargs):
    return model(
        architecture='transformer',
        transformer={
            'embed_dim': 16,
            'depth': 2,
            'heads': 2,
            'dim_head': 8,
            'mlp_dim': 32,
            'dropout': 0.1,
            **kwargs,
        },
    )


def test_transformer_contract_gradients_and_goal_independence():
    torch.manual_seed(7)
    policy = transformer_model().eval()
    info = inputs()
    info['obs_a'].requires_grad_()
    chunk = policy(info)
    assert chunk.shape == (2, 5, 2)
    torch.testing.assert_close(policy.get_action(info), chunk[:, 0])
    torch.testing.assert_close(policy.get_action(info, horizon=5), chunk)
    torch.testing.assert_close(
        policy({**info, 'goal_obs_a': torch.full((2, 1, 1), float('nan'))}),
        chunk,
    )
    chunk.square().mean().backward()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in policy.parameters()
    )
    # The latest action prediction can learn from every historical state.
    assert (info['obs_a'].grad.abs().sum(dim=(0, 2)) > 0).all()


def test_transformer_uses_native_causal_blocks():
    from stable_worldmodel.wm.lewm.module import Transformer

    policy = transformer_model().eval()
    assert isinstance(policy.network.transformer, Transformer)
    tokens = torch.randn(2, 3, 16)
    changed = tokens.clone()
    changed[:, -1] += torch.randn(2, 16)
    with torch.no_grad():
        before = policy.network.transformer(tokens)
        after = policy.network.transformer(changed)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_transformer_checkpoint_round_trip(tmp_path):
    config = {
        '_target_': 'stable_worldmodel.wm.bc.StateBC',
        'state_dim': 3,
        'action_dim': 2,
        'state_columns': ['obs_a', 'obs_b'],
        'architecture': 'transformer',
        'transformer': {
            'embed_dim': 16,
            'depth': 2,
            'heads': 2,
            'dim_head': 8,
            'mlp_dim': 32,
            'dropout': 0.1,
        },
    }
    policy = transformer_model().eval()
    info = inputs()
    save_pretrained(
        policy, 'transformer_bc', config=config, cache_dir=str(tmp_path)
    )
    restored = load_pretrained(
        'transformer_bc', cache_dir=str(tmp_path)
    ).eval()
    assert restored.architecture == 'transformer'
    torch.testing.assert_close(restored(info), policy(info))


@pytest.mark.parametrize(
    'kwargs',
    [
        {'depth': 0},
        {'heads': 0},
        {'embed_dim': -1},
        {'dim_head': 1.5},
        {'mlp_dim': 0},
        {'dropout': 1.0},
        {'dropout': -0.1},
    ],
)
def test_invalid_transformer_configuration(kwargs):
    with pytest.raises(ValueError):
        transformer_model(**kwargs)


def test_unknown_architecture():
    with pytest.raises(ValueError, match='architecture'):
        model(architecture='unknown')
