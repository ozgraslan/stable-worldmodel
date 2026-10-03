"""Generated expert starts preserve a replayable suffix and its final goal."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.plan.eval_wm import evaluation_kwargs
from scripts.plan.eval_wm_updated import _generate_expert_case


class ExpertWorld:
    def __init__(self):
        self.envs = SimpleNamespace(
            envs=[SimpleNamespace(unwrapped=self)], step=self.step
        )

    def set_policy(self, policy):
        self.policy = policy

    def reset(self, seed, options):
        self.state = 0
        self.infos = {}

    def capture_evaluation_goal(self):
        return {'state': self.state}

    def _get_actions(self):
        return np.array([[self.state + 1]])

    def step(self, action):
        self.state = int(action[0, 0])
        return None, None, [self.state == 8], [False], {}


@pytest.mark.parametrize('beginning', [True, False])
def test_expert_start_and_witness_replay(beginning):
    world = ExpertWorld()
    case = _generate_expert_case(
        world,
        object(),
        0,
        {},
        10,
        start_from_beginning=beginning,
        min_solution_steps=3,
    )
    assert case['start_step'] == (0 if beginning else 5)
    assert case['start'] == {'state': case['start_step']}
    assert case['goal'] == {'state': 8}
    assert case['expert_success_step'] == 8 - case['start_step']
    assert len(case['witness_actions']) == case['expert_success_step']
    world.state = case['start']['state']
    for action in case['witness_actions']:
        world.step(np.asarray(action)[None])
    assert world.capture_evaluation_goal() == case['goal']
    repeated = _generate_expert_case(
        world,
        object(),
        0,
        {},
        10,
        start_from_beginning=beginning,
        min_solution_steps=3,
    )
    assert repeated == case


def test_expert_rollout_too_short_for_valid_start():
    assert (
        _generate_expert_case(
            ExpertWorld(),
            object(),
            0,
            {},
            10,
            start_from_beginning=False,
            min_solution_steps=9,
        )
        is None
    )


@pytest.mark.parametrize('beginning', [True, False])
def test_expert_environment_receives_start_setting(beginning):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            'seed': 42,
            'world': {'expert_checkpoint': 'expert.pt'},
            'eval': {
                'source': 'env',
                'num_eval': 2,
                'start_from_beginning': beginning,
                'env_options': {'example': 1},
            },
        }
    )
    kwargs = evaluation_kwargs(cfg, None)
    assert kwargs['options'] == {
        'example': 1,
        'start_from_beginning': beginning,
    }
