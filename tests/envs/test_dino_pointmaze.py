import numpy as np
import pytest

from stable_worldmodel.envs.dino_pointmaze.expert_policy import ExpertPolicy


class FakeMaze:
    maze_map = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    def cell_xy_to_rowcol(self, xy):
        return np.asarray(xy, dtype=int)

    def cell_rowcol_to_xy(self, cell):
        return np.asarray(cell, dtype=float)


class FakeRawEnv:
    env_name = 'DINOPointMaze'
    point_maze = type('PointMaze', (), {'maze': FakeMaze()})()

    @property
    def unwrapped(self):
        return self


class FakePool:
    def __init__(self, n=1):
        self.envs = [FakeRawEnv() for _ in range(n)]


def test_expert_routes_around_u_maze():
    policy = ExpertPolicy(kp=1.0, kd=0.0)
    policy.set_env(FakePool())
    action = policy.get_action(
        {
            'state': np.array([[[1.0, 1.0, 0.0, 0.0]]]),
            'goal_state': np.array([[[3.0, 1.0, 0.0, 0.0]]]),
        }
    )
    np.testing.assert_array_equal(action, [[0.0, 1.0]])


def test_expert_supports_vectorized_infos():
    policy = ExpertPolicy(kp=1.0, kd=0.0)
    policy.set_env(FakePool(2))
    action = policy.get_action(
        {
            'state': np.array(
                [[[1.0, 1.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 0.0]]]
            ),
            'goal_state': np.array(
                [[[3.0, 1.0, 0.0, 0.0]], [[1.0, 3.0, 0.0, 0.0]]]
            ),
        }
    )
    assert action.shape == (2, 2)
    assert action.dtype == np.float32


def test_environment_smoke():
    pytest.importorskip('gymnasium_robotics')
    from stable_worldmodel.envs.dino_pointmaze import DINOPointMazeEnv

    env = DINOPointMazeEnv(render_mode='rgb_array')
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4,)
        assert info['state'].shape == (4,)
        assert info['goal_state'].shape == (4,)
        assert info['goal'].shape == (224, 224, 3)
        obs, _, _, _, info = env.step(env.action_space.sample())
        assert obs.shape == info['state'].shape == (4,)
    finally:
        env.close()
