import gymnasium as gym
import numpy as np

from stable_worldmodel.envs.ogbench.online_rl import CubeOnlineRLWrapper


class _Data:
    mocap_pos = np.array([[0.5, 0.2, 0.02]], dtype=np.float64)


class _CubeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,))

    def __init__(self):
        self._num_cubes = 1
        self._data = _Data()
        self._cube_target_mocap_ids = [0]

    def _info(self, success=False):
        return {
            'privileged/block_0_pos': np.array([0.3, 0.0, 0.02]),
            'proprio/effector_pos': np.array([0.3, 0.0, 0.1]),
            'success': success,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), self._info()

    def step(self, action):
        return np.ones(4, dtype=np.float32), -1.0, False, False, self._info()


def test_wrapper_appends_goal_to_observation():
    env = CubeOnlineRLWrapper(_CubeEnv(), reward_mode='sparse')
    observation, info = env.reset()

    assert observation.shape == (7,)
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(observation[-3:], [0.5, 0.2, 0.02])
    np.testing.assert_allclose(info['desired_goal'], observation[-3:])


def test_sparse_reward_is_preserved():
    env = CubeOnlineRLWrapper(_CubeEnv(), reward_mode='sparse')
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())
    assert reward == -1.0


def test_dense_reward_has_distance_signal():
    env = CubeOnlineRLWrapper(
        _CubeEnv(),
        reward_mode='dense',
        goal_distance_weight=10.0,
        reach_distance_weight=1.0,
        success_bonus=10.0,
    )
    env.reset()
    _, reward, _, _, _ = env.step(env.action_space.sample())

    goal_distance = np.linalg.norm([0.3 - 0.5, 0.0 - 0.2, 0.0])
    reach_distance = 0.08
    np.testing.assert_allclose(
        reward,
        -10.0 * goal_distance - reach_distance,
        rtol=1e-6,
        atol=1e-6,
    )
