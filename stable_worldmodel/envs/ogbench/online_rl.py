"""Gymnasium wrappers for online RL on OGBench manipulation tasks."""

from typing import Literal

import gymnasium as gym
import numpy as np


class CubeOnlineRLWrapper(gym.Wrapper):
    """Expose cube goals to a flat-state policy and optionally shape rewards.

    ``CubeEnv`` returns the task goal in its info dictionary.  Standard online
    RL libraries only pass the observation to the policy, so an unwrapped agent
    cannot tell which target it is meant to reach.  This wrapper appends the
    target position of every cube to the state observation.

    The ``sparse`` reward preserves the environment reward.  The ``dense``
    reward adds cube-to-goal distance, end-effector-to-cube distance, and a
    success bonus, which makes vanilla SAC/PPO usable without HER.

    Args:
        env: A state-observation :class:`CubeEnv` instance.
        reward_mode: ``"sparse"`` or ``"dense"``.
        goal_distance_weight: Weight on cube-to-goal distance.
        reach_distance_weight: Weight on end-effector-to-cube distance.
        success_bonus: Reward added on successful steps.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_mode: Literal['sparse', 'dense'] = 'dense',
        goal_distance_weight: float = 10.0,
        reach_distance_weight: float = 1.0,
        success_bonus: float = 10.0,
    ):
        super().__init__(env)
        if reward_mode not in ('sparse', 'dense'):
            raise ValueError(
                "reward_mode must be either 'sparse' or 'dense', "
                f'got {reward_mode!r}'
            )
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError('CubeOnlineRLWrapper requires a Box observation space')
        if len(env.observation_space.shape) != 1:
            raise ValueError(
                'CubeOnlineRLWrapper requires flat state observations; '
                "construct CubeEnv with ob_type='states'"
            )

        base = env.unwrapped
        if not hasattr(base, '_num_cubes'):
            raise TypeError('CubeOnlineRLWrapper requires an OGBench CubeEnv')

        self.reward_mode = reward_mode
        self.goal_distance_weight = float(goal_distance_weight)
        self.reach_distance_weight = float(reach_distance_weight)
        self.success_bonus = float(success_bonus)
        self._num_cubes = int(base._num_cubes)
        self._desired_goal: np.ndarray | None = None

        goal_low = np.full(3 * self._num_cubes, -np.inf, dtype=np.float32)
        goal_high = np.full(3 * self._num_cubes, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [env.observation_space.low.astype(np.float32), goal_low]
            ),
            high=np.concatenate(
                [env.observation_space.high.astype(np.float32), goal_high]
            ),
            dtype=np.float32,
        )

    def _read_desired_goal(self) -> np.ndarray:
        """Read target positions after reset from CubeEnv's MuJoCo state."""
        base = self.env.unwrapped
        return np.concatenate(
            [
                base._data.mocap_pos[mocap_id].copy()
                for mocap_id in base._cube_target_mocap_ids
            ]
        ).astype(np.float32)

    def _observation(self, observation) -> np.ndarray:
        if self._desired_goal is None:
            raise RuntimeError('reset() must be called before observations are used')
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), self._desired_goal]
        )

    def _dense_reward(self, info: dict) -> float:
        cube_positions = np.concatenate(
            [
                np.asarray(
                    info[f'privileged/block_{i}_pos'], dtype=np.float32
                )
                for i in range(self._num_cubes)
            ]
        ).reshape(self._num_cubes, 3)
        goal_positions = self._desired_goal.reshape(self._num_cubes, 3)
        goal_distance = np.linalg.norm(
            cube_positions - goal_positions, axis=-1
        ).mean()

        end_effector = np.asarray(
            info['proprio/effector_pos'], dtype=np.float32
        )
        reach_distance = np.linalg.norm(
            cube_positions - end_effector[None, :], axis=-1
        ).min()

        return float(
            -self.goal_distance_weight * goal_distance
            - self.reach_distance_weight * reach_distance
            + self.success_bonus * float(info['success'])
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._desired_goal = self._read_desired_goal()
        info = dict(info)
        info['desired_goal'] = self._desired_goal.copy()
        return self._observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_mode == 'dense':
            reward = self._dense_reward(info)
        info = dict(info)
        info['desired_goal'] = self._desired_goal.copy()
        return (
            self._observation(observation),
            float(reward),
            terminated,
            truncated,
            info,
        )


__all__ = ['CubeOnlineRLWrapper']
