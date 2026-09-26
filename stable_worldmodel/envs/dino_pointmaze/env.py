"""DINO-WM-compatible PointMaze backed by Gymnasium-Robotics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DINOPointMazeEnv(gym.Wrapper):
    """Adapt the maintained D4RL PointMaze port to the SWM environment API.

    The dynamics and U-maze come from ``PointMaze_UMaze-v3``.  The wrapper
    exposes the four-dimensional proprioceptive state as the observation and
    adds SWM's ``state``, ``goal_state``, and rendered ``goal`` info fields.
    A variation space is intentionally not defined.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 224,
        height: int = 224,
        camera_distance: float = 5.0,
        camera_azimuth: float = 180.0,
        success_threshold: float = 0.2,
        show_target: bool = False,
        continuing_task: bool = False,
        reset_target: bool = False,
        **kwargs,
    ):
        try:
            import gymnasium_robotics
        except ImportError as exc:
            raise ImportError(
                "DINOPointMaze requires the 'env' extra: "
                "pip install 'stable-worldmodel[env]'"
            ) from exc

        gym.register_envs(gymnasium_robotics)
        env = gym.make(
            'PointMaze_UMaze-v3',
            render_mode=render_mode,
            width=width,
            height=height,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        super().__init__(env)
        self.env_name = 'DINOPointMaze'
        self.width = width
        self.height = height
        if success_threshold <= 0:
            raise ValueError('success_threshold must be positive')
        self.success_threshold = float(success_threshold)
        self.continuing_task = bool(continuing_task)
        self.show_target = bool(show_target)
        if not self.show_target:
            target_id = self.point_maze.target_site_id
            self.point_maze.model.site_rgba[target_id, 3] = 0.0
        # Match DINO-WM's overhead camera. Gymnasium-Robotics creates the
        # actual viewer lazily on the first render, so update its camera
        # configuration before that happens.
        self.point_maze.point_env.mujoco_renderer.default_cam_config = {
            'lookat': np.array([0.0, 0.0, 0.0]),
            'distance': float(camera_distance),
            'azimuth': float(camera_azimuth),
            'elevation': -90.0,
        }
        self.observation_space = env.observation_space['observation']
        self._goal_state = np.zeros(4, dtype=np.float64)
        self._goal_image = None
        # Dataset-backed World.evaluate applies setup callables to the
        # deepest unwrapped environment. Expose the adapter's state hooks
        # there so evaluation can restore PointMaze start and goal states.
        self.point_maze._set_state = self._set_sim_state
        self.point_maze._set_goal_state = self._set_goal_state

    @property
    def point_maze(self):
        """The unwrapped Gymnasium-Robotics PointMaze environment."""
        return self.env.unwrapped

    def _set_sim_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4,):
            raise ValueError(
                f'initial_state must have shape (4,), got {state.shape}'
            )
        self.point_maze.point_env.set_state(state[:2], state[2:])

    def _set_goal_state(self, goal_state: np.ndarray) -> None:
        goal_state = np.asarray(goal_state, dtype=np.float64)
        if goal_state.shape not in ((2,), (4,)):
            raise ValueError(
                f'goal_state must have shape (2,) or (4,), got {goal_state.shape}'
            )
        if goal_state.shape == (2,):
            goal_state = np.concatenate([goal_state, np.zeros(2)])
        self._goal_state = goal_state.copy()
        self.point_maze.goal = goal_state[:2].copy()
        self.point_maze.update_target_site_pos()

    def _state(self) -> np.ndarray:
        point = self.point_maze.point_env
        return np.concatenate(
            [point.data.qpos[:2], point.data.qvel[:2]]
        ).astype(np.float64, copy=True)

    def _render_goal(self) -> np.ndarray | None:
        if self.render_mode != 'rgb_array':
            return None
        state = self._state()
        self._set_sim_state(self._goal_state)
        frame = np.asarray(self.render()).copy()
        self._set_sim_state(state)
        return frame

    def _info(self, info: dict | None = None) -> dict:
        info = dict(info or {})
        state = self._state()
        distance = float(np.linalg.norm(state[:2] - self._goal_state[:2]))
        info.update(
            state=state,
            goal_state=self._goal_state.copy(),
            success=distance <= self.success_threshold,
            state_dist=distance,
        )
        if self._goal_image is not None:
            info['goal'] = self._goal_image.copy()
        return info

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        initial_state = options.pop('initial_state', None)
        goal_state = options.pop('goal_state', None)
        observation, info = self.env.reset(seed=seed, options=options or None)

        if initial_state is not None:
            self._set_sim_state(initial_state)
        if goal_state is None:
            goal_state = np.concatenate(
                [observation['desired_goal'], np.zeros(2, dtype=np.float64)]
            )
        self._set_goal_state(goal_state)
        self._goal_image = self._render_goal()
        state = self._state()
        return state, self._info(info)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        state = self._state()
        info = self._info(info)
        success = bool(info['success'])
        # Gymnasium-Robotics uses a fixed 0.45 radius internally. Override
        # its sparse task signals so this adapter's configured threshold is
        # authoritative.
        reward = float(success)
        terminated = False if self.continuing_task else success
        return state, reward, terminated, truncated, info
