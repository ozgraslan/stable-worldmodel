"""Shortest-path waypoint expert for DINO PointMaze."""

from __future__ import annotations

from collections import deque

import numpy as np

from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Grid shortest-path planner followed by a PD waypoint controller."""

    def __init__(
        self,
        kp: float = 1.0,
        kd: float = 0.2,
        waypoint_tolerance: float = 0.2,
        action_noise: float = 0.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'expert'
        self.kp = float(kp)
        self.kd = float(kd)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.action_noise = float(action_noise)
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env) -> None:
        self.env = env
        envs = getattr(env, 'envs', [env])
        self._pointmaze_envs = []
        for wrapped_env in envs:
            current = wrapped_env
            while current is not None:
                if getattr(current, 'env_name', None) == 'DINOPointMaze':
                    self._pointmaze_envs.append(current)
                    break
                current = getattr(current, 'env', None)
            else:
                raise ValueError(
                    'DINOPointMaze ExpertPolicy requires '
                    'swm/DINOPointMaze-v0.'
                )

    @staticmethod
    def _shortest_path(maze_map, start, goal):
        start, goal = tuple(start), tuple(goal)
        queue = deque([start])
        parent = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + di, cell[1] + dj)
                if (
                    0 <= nxt[0] < len(maze_map)
                    and 0 <= nxt[1] < len(maze_map[0])
                    and maze_map[nxt[0]][nxt[1]] != 1
                    and nxt not in parent
                ):
                    parent[nxt] = cell
                    queue.append(nxt)
        if goal not in parent:
            return [start]
        path = []
        cell = goal
        while cell is not None:
            path.append(cell)
            cell = parent[cell]
        return path[::-1]

    def _action(self, env, state, goal_state):
        maze = env.point_maze.maze
        start = maze.cell_xy_to_rowcol(state[:2])
        goal = maze.cell_xy_to_rowcol(goal_state[:2])
        path = self._shortest_path(maze.maze_map, start, goal)

        if len(path) > 1:
            waypoint = maze.cell_rowcol_to_xy(np.asarray(path[1]))
            if np.linalg.norm(waypoint - state[:2]) <= self.waypoint_tolerance:
                waypoint = (
                    maze.cell_rowcol_to_xy(np.asarray(path[2]))
                    if len(path) > 2
                    else goal_state[:2]
                )
        else:
            waypoint = goal_state[:2]

        action = self.kp * (waypoint - state[:2]) - self.kd * state[2:4]
        if self.action_noise:
            action += self.rng.normal(0.0, self.action_noise, size=2)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def get_action(self, info_dict, **kwargs):
        if not hasattr(self, 'env'):
            raise RuntimeError('Environment not set for the policy')
        if 'state' not in info_dict or 'goal_state' not in info_dict:
            raise KeyError(
                "PointMaze expert requires 'state' and 'goal_state'"
            )

        envs = self._pointmaze_envs
        states = np.asarray(info_dict['state'])
        goals = np.asarray(info_dict['goal_state'])
        if len(envs) == 1 and states.ndim == 1:
            return self._action(envs[0], states, goals)

        states = states.reshape(len(envs), -1)
        goals = goals.reshape(len(envs), -1)
        return np.stack(
            [
                self._action(env, state, goal)
                for env, state, goal in zip(envs, states, goals)
            ]
        )
