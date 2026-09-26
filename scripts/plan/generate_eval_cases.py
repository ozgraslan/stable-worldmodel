"""Benchmark generation of environment-backed evaluation cases."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

import stable_worldmodel as swm
from stable_worldmodel.plot import save_panel_videos

from eval_wm import (
    generate_env_eval_cases,
)


def render_offset_trajectories(cfg, world_kwargs, cases) -> Path:
    """Replay cached offset witnesses and save agent/goal panels."""
    render_kwargs = dict(world_kwargs)
    render_kwargs['num_envs'] = 1
    render_kwargs['max_episode_steps'] = (
        max(len(eval_case['witness_actions']) for eval_case in cases) + 1
    )
    render_kwargs['add_pixels'] = True
    world = swm.World(**render_kwargs)
    trajectories = []
    goals = []
    successes = []
    try:
        for eval_case in cases:
            options = {
                'render_goal': True,
                'evaluation_start': eval_case['start'],
                'evaluation_goal': eval_case['goal'],
            }
            world.reset(seed=int(eval_case['seed']), options=options)
            frames = [np.asarray(world.infos['pixels'][0, -1]).copy()]
            goal = np.asarray(world.infos['goal'][0, -1]).copy()
            success = False
            for action in eval_case['witness_actions']:
                _, _, terminated, truncated, world.infos = world.envs.step(
                    np.asarray(action)[None, ...]
                )
                frames.append(
                    np.asarray(world.infos['pixels'][0, -1]).copy()
                )
                if bool(terminated[0]):
                    success = True
                    break
                if bool(truncated[0]):
                    break
            trajectories.append(np.stack(frames))
            goals.append(goal)
            successes.append(success)
    finally:
        world.close()

    render_dir = Path(cfg.eval.generated.render_dir)
    save_panel_videos(
        render_dir,
        {'expert trajectory': trajectories, 'goal': goals},
        successes=successes,
    )
    return render_dir


@hydra.main(version_base=None, config_path='./config', config_name='cube')
def run(cfg: DictConfig) -> None:
    """Generate accepted cases without loading a dataset or world model."""
    cfg.world.max_episode_steps = cfg.eval.eval_budget
    world_kwargs = dict(cfg.world)
    env_img_size = cfg.get('env_img_size', cfg.eval.img_size)
    world_kwargs['image_shape'] = (env_img_size, env_img_size)

    for wrappers_key in ('pre_wrappers', 'extra_wrappers'):
        wrapper_configs = world_kwargs.get(wrappers_key)
        if wrapper_configs is not None:
            world_kwargs[wrappers_key] = [
                hydra.utils.instantiate(wrapper_config)
                for wrapper_config in wrapper_configs
            ]

    start = time.perf_counter()
    cases = generate_env_eval_cases(cfg, world_kwargs)
    elapsed = time.perf_counter() - start
    seconds_per_case = elapsed / len(cases)

    print('\n==== GENERATION BENCHMARK ====')
    print(f'accepted_cases: {len(cases)}')
    print(f'elapsed_seconds: {elapsed:.2f}')
    print(f'seconds_per_case: {seconds_per_case:.2f}')
    print(f'projected_50_case_minutes: {seconds_per_case * 50 / 60:.2f}')

    if cfg.eval.generated.get('render_trajectories', False):
        render_start = time.perf_counter()
        render_dir = render_offset_trajectories(cfg, world_kwargs, cases)
        render_elapsed = time.perf_counter() - render_start
        print(f'render_seconds: {render_elapsed:.2f}')
        print(f'render_dir: {render_dir.resolve()}')


if __name__ == '__main__':
    run()
