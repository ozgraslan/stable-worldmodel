"""Export generated cube evaluation cases as a fixed HDF5 dataset.

This lets the expensive environment/expert/random filtering run once. Later
evaluations can use ``eval.source=dataset`` and the normal dataset-backed
evaluation path.
"""

import json
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

from eval_wm import generate_env_eval_cases


def _default_dataset_path(cfg: DictConfig, world_kwargs: dict) -> Path:
    dataset_root = Path(
        swm.data.utils.get_cache_dir(
            cfg.get('cache_dir', None), sub_folder='datasets'
        )
    )
    env_slug = world_kwargs['env_name'].replace('/', '_')
    generated = cfg.eval.generated
    min_distance = float(
        generated.get(
            'min_expert_trajectory_distance',
            generated.get('min_goal_distance', 0.10),
        )
    )
    return dataset_root / 'generated_eval' / (
        f'{env_slug}_seed{cfg.seed}_n{cfg.eval.num_eval}'
        f'_full_b{cfg.eval.eval_budget}'
        f'_min{int(generated.get("min_solution_steps", 5))}'
        f'_d{min_distance:g}'
        f'_r{int(generated.get("random_horizon", cfg.eval.eval_budget))}'
        f'x{int(generated.get("random_trials", 16))}.h5'
    )


def _flatten_key(key: str) -> str:
    return key.replace('/', '_').replace('.', '_')


def _numeric_info_columns(info: dict) -> dict:
    cols = {}
    for key, value in info.items():
        if key.startswith('_') or key in (
            'action',
            'goal',
            'id',
            'target',
            'step_idx',
            'env_name',
        ):
            continue
        if not isinstance(value, np.ndarray):
            continue
        arr = np.asarray(value[0, -1] if value.ndim > 1 else value[0])
        if arr.dtype.kind in ('b', 'i', 'u', 'f'):
            cols[_flatten_key(key)] = arr.copy()
    return cols


def _state_columns(state: dict) -> dict:
    cols = {
        'qpos': np.asarray(state['qpos'], dtype=np.float32),
        'qvel': np.asarray(state['qvel'], dtype=np.float32),
        'mocap_pos': np.asarray(state['mocap_pos'], dtype=np.float32),
        'mocap_quat': np.asarray(state['mocap_quat'], dtype=np.float32),
        'ctrl': np.asarray(state['ctrl'], dtype=np.float32),
        'act': np.asarray(state['act'], dtype=np.float32),
        'sim_time': np.asarray([state['time']], dtype=np.float32),
        'effector_position': np.asarray(
            state['effector_position'], dtype=np.float32
        ),
        'gripper_opening': np.asarray(
            state['gripper_opening'], dtype=np.float32
        ),
    }
    for i, (pos, quat) in enumerate(
        zip(state['block_positions'], state['block_quaternions'])
    ):
        cols[f'privileged_block_{i}_pos'] = np.asarray(
            pos, dtype=np.float32
        )
        cols[f'privileged_block_{i}_quat'] = np.asarray(
            quat, dtype=np.float32
        )
    return cols


def _append_row(episode: dict, row: dict) -> None:
    for key, value in row.items():
        episode.setdefault(key, []).append(value)


def _cases_to_episodes(cfg: DictConfig, world_kwargs: dict, cases: list[dict]):
    render_kwargs = dict(world_kwargs)
    render_kwargs['num_envs'] = 1
    render_kwargs['max_episode_steps'] = (
        max(len(eval_case['witness_actions']) for eval_case in cases) + 1
    )
    render_kwargs['add_pixels'] = True
    world = swm.World(**render_kwargs)
    episodes = []
    try:
        for ep_idx, eval_case in enumerate(cases):
            options = OmegaConf.to_container(
                cfg.eval.get('env_options', {}), resolve=True
            )
            options['render_goal'] = True
            options['evaluation_start'] = eval_case['start']
            options['evaluation_goal'] = eval_case['goal']
            world.reset(seed=int(eval_case['seed']), options=options)
            env = world.envs.envs[0].unwrapped
            capture = getattr(env, 'capture_evaluation_goal', None)
            if capture is None:
                raise TypeError(
                    f'{type(env).__name__} must implement '
                    'capture_evaluation_goal()'
                )

            actions = [
                np.asarray(action, dtype=np.float32)
                for action in eval_case['witness_actions']
            ]
            if not actions:
                raise ValueError('Generated case has no witness actions.')
            terminal_action = np.zeros_like(actions[-1])
            episode = {}

            def add_current_row(step_idx: int, action: np.ndarray) -> None:
                state = capture()
                row = {
                    'episode_idx': np.asarray(ep_idx, dtype=np.int64),
                    'ep_idx': np.asarray(ep_idx, dtype=np.int64),
                    'step_idx': np.asarray(step_idx, dtype=np.int64),
                    'seed': np.asarray(int(eval_case['seed']), dtype=np.int64),
                    'id': np.asarray(ep_idx, dtype=np.int64),
                    'pixels': np.asarray(
                        world.infos['pixels'][0, -1], dtype=np.uint8
                    ).copy(),
                    'action': np.asarray(action, dtype=np.float32),
                }
                row.update(_numeric_info_columns(world.infos))
                row.update(_state_columns(state))
                _append_row(episode, row)

            add_current_row(0, actions[0])
            for step_idx, action in enumerate(actions, start=1):
                _, _, _, _, world.infos = world.envs.step(
                    action[None, ...]
                )
                next_action = (
                    actions[step_idx]
                    if step_idx < len(actions)
                    else terminal_action
                )
                add_current_row(step_idx, next_action)

            episodes.append(episode)
    finally:
        world.close()
    return episodes


@hydra.main(version_base=None, config_path='./config', config_name='cube')
def run(cfg: DictConfig) -> None:
    """Generate accepted cases and export their witness trajectories to HDF5."""
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

    cases = generate_env_eval_cases(cfg, world_kwargs)
    episodes = _cases_to_episodes(cfg, world_kwargs, cases)

    dataset_path = cfg.eval.generated.get('dataset_path')
    dataset_path = (
        Path(dataset_path)
        if dataset_path is not None
        else _default_dataset_path(cfg, world_kwargs)
    )
    mode = str(cfg.eval.generated.get('dataset_mode', 'overwrite'))
    with swm.data.HDF5Writer(dataset_path, mode=mode) as writer:
        writer.write_episodes(episodes)

    sidecar_path = dataset_path.with_suffix(dataset_path.suffix + '.json')
    sidecar_path.write_text(
        json.dumps(
            {
                'source': 'generated_env',
                'num_episodes': len(episodes),
                'episode_lengths': [len(ep['step_idx']) for ep in episodes],
                'generated_case_seeds': [case['seed'] for case in cases],
                'eval_command_overrides': {
                    'eval.source': 'dataset',
                    'eval.dataset_name': str(dataset_path),
                    'eval.start_from_beginning': True,
                    'eval.goal_from_episode_end': True,
                    'eval.num_eval': len(episodes),
                },
            },
            indent=2,
        )
        + '\n'
    )
    print(f'[export] wrote {len(episodes)} episodes to {dataset_path}')
    print(f'[export] wrote metadata to {sidecar_path}')
    print('[export] evaluate with:')
    print(
        '  eval.source=dataset '
        f'eval.dataset_name={dataset_path} '
        'eval.start_from_beginning=true '
        'eval.goal_from_episode_end=true'
    )


if __name__ == '__main__':
    run()
