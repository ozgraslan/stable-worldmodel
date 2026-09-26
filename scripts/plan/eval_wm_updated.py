"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import json
import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def _eval_dir(policy: str, subdir: str) -> str:
    checkpoints_dir = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
    if policy == 'random':
        return str(Path(checkpoints_dir, subdir))

    policy_path = Path(checkpoints_dir, policy)
    training_dir = policy_path.parent if policy_path.suffix else policy_path
    return str(training_dir / 'evals' / subdir)


if not OmegaConf.has_resolver('eval_dir'):
    OmegaConf.register_new_resolver('eval_dir', _eval_dir)


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episode_index_column(dataset) -> str:
    """Return the dataset's structural episode-index column name."""
    index_columns = getattr(dataset, '_index_columns', None)
    if index_columns:
        return index_columns[0]
    if 'episode_idx' in dataset.column_names:
        return 'episode_idx'
    if 'ep_idx' in dataset.column_names:
        return 'ep_idx'
    # HDF5 readers historically expose the structural column through
    # get_col_data without including it in column_names.
    return 'episode_idx'


def get_episodes_length(dataset, episodes):
    col_name = get_episode_index_column(dataset)

    # Scalar Lance columns may be represented as (N, 1), whereas HDF5
    # returns (N,). Normalize both layouts before constructing row masks.
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    keys_to_merge = cfg.dataset.get('keys_to_merge')
    if keys_to_merge is not None:
        keys_to_merge = OmegaConf.to_container(keys_to_merge, resolve=True)
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
        keys_to_merge=keys_to_merge,
    )
    return dataset


def select_dataset_eval_rows(cfg, dataset, ep_indices):
    """Select dataset-backed init/goal pairs for evaluation."""
    episode_len = get_episodes_length(dataset, ep_indices)
    episode_len_dict = {
        ep_id: episode_len[i] for i, ep_id in enumerate(ep_indices)
    }
    col_name = get_episode_index_column(dataset)
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    episode_len_per_row = np.array(
        [episode_len_dict[ep_id] for ep_id in episode_idx]
    )

    if cfg.eval.get('goal_from_episode_end', False):
        valid_mask = step_idx < episode_len_per_row - 1
    else:
        valid_mask = (
            step_idx + cfg.eval.goal_offset_steps < episode_len_per_row
        )
    if cfg.eval.get('start_from_beginning', False):
        valid_mask &= step_idx == 0

    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f'Only {len(valid_indices)} valid starting points are available, '
            f'but eval.num_eval={cfg.eval.num_eval}.'
        )

    selected_indices = np.sort(
        np.random.default_rng(cfg.seed).choice(
            valid_indices, size=cfg.eval.num_eval, replace=False
        )
    )
    print(selected_indices)
    # Structural index columns are intentionally excluded from Lance
    # get_row_data(), so select them from the full arrays loaded above.
    eval_episodes = episode_idx[selected_indices]
    eval_start_idx = step_idx[selected_indices]
    goal_offsets = cfg.eval.goal_offset_steps
    if cfg.eval.get('goal_from_episode_end', False):
        goal_offsets = np.array(
            [episode_len_dict[ep_id] for ep_id in eval_episodes]
        ) - eval_start_idx - 1
    return eval_episodes, eval_start_idx, goal_offsets


def _update_evaluation_expert_info(world) -> None:
    """Add optional environment-provided metadata to a one-env world."""
    env = world.envs.envs[0].unwrapped
    expert_info_fn = getattr(env, 'evaluation_expert_info', None)
    if expert_info_fn is not None:
        for key, value in expert_info_fn().items():
            world.infos[key] = np.asarray(value)[None, None, ...]


def _run_generated_case(world, policy, seed, options, horizon) -> int | None:
    """Return the first success step for one deterministically reset task."""
    world.set_policy(policy)
    world.reset(seed=seed, options=options)
    for step in range(1, horizon + 1):
        _update_evaluation_expert_info(world)
        actions = world._get_actions()
        _, _, terminated, truncated, world.infos = world.envs.step(actions)
        if bool(terminated[0]):
            return step
        if bool(truncated[0]):
            return None
    return None


def _generate_expert_case(world, expert, seed, options, expert_horizon):
    """Roll out an expert and capture its complete successful trajectory."""
    world.set_policy(expert)
    world.reset(seed=seed, options=options)
    actions = []
    env = world.envs.envs[0].unwrapped
    capture = getattr(env, 'capture_evaluation_goal', None)
    if capture is None:
        raise TypeError(
            f'{type(env).__name__} must implement capture_evaluation_goal()'
        )
    states = [capture()]
    expert_final_state = None
    success_step = None
    for step in range(1, expert_horizon + 1):
        _update_evaluation_expert_info(world)
        action = world._get_actions()
        actions.append(np.asarray(action[0]).tolist())
        _, _, terminated, truncated, world.infos = world.envs.step(action)
        states.append(capture())
        if bool(terminated[0]):
            success_step = step
            expert_final_state = states[-1]
            break
        if bool(truncated[0]):
            break
    if success_step is None:
        return None
    return {
        'seed': seed,
        'start': states[0],
        'goal': states[success_step],
        'expert_final_state': expert_final_state,
        'witness_actions': actions[:success_step],
        'expert_success_step': success_step,
    }


def generate_env_eval_cases(cfg, world_kwargs) -> list[dict]:
    """Cache environment tasks with an expert witness and random baseline."""
    generated = cfg.eval.generated
    expert_horizon = int(cfg.eval.eval_budget)
    min_steps = int(generated.get('min_solution_steps', 5))
    min_expert_distance = float(
        generated.get(
            'min_expert_trajectory_distance',
            generated.get('min_goal_distance', 0.10),
        )
    )
    random_horizon = int(generated.get('random_horizon', 4))
    random_trials = int(generated.get('random_trials', 16))
    max_attempts = int(generated.get('max_attempts', 10_000))
    progress_interval = int(generated.get('progress_interval', 25))
    replay_tolerance = float(generated.get('replay_tolerance', 1e-5))
    specification = {
        'schema_version': 5,
        'env_name': world_kwargs['env_name'],
        'env_type': world_kwargs.get('env_type'),
        'base_seed': int(cfg.seed),
        'num_eval': int(cfg.eval.num_eval),
        'eval_budget': int(cfg.eval.eval_budget),
        'min_solution_steps': min_steps,
        'min_expert_trajectory_distance': min_expert_distance,
        'random_horizon': random_horizon,
        'random_trials': random_trials,
        'expert_policy': OmegaConf.to_container(
            generated.expert_policy, resolve=True
        ),
    }

    cache_path = generated.get('cache_path')
    if cache_path is None:
        cache_root = Path(
            swm.data.utils.get_cache_dir(sub_folder='generated_eval')
        )
        env_slug = world_kwargs['env_name'].replace('/', '_')
        cache_path = cache_root / (
            f'{env_slug}_seed{cfg.seed}_n{cfg.eval.num_eval}'
            f'_full_b{cfg.eval.eval_budget}'
            f'_min{min_steps}_d{min_expert_distance:g}'
            f'_r{random_horizon}x{random_trials}.json'
        )
    cache_path = Path(cache_path)
    if cache_path.exists() and not generated.get('force_regenerate', False):
        payload = json.loads(cache_path.read_text())
        cases = payload.get('cases', [])
        if (
            payload.get('specification') == specification
            and len(cases) == cfg.eval.num_eval
        ):
            print(
                f'[eval] loaded {len(cases)} generated cases from '
                f'{cache_path}'
            )
            return cases

    generator_kwargs = dict(world_kwargs)
    generator_kwargs['num_envs'] = 1
    generator_kwargs['max_episode_steps'] = (
        max(expert_horizon, random_horizon) + 1
    )
    generator_kwargs['add_pixels'] = bool(
        generated.get('validation_add_pixels', False)
    )
    generator = swm.World(**generator_kwargs)
    options = OmegaConf.to_container(
        cfg.eval.get('env_options', {}), resolve=True
    )
    if not generated.get('render_during_validation', False):
        options['render_goal'] = False
    expert = hydra.utils.instantiate(generated.expert_policy)

    accepted = []
    try:
        for attempt in range(max_attempts):
            seed = int(cfg.seed + attempt)
            if attempt % progress_interval == 0:
                print(
                    f'[eval] checked {attempt} candidate seeds; '
                    f'accepted {len(accepted)}/{cfg.eval.num_eval}',
                    flush=True,
                )
            eval_case = _generate_expert_case(
                generator,
                expert,
                seed,
                options,
                expert_horizon,
            )
            if (
                eval_case is None
                or eval_case['expert_success_step'] < min_steps
            ):
                continue

            window_options = dict(options)
            window_options['evaluation_start'] = eval_case['start']
            generator.reset(seed=seed, options=window_options)
            env = generator.envs.envs[0].unwrapped
            distance_fn = getattr(env, 'evaluation_goal_distance', None)
            if distance_fn is None:
                raise TypeError(
                    f'{type(env).__name__} must implement '
                    'evaluation_goal_distance() for generated_env evaluation'
                )
            trajectory_distance = float(distance_fn(eval_case['goal']))
            if trajectory_distance < min_expert_distance:
                continue

            # The cached actions are the constructive reachability witness.
            generator.reset(seed=seed, options=window_options)
            for action in eval_case['witness_actions']:
                generator.envs.step(np.asarray(action)[None, ...])
            replay_error = float(distance_fn(eval_case['goal']))
            if replay_error > replay_tolerance:
                continue

            case_options = dict(options)
            case_options['evaluation_start'] = eval_case['start']
            case_options['evaluation_goal'] = eval_case['goal']
            random_successes = 0
            for trial in range(random_trials):
                random_policy = swm.policy.RandomPolicy(
                    seed=seed + 1_000_003 * (trial + 1)
                )
                if (
                    _run_generated_case(
                        generator,
                        random_policy,
                        seed,
                        case_options,
                        random_horizon,
                    )
                    is not None
                ):
                    random_successes += 1
                    break
            if random_successes:
                continue

            eval_case['expert_trajectory_distance'] = trajectory_distance
            eval_case['witness_replay_error'] = replay_error
            accepted.append(eval_case)
            print(
                f'[eval] generated case {len(accepted)}/{cfg.eval.num_eval}: '
                f'seed={seed}, '
                f'expert_steps={eval_case["expert_success_step"]}, '
                f'trajectory_distance={trajectory_distance:.3f}',
                flush=True,
            )
            if len(accepted) == cfg.eval.num_eval:
                break
    finally:
        generator.close()

    if len(accepted) != cfg.eval.num_eval:
        raise RuntimeError(
            f'Generated only {len(accepted)}/{cfg.eval.num_eval} cases after '
            f'{max_attempts} attempts. Relax eval.generated filters.'
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                'specification': specification,
                'cases': accepted,
            },
            indent=2,
        )
        + '\n'
    )
    print(f'[eval] saved generated cases to {cache_path}')
    return accepted


def render_generated_references(world_kwargs, generated_cases, base_options):
    """Render cached witnesses and recover their simulator states."""
    reference_kwargs = dict(world_kwargs)
    reference_kwargs['num_envs'] = 1
    reference_kwargs['max_episode_steps'] = (
        max(
            len(eval_case['witness_actions'])
            for eval_case in generated_cases
        )
        + 1
    )
    reference_kwargs['add_pixels'] = True
    reference_world = swm.World(**reference_kwargs)
    trajectories = []
    state_trajectories = []
    try:
        for eval_case in generated_cases:
            options = dict(base_options)
            options['render_goal'] = True
            options['evaluation_start'] = eval_case['start']
            options['evaluation_goal'] = eval_case['goal']
            reference_world.reset(seed=eval_case['seed'], options=options)
            frames = [
                np.asarray(reference_world.infos['pixels'][0, -1]).copy()
            ]
            env = reference_world.envs.envs[0].unwrapped
            capture = getattr(env, 'capture_evaluation_goal', None)
            if capture is None:
                raise TypeError(
                    f'{type(env).__name__} must implement '
                    'capture_evaluation_goal()'
                )
            states = [capture()]
            for action in eval_case['witness_actions']:
                _, _, _, _, reference_world.infos = (
                    reference_world.envs.step(np.asarray(action)[None, ...])
                )
                frames.append(
                    np.asarray(
                        reference_world.infos['pixels'][0, -1]
                    ).copy()
                )
                states.append(capture())
            trajectories.append(np.stack(frames))
            state_trajectories.append(states)
    finally:
        reference_world.close()
    return trajectories, state_trajectories


def sample_reference_subgoals(
    reference_videos, reference_states, num_subgoals
):
    """Sample evenly spaced non-initial frames, always including the final."""
    if num_subgoals < 1:
        raise ValueError('eval.generated.num_subgoals must be at least 1')
    sampled = []
    sampled_states = []
    for trajectory, states in zip(reference_videos, reference_states):
        if len(trajectory) < 2:
            raise ValueError(
                'Reference trajectory must contain at least 2 frames'
            )
        if num_subgoals > len(trajectory) - 1:
            raise ValueError(
                'num_subgoals cannot exceed the number of expert actions'
            )
        # Split the complete trajectory into equal segments and use each
        # segment endpoint. The initial frame is a boundary, not a subgoal.
        indices = np.linspace(
            0,
            len(trajectory) - 1,
            num=num_subgoals + 1,
            dtype=int,
        )[1:]
        sampled.append(np.stack([trajectory[i] for i in indices]))
        sampled_states.append([states[i] for i in indices])
    return sampled, sampled_states


def get_results_path(cfg):
    return Path(_eval_dir(cfg.policy, cfg.subdir))


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    eval_source = cfg.eval.get('source', 'dataset')
    if eval_source not in ('dataset', 'env', 'generated_env'):
        raise ValueError(
            "eval.source must be 'dataset', 'env', or 'generated_env'"
        )
    cfg.world.max_episode_steps = (
        cfg.eval.eval_budget
        if eval_source in ('env', 'generated_env')
        else 2 * cfg.eval.eval_budget
    )
    env_img_size = cfg.get('env_img_size', cfg.eval.img_size)
    world_kwargs = dict(cfg.world)
    world_kwargs['image_shape'] = (env_img_size, env_img_size)
    for wrappers_key in ('pre_wrappers', 'extra_wrappers'):
        wrapper_configs = world_kwargs.get(wrappers_key)
        if wrapper_configs is not None:
            world_kwargs[wrappers_key] = [
                hydra.utils.instantiate(wrapper_config)
                for wrapper_config in wrapper_configs
            ]
    world = swm.World(**world_kwargs)

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset_name = cfg.dataset.get('stats', cfg.eval.dataset_name)
    stats_dataset = (
        dataset
        if str(stats_dataset_name) == str(cfg.eval.dataset_name)
        else get_dataset(cfg, stats_dataset_name)
    )
    col_name = get_episode_index_column(dataset)
    episode_idx = np.asarray(
        stats_dataset.get_col_data(col_name)
    ).reshape(-1)
    ep_indices, _ = np.unique(episode_idx, return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')
    print("Policy:", policy)

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if hasattr(model, 'backbone') and hasattr(
            model.backbone, 'get_vision_features'
        ):
            model.is_video_encoder = True
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = get_results_path(cfg)

    if eval_source == 'dataset':
        (
            eval_episodes,
            eval_start_idx,
            eval_goal_offsets,
        ) = select_dataset_eval_rows(cfg, dataset, ep_indices)

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False) and eval_source == 'dataset':
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=(
                    eval_goal_offsets[:n]
                    if isinstance(eval_goal_offsets, np.ndarray)
                    else eval_goal_offsets
                ),
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=results_path,
            )
        print('Warmup done.')

    start_time = time.time()
    with autocast_ctx:
        if eval_source == 'env':
            metrics = world.evaluate(
                episodes=cfg.eval.num_eval,
                seed=cfg.seed,
                options=OmegaConf.to_container(
                    cfg.eval.get('env_options', {}), resolve=True
                ),
                video=results_path,
            )
        elif eval_source == 'generated_env':
            generated_cases = generate_env_eval_cases(cfg, world_kwargs)
            base_options = OmegaConf.to_container(
                cfg.eval.get('env_options', {}), resolve=True
            )
            generated_seeds = [
                eval_case['seed'] for eval_case in generated_cases
            ]
            env_options = []
            for eval_case in generated_cases:
                options = dict(base_options)
                options['evaluation_start'] = eval_case['start']
                options['evaluation_goal'] = eval_case['goal']
                env_options.append(options)
            reference_videos, reference_states = render_generated_references(
                world_kwargs, generated_cases, base_options
            )
            subgoal_images, subgoal_states = sample_reference_subgoals(
                reference_videos,
                reference_states,
                int(cfg.eval.generated.get('num_subgoals', 1)),
            )
            world.reset(seed=generated_seeds, options=env_options)
            metrics = world.evaluate(
                episodes=cfg.eval.num_eval,
                video=results_path,
                reset_mode='wait',
                reference_videos=reference_videos,
                subgoal_images=subgoal_images,
                subgoal_states=subgoal_states,
                subgoal_horizon=cfg.eval.eval_budget,
                subgoal_switch_mode=cfg.eval.generated.get(
                    'subgoal_switch_mode', 'reached'
                ),
            )
        else:
            metrics = world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist(),
                goal_offset=eval_goal_offsets,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist(),
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=results_path,
            )
    end_time = time.time()

    print(metrics)
    print(f'[eval] videos saved to {results_path.resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
