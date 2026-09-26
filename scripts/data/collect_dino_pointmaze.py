"""Collect expert trajectories from the DINO-WM PointMaze task.

Example:
    python scripts/data/collect_dino_pointmaze.py \
        --episodes 10 --num-envs 4 --max-episode-steps 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect a Stable WorldModel DINO PointMaze dataset.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'Output dataset path. Defaults to <SWM cache>/datasets/'
            'dino_pointmaze_expert.lance for Lance or '
            'dino_pointmaze_expert_video for video.'
        ),
    )
    parser.add_argument(
        '--format',
        choices=('lance', 'video'),
        default='lance',
        help=(
            "Dataset format. 'video' writes one MP4 per episode together "
            'with NPZ metadata.'
        ),
    )
    parser.add_argument(
        '--policy',
        choices=('expert', 'random'),
        default='expert',
        help='Policy used to collect trajectories.',
    )
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--max-episode-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kp', type=float, default=1.0)
    parser.add_argument('--kd', type=float, default=0.2)
    parser.add_argument('--action-noise', type=float, default=0.0)
    parser.add_argument(
        '--camera-distance',
        type=float,
        default=5.0,
        help='Distance of the centered top-down camera.',
    )
    parser.add_argument(
        '--camera-azimuth',
        type=float,
        default=180.0,
        help='Top-down camera rotation in degrees.',
    )
    parser.add_argument(
        '--success-threshold',
        type=float,
        default=0.2,
        help='Position distance required for success and termination.',
    )
    parser.add_argument(
        '--show-target',
        action='store_true',
        help='Render the red target marker (hidden by default).',
    )
    # parser.add_argument(
    #     '--mujoco-gl',
    #     choices=('egl', 'glfw', 'osmesa'),
    #     default=None,
    #     help='MuJoCo rendering backend (for example, egl on a headless GPU).',
    # )
    parser.add_argument(
        '--no-progress', action='store_true', help='Disable the progress bar.'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError('--episodes must be positive')
    if args.num_envs <= 0:
        raise ValueError('--num-envs must be positive')
    if args.max_episode_steps <= 0:
        raise ValueError('--max-episode-steps must be positive')
    if args.success_threshold <= 0:
        raise ValueError('--success-threshold must be positive')
    if args.camera_distance <= 0:
        raise ValueError('--camera-distance must be positive')
    # if args.mujoco_gl is not None:
    #     os.environ['MUJOCO_GL'] = args.mujoco_gl

    # Import after selecting MUJOCO_GL, since MuJoCo reads it at import time.
    import stable_worldmodel as swm
    from stable_worldmodel.envs.dino_pointmaze import ExpertPolicy
    from stable_worldmodel.policy import RandomPolicy

    output = args.output
    if output is None:
        suffix = '.lance' if args.format == 'lance' else '_video'
        name = f'dino_pointmaze_{args.policy}{suffix}'
        output = Path(swm.data.utils.get_cache_dir()) / 'datasets' / name

    world = swm.World(
        'swm/DINOPointMaze-v0',
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=args.max_episode_steps,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        success_threshold=args.success_threshold,
        show_target=args.show_target,
    )
    try:
        policy = (
            ExpertPolicy(
                kp=args.kp,
                kd=args.kd,
                action_noise=args.action_noise,
                seed=args.seed,
            )
            if args.policy == 'expert'
            else RandomPolicy(seed=args.seed)
        )
        world.set_policy(policy)
        print(
            f'Collecting {args.episodes} {args.policy} episodes into {output}'
        )
        world.collect(
            path=output,
            episodes=args.episodes,
            seed=args.seed,
            format=args.format,
            progress=not args.no_progress,
        )
    finally:
        world.close()

    print(f'Finished: {output}')


if __name__ == '__main__':
    main()
