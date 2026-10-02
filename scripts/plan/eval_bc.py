"""Evaluate feed-forward policies using SWM's native World.evaluate path."""

import json
import os
import time
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

import stable_worldmodel as swm
from scripts.plan.eval_wm import evaluation_kwargs, get_dataset, img_transform
from stable_worldmodel.data.normalization import ZScoreScaler
from stable_worldmodel.wm.bc import StateBC


def checkpoint_dir(cfg):
    """Locate metadata beside a native SWM export (or cached HF model)."""
    root = swm.data.utils.get_cache_dir(
        cfg.get('cache_dir'), sub_folder='checkpoints'
    )
    path = root / cfg.policy
    if path.suffix == '.pt':
        return path.parent
    if path.is_dir():
        return path
    return root / f'models--{cfg.policy.replace("/", "--")}'


def state_bc_process(cfg, model):
    """Restore SWM scalers fitted on BC training episodes; never refit at eval."""
    path = cfg.get('normalization_path')
    path = Path(path) if path else checkpoint_dir(cfg) / 'normalization.json'
    with path.open() as file:
        stats = json.load(file)
    process = {}
    for key in (*model.state_columns, 'action'):
        value = stats[key]
        mean, std = np.asarray(value['mean']), np.asarray(value['std'])
        if (
            mean.ndim != 1
            or mean.shape != std.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or (std < 0).any()
        ):
            raise ValueError(f'Invalid normalization statistics for {key}')
        process[key] = ZScoreScaler(**value)
    if (
        sum(len(process[key].mean) for key in model.state_columns)
        != model.state_dim
        or len(process['action'].mean) != model.action_dim
    ):
        raise ValueError('Normalization dimensions do not match BC checkpoint')
    return process


def make_policy(cfg, dataset=None):
    if cfg.policy == 'random':
        return swm.policy.RandomPolicy(seed=cfg.seed)
    model = swm.wm.utils.load_pretrained(
        cfg.policy, cache_dir=cfg.get('cache_dir')
    )
    model = model.to(cfg.get('device', 'cuda')).eval()
    model.requires_grad_(False)
    if isinstance(model, StateBC):
        return swm.policy.FeedForwardPolicy(
            model=model,
            process=state_bc_process(cfg, model),
            history_len=model.history_size,
            action_chunk_size=model.frameskip,
            history_keys=model.state_columns,
            require_goal=False,
            clip_actions=True,
        )
    # Preserve visual GCBC preprocessing for existing checkpoints.
    if dataset is None:
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = {}
    for col in ('action', 'proprio'):
        if col == 'proprio' and col not in dataset.column_names:
            continue
        values = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(
            values[np.isfinite(values).all(axis=1)]
        )
        if col != 'action':
            process[f'goal_{col}'] = process[col]
    transform = {key: img_transform(cfg) for key in ('pixels', 'goal')}
    return swm.policy.FeedForwardPolicy(
        model, process=process, transform=transform
    )


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Reuse world-model evaluation resets, episode selection, and metrics."""
    dataset = None
    if cfg.eval.get('source', 'dataset') == 'dataset':
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
    policy = make_policy(cfg, dataset)
    image_size = cfg.eval.get('img_size', 224)
    if (
        OmegaConf.is_missing(cfg.world, 'max_episode_steps')
        or cfg.world.get('max_episode_steps') is None
    ):
        cfg.world.max_episode_steps = cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(image_size, image_size))
    base = (
        swm.data.utils.get_cache_dir(
            cfg.get('cache_dir'), sub_folder='checkpoints'
        )
        if cfg.policy == 'random'
        else checkpoint_dir(cfg)
    )
    results_path = base / 'evals' / cfg.get('subdir', 'feed_forward')
    results_path.mkdir(parents=True, exist_ok=True)
    try:
        world.set_policy(policy)
        start_time = time.time()
        metrics = world.evaluate(
            **evaluation_kwargs(cfg, dataset),
            video=results_path if cfg.eval.get('save_video', True) else None,
        )
        elapsed = time.time() - start_time
    finally:
        world.close()
    with (results_path / cfg.output.filename).open('a') as file:
        file.write('\n==== CONFIG ====\n')
        file.write(OmegaConf.to_yaml(cfg))
        file.write(f'\n==== RESULTS ====\nmetrics: {metrics}\n')
        file.write(f'evaluation_time: {elapsed} seconds\n')
    print(metrics)
    print(f'[eval] results saved to {results_path}')
    return metrics


if __name__ == '__main__':
    run()
