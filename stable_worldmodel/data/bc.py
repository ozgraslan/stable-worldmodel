"""BC episode selection using SWM dataset access and normalization."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset

from .dataset import Dataset
from .filtering import successful_episode_indices
from .utils import column_normalizer

logger = logging.getLogger(__name__)


class _TrainingRows:
    """Expose only finite training rows to SWM's column normalizer."""

    def __init__(self, arrays, rows, finite):
        self.arrays, self.rows, self.finite = arrays, rows, finite

    def get_col_data(self, key):
        return self.arrays[key][self.rows[self.finite[key][self.rows]]]


@dataclass
class BCData:
    train: Subset
    val: Subset
    normalization: dict
    split: dict


def prepare_bc_data(
    dataset: Dataset,
    state_columns: list[str],
    *,
    train_fraction: float = 0.9,
    seed: int = 42,
    success_key: str = 'success',
    max_episode_length: int | None = None,
) -> BCData:
    """Filter, split episodes, then fit z-scores on training rows only.

    Invalid observation histories or target action chunks are excluded.
    Statistics use each finite raw training row once, avoiding overlap
    weighting. Episode ordinals and exclusion counts are saved for auditing.
    No sampled observation comes after the start of the target action chunk.
    """
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be strictly between 0 and 1')
    if not state_columns or 'action' in state_columns:
        raise ValueError('Provide state_columns without action')
    if len(set(state_columns)) != len(state_columns):
        raise ValueError('state_columns must be unique')
    episodes = successful_episode_indices(
        dataset, success_key, max_episode_length=max_episode_length
    )
    arrays = {}
    for key in [*state_columns, 'action']:
        values = np.asarray(dataset.get_col_data(key), dtype=np.float32)
        if values.ndim != 2 or len(values) != int(np.sum(dataset.lengths)):
            raise ValueError(f'{key} must have shape (total_raw_rows, D)')
        arrays[key] = values
    finite = {
        key: np.isfinite(value).all(axis=1) for key, value in arrays.items()
    }
    # Prefix sums make checking all five target actions constant-time.
    invalid_actions = np.concatenate(([0], np.cumsum(~finite['action'])))
    clips_by_episode = {ep: [] for ep in episodes}
    dropped = 0
    for clip_idx, (ep, local_start) in enumerate(dataset.clip_indices):
        if ep not in clips_by_episode:
            continue
        start = int(dataset.offsets[ep] + local_start)
        rows = start + np.arange(dataset.num_steps) * dataset.frameskip
        end = rows[-1] + dataset.frameskip
        valid = all(finite[key][rows].all() for key in state_columns)
        valid = valid and invalid_actions[end] == invalid_actions[rows[-1]]
        if valid:
            clips_by_episode[ep].append(clip_idx)
        else:
            dropped += 1
    usable = sorted(ep for ep, clips in clips_by_episode.items() if clips)
    if len(usable) < 2:
        raise ValueError(
            'Need at least two successful episodes with valid clips'
        )
    generator = torch.Generator().manual_seed(seed)
    shuffled = np.asarray(usable)[
        torch.randperm(len(usable), generator=generator).numpy()
    ]
    count = min(len(usable) - 1, max(1, int(len(usable) * train_fraction)))
    train_eps = sorted(shuffled[:count].tolist())
    val_eps = sorted(shuffled[count:].tolist())
    train_rows = np.concatenate(
        [
            np.arange(
                dataset.offsets[ep], dataset.offsets[ep] + dataset.lengths[ep]
            )
            for ep in train_eps
        ]
    )
    # Same transform API as scripts/train/lewm.py, fitted only on train rows.
    import stable_pretraining as spt

    stats_source = _TrainingRows(arrays, train_rows, finite)
    transforms, normalization = [], {}
    for key in arrays:
        transform = column_normalizer(stats_source, key, key)
        scaler = transform.lambd
        normalization[key] = {
            'mean': scaler.mean.reshape(-1).tolist(),
            'std': scaler.std.reshape(-1).tolist(),
            'eps': scaler.eps,
        }
        transforms.append(transform)
    dataset.transform = spt.data.transforms.Compose(*transforms)
    split = {
        'train_episodes': train_eps,
        'val_episodes': val_eps,
        'successful_episodes': episodes,
        'unusable_episodes': sorted(set(episodes) - set(usable)),
        'invalid_clips_dropped': dropped,
        'seed': seed,
    }

    def windows(ep_indices):
        clips = [clip for ep in ep_indices for clip in clips_by_episode[ep]]
        return Subset(dataset, clips)

    result = BCData(windows(train_eps), windows(val_eps), normalization, split)
    logger.info(
        'BC: %d train / %d validation episodes; %d / %d clips; '
        '%d invalid clips excluded',
        len(train_eps),
        len(val_eps),
        len(result.train),
        len(result.val),
        dropped,
    )
    return result
