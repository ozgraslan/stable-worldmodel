"""Select complete successful demonstrations before splitting BC data."""

import logging
from typing import Literal

import numpy as np
from torch.utils.data import Subset

from .dataset import Dataset

logger = logging.getLogger(__name__)


def successful_episode_indices(
    dataset: Dataset,
    success_key: str = 'success',
    success_mode: Literal['any', 'final'] = 'any',
    *,
    max_episode_length: int | None = None,
) -> list[int]:
    """Return successful episode ordinals, using raw success flags.

    Read either one flag per episode or one flag per raw environment step.
    ``any`` keeps episodes that succeeded at least once; ``final`` requires
    success at the last recorded step. This is independent of frameskip.
    A flag must be boolean or numeric 0/1; invalid flags raise.
    If the field is absent, ``max_episode_length`` enables a heuristic:
    nonempty episodes shorter than this limit are considered successful.
    The limit is in raw stored rows, before frameskip. Include the initial
    observation if collection stores it (e.g. 101 rows for 100 actions).
    This assumes collection stops on success; early failures or interrupted
    recordings cannot be distinguished without explicit success flags.
    Existing flags always take precedence, including all-false flags.
    ``success_key`` is explicit: termination is not a success signal.

    Use returned ordinals for episode-level splits and normalization, then
    select clips belonging to those episodes. These are positions in
    ``dataset.lengths``, not external episode IDs stored in a file.
    """
    if success_mode not in ('any', 'final'):
        raise ValueError("success_mode must be 'any' or 'final'")
    if max_episode_length is not None and (
        not isinstance(max_episode_length, int)
        or isinstance(max_episode_length, bool)
        or max_episode_length < 1
    ):
        raise ValueError('max_episode_length must be a positive integer')
    episode_scoped = success_key in dataset.episode_column_names
    if not episode_scoped and success_key not in dataset.column_names:
        if max_episode_length is None:
            raise KeyError(
                f'Missing success field {success_key!r}; include it in '
                'keys_to_load or set max_episode_length for length filtering'
            )
        lengths = np.asarray(dataset.lengths)
        selected = np.flatnonzero(
            (lengths > 0) & (lengths < max_episode_length)
        ).tolist()
        logger.info(
            'Success field %s absent: kept %d/%d episodes using length < %d '
            'raw rows as a success heuristic',
            success_key,
            len(selected),
            len(lengths),
            max_episode_length,
        )
        return selected
    if episode_scoped:
        values = dataset.get_episode_data()[success_key]
        expected = len(dataset.lengths)
    else:
        values = dataset.get_col_data(success_key)
        expected = int(np.sum(dataset.lengths))
    flags = np.asarray(values)
    if flags.shape not in ((expected,), (expected, 1)):
        raise ValueError(f'{success_key} must contain one flag per row')
    if flags.dtype.kind not in 'biuf' or not np.isin(flags, [0, 1]).all():
        raise ValueError(
            f'{success_key} must contain only boolean or 0/1 flags'
        )
    flags = flags.reshape(-1).astype(bool)
    if episode_scoped:
        selected = np.flatnonzero(flags).tolist()
    else:
        selected = []
        for ep, (offset, length) in enumerate(
            zip(dataset.offsets, dataset.lengths)
        ):
            episode_flags = flags[int(offset) : int(offset + length)]
            if not len(episode_flags):
                continue
            success = (
                episode_flags.any()
                if success_mode == 'any'
                else episode_flags[-1]
            )
            if success:
                selected.append(ep)
    logger.info(
        'Successful episodes: kept %d/%d using %s (%s)',
        len(selected),
        len(dataset.lengths),
        success_key,
        success_mode,
    )
    return selected


def filter_successful_episodes(
    dataset: Dataset,
    success_key: str = 'success',
    success_mode: Literal['any', 'final'] = 'any',
    *,
    max_episode_length: int | None = None,
) -> Subset:
    """Keep every existing clip from successful episodes without mutation.

    Uses explicit flags, falling back to the raw-row length heuristic only
    when the field is missing and ``max_episode_length`` is supplied.
    Raises if no successful episode has a full clip. The returned Subset
    retains original clip indices in ``indices``; split by their episode
    membership, not randomly by clip, to avoid train/validation leakage.
    """
    episodes = set(
        successful_episode_indices(
            dataset,
            success_key,
            success_mode,
            max_episode_length=max_episode_length,
        )
    )
    indices = [
        idx
        for idx, (ep, _) in enumerate(dataset.clip_indices)
        if ep in episodes
    ]
    if not indices:
        raise ValueError('No usable clips remain after success filtering')
    logger.info(
        'Successful episode clips: kept %d/%d', len(indices), len(dataset)
    )
    return Subset(dataset, indices)
