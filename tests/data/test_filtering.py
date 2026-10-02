"""Success filtering preserves whole episodes and raw-step success flags."""

import numpy as np
import pytest

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.filtering import (
    filter_successful_episodes,
    successful_episode_indices,
)


class Demonstrations(Dataset):
    def __init__(self, flags, episode_flags=None, **kwargs):
        lengths = np.array([20, 20, 20])
        super().__init__(lengths, np.array([0, 20, 40]), **kwargs)
        self.flags = np.asarray(flags)
        self.episode_flags = episode_flags

    @property
    def column_names(self):
        return ['task_success']

    @property
    def episode_column_names(self):
        return ['success'] if self.episode_flags is not None else []

    def get_episode_data(self, episodes_idx=None):
        return {'success': self.episode_flags}

    def get_col_data(self, col):
        assert col == 'task_success'
        return self.flags

    def _load_slice(self, ep_idx, start, end):
        return {'episode': ep_idx, 'start': start}


def demonstrations(**kwargs):
    flags = np.zeros(60, dtype=bool)
    flags[7] = True  # success is not on a frameskip boundary or final step
    flags[59] = True
    return Demonstrations(flags, frameskip=5, num_steps=3, **kwargs)


def test_keeps_all_clips_of_successful_episodes_without_mutating():
    dataset = demonstrations()
    original = list(dataset.clip_indices)
    assert successful_episode_indices(dataset, 'task_success') == [0, 2]
    subset = filter_successful_episodes(dataset, 'task_success')
    assert len(subset) == 12  # six full windows per successful episode
    assert [subset[i]['episode'] for i in range(len(subset))] == [0] * 6 + [
        2
    ] * 6
    assert [subset[i]['start'] for i in range(6)] == list(range(6))
    assert dataset.clip_indices == original


def test_final_success_mode():
    assert successful_episode_indices(
        demonstrations(), 'task_success', 'final'
    ) == [2]


def test_episode_metadata_and_scalar_column_shapes():
    dataset = demonstrations(episode_flags=[False, True, False])
    assert successful_episode_indices(dataset) == [1]
    assert len(filter_successful_episodes(dataset)) == 6
    dataset.flags = dataset.flags[:, None]
    assert successful_episode_indices(dataset, 'task_success') == [0, 2]


def test_missing_success_and_invalid_mode():
    with pytest.raises(KeyError, match='Missing success field'):
        filter_successful_episodes(demonstrations())
    with pytest.raises(ValueError, match='success_mode'):
        successful_episode_indices(demonstrations(), 'task_success', 'bad')


@pytest.mark.parametrize('invalid', [np.nan, -1, 2, 'false', None])
def test_invalid_flags_are_not_treated_as_success(invalid):
    dataset = demonstrations()
    dataset.flags = dataset.flags.astype(
        object if invalid is None else type(invalid)
    )
    dataset.flags[0] = invalid
    with pytest.raises(ValueError, match='boolean or 0/1'):
        successful_episode_indices(dataset, 'task_success')


def test_rejects_malformed_flags():
    dataset = demonstrations()
    dataset.flags = np.zeros((60, 2))
    with pytest.raises(ValueError, match='one flag per row'):
        successful_episode_indices(dataset, 'task_success')


def test_no_successes_or_no_full_windows():
    dataset = demonstrations()
    dataset.flags[:] = False
    with pytest.raises(ValueError, match='No usable clips'):
        filter_successful_episodes(dataset, 'task_success')
    dataset.flags[0] = True
    dataset.clip_indices = []
    with pytest.raises(ValueError, match='No usable clips'):
        filter_successful_episodes(dataset, 'task_success')


class UnlabeledDemonstrations(Demonstrations):
    @property
    def column_names(self):
        return ['state', 'action']


def test_length_fallback_filters_whole_episodes(caplog):
    dataset = UnlabeledDemonstrations([], frameskip=5, num_steps=3)
    # Include empty, just-short-of-limit, at-limit and over-limit episodes.
    lengths = np.array([0, 19, 20, 21])
    Dataset.__init__(dataset, lengths, np.array([0, 0, 19, 39]), 5, 3)
    original = list(dataset.clip_indices)
    with caplog.at_level('INFO'):
        assert successful_episode_indices(dataset, max_episode_length=20) == [
            1
        ]
    assert 'success heuristic' in caplog.text
    subset = filter_successful_episodes(dataset, max_episode_length=20)
    assert len(subset) == 5
    assert all(subset[i]['episode'] == 1 for i in range(len(subset)))
    assert dataset.clip_indices == original


def test_explicit_flags_override_length_heuristic():
    dataset = demonstrations()
    # All episodes reach the length limit, but explicit success wins.
    assert successful_episode_indices(
        dataset, 'task_success', max_episode_length=20
    ) == [0, 2]
    dataset.flags[:] = False
    # All episodes are shorter than this limit, but explicit failure wins.
    assert (
        successful_episode_indices(
            dataset, 'task_success', max_episode_length=100
        )
        == []
    )
    dataset.episode_flags = [False, True, False]
    assert successful_episode_indices(dataset, max_episode_length=20) == [1]


def test_invalid_flags_do_not_trigger_length_fallback():
    dataset = demonstrations()
    dataset.flags = dataset.flags.astype(float)
    dataset.flags[0] = np.nan
    with pytest.raises(ValueError, match='boolean or 0/1'):
        successful_episode_indices(
            dataset, 'task_success', max_episode_length=100
        )


def test_no_short_episodes():
    dataset = UnlabeledDemonstrations([])
    with pytest.raises(ValueError, match='No usable clips'):
        filter_successful_episodes(dataset, max_episode_length=20)


@pytest.mark.parametrize('limit', [0, -1, 2.5, True])
def test_invalid_length_limit(limit):
    with pytest.raises(ValueError, match='positive integer'):
        successful_episode_indices(demonstrations(), max_episode_length=limit)
