"""BC temporal alignment, leakage prevention, and invalid target handling."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from stable_worldmodel.data.bc import prepare_bc_data
from stable_worldmodel.data.dataset import Dataset


class NumericDemonstrations(Dataset):
    def __init__(self):
        super().__init__(np.full(4, 20), np.arange(4) * 20, 5, 3)
        rows = np.arange(80, dtype=np.float32)
        self.arrays = {
            'state': rows[:, None],
            'action': np.stack([rows, -rows], axis=1),
            'success': np.repeat([True, True, True, False], 20),
        }

    @property
    def column_names(self):
        return list(self.arrays)

    def get_col_data(self, col):
        return self.arrays[col]

    def _load_slice(self, ep_idx, start, end):
        offset = self.offsets[ep_idx]
        sample = {
            key: torch.from_numpy(values[offset + start : offset + end].copy())
            for key, values in self.arrays.items()
        }
        sample = {
            key: value if key == 'action' else value[:: self.frameskip]
            for key, value in sample.items()
        }
        return self.transform(sample) if self.transform else sample


def clip_pairs(data):
    return [data.dataset.clip_indices[index] for index in data.indices]


def test_split_alignment_and_training_only_stats():
    source = NumericDemonstrations()
    data = prepare_bc_data(source, ['state'], train_fraction=0.67, seed=42)
    train, val = data.split['train_episodes'], data.split['val_episodes']
    assert set(train).isdisjoint(val)
    assert set(train + val) == {0, 1, 2}
    expected_rows = np.concatenate(
        [np.arange(ep * 20, (ep + 1) * 20) for ep in train]
    )
    for key in ('state', 'action'):
        expected = source.arrays[key][expected_rows]
        np.testing.assert_allclose(
            data.normalization[key]['mean'], expected.mean(0)
        )
        np.testing.assert_allclose(
            data.normalization[key]['std'], expected.std(0)
        )
    assert isinstance(data.train, Subset)
    assert data.train.dataset is source
    ep, local_start = clip_pairs(data.train)[0]
    start = source.offsets[ep] + local_start
    sample = data.train[0]
    for key, rows in (
        ('state', [start, start + 5, start + 10]),
        ('action', np.arange(start + 10, start + 15)),
    ):
        stats = data.normalization[key]
        values = (
            sample[key][-1].reshape(5, -1) if key == 'action' else sample[key]
        )
        raw = values.numpy() * stats['std'] + stats['mean']
        np.testing.assert_allclose(raw, source.arrays[key][rows], atol=1e-5)
    again = prepare_bc_data(source, ['state'], train_fraction=0.67, seed=42)
    assert again.split == data.split


def test_validation_and_failed_episodes_cannot_change_statistics():
    source = NumericDemonstrations()
    baseline = prepare_bc_data(source, ['state'], seed=42)
    for ep in baseline.split['val_episodes'] + [3]:
        for key in ('state', 'action'):
            source.arrays[key][ep * 20 : (ep + 1) * 20] += 1e6
    changed = prepare_bc_data(source, ['state'], seed=42)
    assert changed.normalization == baseline.normalization


def test_invalid_last_action_drops_only_affected_chunk():
    source = NumericDemonstrations()
    source.arrays['action'][19] = np.nan
    data = prepare_bc_data(source, ['state'])
    all_clips = clip_pairs(data.train) + clip_pairs(data.val)
    assert (0, 5) not in all_clips
    assert (0, 4) in all_clips
    assert data.split['invalid_clips_dropped'] == 1
    for windows in (data.train, data.val):
        for sample in windows:
            assert all(
                np.isfinite(value.numpy()).all() for value in sample.values()
            )


def test_bad_state_and_insufficient_successes():
    source = NumericDemonstrations()
    source.arrays['state'][10] = np.inf
    data = prepare_bc_data(source, ['state'])
    assert (0, 0) not in clip_pairs(data.train) + clip_pairs(data.val)
    source.arrays['success'][20:] = False
    with pytest.raises(ValueError, match='at least two'):
        prepare_bc_data(source, ['state'])


def test_dataloader_preserves_native_action_block_layout():
    source = NumericDemonstrations()
    data = prepare_bc_data(source, ['state'])
    batch = next(iter(DataLoader(data.train, batch_size=2)))
    assert batch['state'].shape == (2, 3, 1)
    assert batch['action'].shape == (2, 3, 10)
    torch.testing.assert_close(batch['action'][0], data.train[0]['action'])
