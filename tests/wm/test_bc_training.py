"""Exercise the real Lance -> Hydra -> BC training -> checkpoint path."""

import importlib.util
import json
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch

from stable_worldmodel.wm.utils import load_pretrained

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('config_name', ['bc', 'bc_transformer'])
def test_lance_training_and_checkpoint(tmp_path, monkeypatch, config_name):
    lance = pytest.importorskip('lance')
    pa = pytest.importorskip('pyarrow')
    pytest.importorskip('lancedb')
    spec = importlib.util.spec_from_file_location(
        'bc_train_test', ROOT / 'scripts/train/bc.py'
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    rows = np.tile(np.arange(20, dtype=np.float32), 4)
    dataset_path = tmp_path / 'expert.lance'
    lance.write_dataset(
        pa.table(
            {
                'episode_idx': np.repeat([10, 20, 30, 40], 20),
                'step_idx': np.tile(np.arange(20), 4),
                'obs_a': rows[:, None].tolist(),
                'action': (rows[:, None] * 0.1).tolist(),
                'success': np.tile([False] * 19 + [True], 4),
            }
        ),
        str(dataset_path),
    )
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / 'scripts/train/config')
    ):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[
                f'data.dataset.name={dataset_path}',
                'data.state_dim=1',
                'data.state_columns=[obs_a]',
                'model.action_dim=1',
                'model.hidden_dims=[32]',
                'trainer.max_epochs=40',
                'loader.batch_size=64',
                'loader.drop_last=false',
                'num_workers=0',
                'loader.persistent_workers=false',
                'loader.prefetch_factor=null',
                'trainer.accelerator=cpu',
                'trainer.devices=1',
                'trainer.precision=32-true',
                '+trainer.enable_progress_bar=false',
                '+trainer.enable_model_summary=false',
                'wandb.enabled=false',
                'optimizer.lr=0.02',
                'subdir=bc_smoke',
            ]
            + (
                [
                    'model.transformer.embed_dim=16',
                    'model.transformer.depth=1',
                    'model.transformer.heads=2',
                    'model.transformer.dim_head=8',
                    'model.transformer.mlp_dim=32',
                    'model.transformer.dropout=0.0',
                ]
                if config_name == 'bc_transformer'
                else []
            ),
        )
    import stable_pretraining as spt

    monkeypatch.setattr(spt.get_config(), 'cache_dir', str(tmp_path / 'spt'))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = trainer.run.__wrapped__(cfg)
    finally:
        torch.set_num_threads(previous_threads)
    # Compare exported epochs through SWM's real normalized validation data.
    _, _, val = trainer.get_data(cfg)
    losses = []
    for epoch in (1, 40):
        checkpoint = load_pretrained(str(output / f'weights_epoch_{epoch}.pt'))
        batch = next(iter(val))
        with torch.no_grad():
            target = batch['action'][:, -1].reshape(-1, 5, 1)
            losses.append(
                torch.nn.functional.mse_loss(checkpoint(batch), target).item()
            )
    assert losses[-1] < losses[0] * 0.1
    stats = json.loads((output / 'normalization.json').read_text())
    split = json.loads((output / 'split.json').read_text())
    assert set(split['train_episodes']).isdisjoint(split['val_episodes'])
    assert set(split['train_episodes'] + split['val_episodes']) == {0, 1, 2, 3}
    assert set(stats) == {'obs_a', 'action'}
    restored = load_pretrained(str(output / 'weights_epoch_40.pt'))
    assert restored.history_size == 3 and restored.frameskip == 5
    assert restored.architecture == cfg.model.architecture
    assert restored({'obs_a': torch.zeros(2, 3, 1)}).shape == (2, 5, 1)
    assert (output / 'config.yaml').exists()
    cfg.seed += 1
    with pytest.raises(
        ValueError, match='different (split|normalization).json'
    ):
        trainer.run.__wrapped__(cfg)


@pytest.mark.parametrize('config_name', ['bc', 'bc_transformer'])
def test_bc_defaults_match_state_world_model_training(
    monkeypatch, tmp_path, config_name
):
    import stable_pretraining  # noqa: F401  # registers Hydra resolvers

    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / 'scripts/train/config')
    ):
        bc = hydra.compose(config_name=config_name)
        wm = hydra.compose(
            config_name='state_lewm',
            overrides=['data=maniskill_state_overhead'],
        )
    assert bc.seed == wm.seed
    assert bc.train_split == wm.train_split
    assert dict(bc.optimizer) == dict(wm.optimizer)
    assert dict(bc.loader) == dict(wm.loader)
    assert bc.wm.history_size == wm.wm.history_size
    assert bc.data.dataset.frameskip == wm.data.dataset.frameskip
    assert list(bc.state_columns) == list(wm.state_columns)
    for key in (
        'max_epochs',
        'max_steps',
        'devices',
        'accelerator',
        'precision',
        'gradient_clip_val',
    ):
        assert bc.trainer[key] == wm.trainer[key]

    if config_name == 'bc_transformer':
        assert bc.model.transformer.embed_dim == wm.embed_dim
        for key in ('depth', 'heads', 'dim_head', 'mlp_dim', 'dropout'):
            assert bc.model.transformer[key] == wm.model.predictor[key]
