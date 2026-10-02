"""State BC using the same training infrastructure as scripts/train/lewm.py."""

import json
import logging
import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.nn import functional as F

import stable_worldmodel as swm
from scripts.train.lewm import SaveCkptCallback
from stable_worldmodel.data.bc import prepare_bc_data

logger = logging.getLogger(__name__)


def bc_forward(self, batch, stage):
    """Train on the last action block returned by SWM's dataset reader."""
    prediction = self.model(batch)
    target = batch['action'][:, -1].reshape(
        -1, self.model.frameskip, self.model.action_dim
    )
    if prediction.shape != target.shape:
        raise ValueError('Predicted and target action shapes differ')
    batch['action_loss'] = F.mse_loss(prediction, target)
    batch['loss'] = batch['action_loss']
    if not torch.isfinite(batch['loss']):
        raise ValueError('Nonfinite BC loss')
    self.log_dict(
        {f'{stage}/action_loss': batch['action_loss'].detach()},
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        batch_size=prediction.shape[0],
    )
    return batch


def get_data(cfg):
    """Use native readers/transforms; select episodes before normalization."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop('name')
    cache_dir = dataset_cfg.pop('cache_dir', None) or os.getenv(
        'LOCAL_DATASET_DIR'
    )
    # Inspect the schema without decoding images. Then load only configured
    # numeric columns, plus success when available, via the native reader.
    schema_cfg = {**dataset_cfg, 'keys_to_load': None, 'keys_to_cache': None}
    schema = swm.data.load_dataset(name, cache_dir=cache_dir, **schema_cfg)
    keys = list(dataset_cfg['keys_to_load'])
    if (
        cfg.filter.success_key in schema.column_names
        and cfg.filter.success_key not in keys
    ):
        keys.append(cfg.filter.success_key)
    dataset_cfg['keys_to_load'] = keys
    del schema
    dataset = swm.data.load_dataset(
        name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    if (dataset.num_steps, dataset.frameskip) != (
        cfg.model.history_size,
        cfg.model.frameskip,
    ):
        raise ValueError('Dataset history/frameskip must match model settings')
    data = prepare_bc_data(
        dataset,
        list(cfg.state_columns),
        train_fraction=cfg.train_split,
        seed=cfg.seed,
        **OmegaConf.to_container(cfg.filter),
    )
    state_dim = sum(
        len(data.normalization[key]['mean']) for key in cfg.state_columns
    )
    action_dim = len(data.normalization['action']['mean'])
    if (state_dim, action_dim) != (cfg.model.state_dim, cfg.model.action_dim):
        raise ValueError(
            'Configured state/action dimensions do not match dataset'
        )
    generator = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(
        data.train, **cfg.loader, generator=generator
    )
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = torch.utils.data.DataLoader(data.val, **val_cfg)
    if not len(train):
        raise ValueError(
            'No training batches; reduce batch size or disable drop_last'
        )
    return data, train, val


@hydra.main(version_base=None, config_path='./config', config_name='bc')
def run(cfg):
    if cfg.trainer.max_epochs < 1:
        raise ValueError('trainer.max_epochs must be positive')
    pl.seed_everything(cfg.seed, workers=True)
    data, train, val = get_data(cfg)
    model = hydra.utils.instantiate(cfg.model)
    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }
    module = spt.Module(model=model, forward=bc_forward, optim=optimizers)
    data_module = spt.data.DataModule(train=train, val=val)
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.subdir
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep normalization and episode selection beside native weight exports.
    for filename, value in (
        ('normalization.json', data.normalization),
        ('split.json', data.split),
    ):
        path = run_dir / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Existing BC run has different {filename}')
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    OmegaConf.save(cfg, run_dir / 'config.yaml', resolve=True)
    pl_logger = None
    if cfg.wandb.enabled:
        pl_logger = WandbLogger(**cfg.wandb.config)
        pl_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=str(run_dir), cfg=cfg.model, epoch_interval=1
            )
        ],
        num_sanity_val_steps=1,
        logger=pl_logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()
    return run_dir


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s | %(message)s'
    )
    run()
