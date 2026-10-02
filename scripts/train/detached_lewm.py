"""Train LeWM with detached autoregressive prefixes and endpoint supervision.

Run from the repository root with ``python -m scripts.train.detached_lewm``.
Existing models, training functions, and planning interfaces are unchanged.
"""

import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

import stable_worldmodel as swm
from scripts.train.lewm import SaveCkptCallback, get_img_preprocessor
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.lewm.detached_rollout import detached_rollout
from stable_worldmodel.wm.loss import SIGReg


def detached_lejepa_forward(self, batch, stage, cfg):
    """Learn dynamics from a detached rollout; learn encodings separately."""
    context_len = cfg.wm.history_size
    max_horizon = cfg.rollout.horizon
    if cfg.wm.num_preds != 1:
        raise ValueError('Detached rollout requires wm.num_preds=1.')
    if context_len < 1 or max_horizon < 1:
        raise ValueError('History size and rollout horizon must be positive.')
    if cfg.rollout.weight < 0 or cfg.rollout.teacher_weight < 0:
        raise ValueError('Loss weights must be nonnegative.')

    horizon = max_horizon
    if stage == 'train' and cfg.rollout.sample_horizon:
        horizon = int(torch.randint(1, max_horizon + 1, ()).item())

    # Preserve the caller's batch and leave observation encoding independent
    # of action encoding, which is recomputed on the final rollout window.
    actions = torch.nan_to_num(batch['action'], nan=0.0)
    observations = {
        key: value for key, value in batch.items() if key != 'action'
    }
    train_encoder = cfg.rollout.teacher_weight or cfg.loss.sigreg.weight
    with nullcontext() if train_encoder else torch.no_grad():
        output = self.model.encode(observations)
    emb = output['emb']
    if emb.size(1) < context_len + max_horizon:
        raise ValueError(
            'Dataset needs history_size + rollout.horizon frames.'
        )
    predictions = detached_rollout(
        self.model,
        emb[:, :context_len],
        actions[:, : context_len - 1 + horizon],
        horizon,
        history_size=context_len,
    )
    targets = emb[:, context_len : context_len + horizon].detach()
    step_errors = (predictions - targets).square().mean(dim=(0, 2))
    output['rollout_loss'] = step_errors[-1]

    teacher_loss = emb.new_zeros(())
    if cfg.rollout.teacher_weight:
        teacher_pred = self.model.predict(
            emb[:, :context_len],
            self.model.action_encoder(actions[:, :context_len]),
        )
        teacher_loss = (
            (teacher_pred - emb[:, 1 : context_len + 1]).square().mean()
        )
    output['teacher_loss'] = teacher_loss
    output['sigreg_loss'] = (
        self.sigreg(emb.transpose(0, 1))
        if cfg.loss.sigreg.weight
        else emb.new_zeros(())
    )
    output['loss'] = (
        cfg.rollout.teacher_weight * teacher_loss
        + cfg.rollout.weight * output['rollout_loss']
        + cfg.loss.sigreg.weight * output['sigreg_loss']
    )
    metrics = {
        f'{stage}/{key}': value.detach()
        for key, value in output.items()
        if key.endswith('loss')
    }
    metrics[f'{stage}/rollout_horizon'] = float(horizon)
    if stage != 'train':
        metrics[f'{stage}/rollout_mean_error'] = step_errors.mean().detach()
        metrics[f'{stage}/embedding_std'] = (
            emb.detach().flatten(0, 1).std(dim=0, unbiased=False).mean()
        )
        for index, error in enumerate(step_errors):
            metrics[f'{stage}/rollout_error_{index + 1}'] = error.detach()
    self.log_dict(
        metrics,
        on_step=stage == 'train',
        on_epoch=True,
        sync_dist=True,
        batch_size=emb.size(0),
    )
    return output


def split_rollout_episodes(dataset, train_fraction, generator):
    """Keep overlapping clips from one episode in the same partition."""
    if not 0 < train_fraction < 1:
        raise ValueError('train_split must be between zero and one.')
    episodes = sorted({episode for episode, _ in dataset.clip_indices})
    if len(episodes) < 2:
        raise ValueError('Need at least two episodes with full rollout clips.')
    order = torch.randperm(len(episodes), generator=generator).tolist()
    count = min(len(episodes) - 1, max(1, int(len(episodes) * train_fraction)))
    train_episodes = {episodes[index] for index in order[:count]}
    train_indices, val_indices = [], []
    for index, (episode, _) in enumerate(dataset.clip_indices):
        indices = train_indices if episode in train_episodes else val_indices
        indices.append(index)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@hydra.main(
    version_base=None, config_path='./config', config_name='detached_lewm'
)
def run(cfg):
    """Independent entry point using the existing dataset and model APIs."""
    pl.seed_everything(cfg.seed, workers=True)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop('name')
    cache_dir = dataset_cfg.pop('cache_dir', None) or os.environ.get(
        'LOCAL_DATASET_DIR'
    )
    dataset = swm.data.load_dataset(
        name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = []
    for column in cfg.data.dataset.keys_to_load:
        if column.startswith('pixels'):
            transforms.append(
                get_img_preprocessor(column, column, cfg.img_size)
            )
        else:
            transforms.append(column_normalizer(dataset, column, column))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = split_rollout_episodes(
        dataset, cfg.train_split, generator
    )
    train = DataLoader(train_set, **cfg.loader, generator=generator)
    val_options = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = DataLoader(val_set, **val_options)
    if not len(train):
        raise ValueError(
            'No training batches; reduce batch_size or drop_last.'
        )

    module = spt.Module(
        model=hydra.utils.instantiate(cfg.model),
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(detached_lejepa_forward, cfg=cfg),
        optim={
            'model_opt': {
                'modules': 'model',
                'optimizer': dict(cfg.optimizer),
            }
        },
    )
    run_dir = (
        Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
        / cfg.subdir
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')
    logger = False
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(cfg.output_model_name, cfg.model)],
        logger=logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        ckpt_path=cfg.resume_from,
    )
    manager()


if __name__ == '__main__':
    run()
