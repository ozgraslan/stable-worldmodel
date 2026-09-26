import os
import copy
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def _vit_embed_dim(scale: str) -> int:
    dims = {
        'tiny': 192,
        'small': 384,
        'base': 768,
        'large': 1024,
        'huge': 1280,
    }
    if scale not in dims:
        raise ValueError(
            f"Unknown ViT encoder scale '{scale}'. "
            f"Expected one of {sorted(dims)}."
        )
    return dims[scale]


if not OmegaConf.has_resolver('vit_embed_dim'):
    OmegaConf.register_new_resolver('vit_embed_dim', _vit_embed_dim)


def get_img_preprocessor(
    source: str,
    target: str,
    img_size: int = 224,
    augmentation=None,
):
    imagenet_stats = dt.dataset_stats.ImageNet
    stats = dict(imagenet_stats)
    mean = stats.pop('mean')
    std = stats.pop('std')
    to_image = dt.transforms.ToImage(
        **stats, source=source, target=target
    )

    transforms = [to_image]
    crop = {}
    if augmentation and augmentation.get('enabled', False):
        crop = augmentation.get('random_resized_crop', {})
    if crop.get('enabled', False):
        transforms.append(
            dt.transforms.RandomResizedCrop(
                img_size,
                scale=tuple(crop.get('scale', (0.8, 1.0))),
                ratio=tuple(crop.get('ratio', (1.0, 1.0))),
                source=target,
                target=target,
            )
        )
    else:
        transforms.append(
            dt.transforms.Resize(img_size, source=target, target=target)
        )

    transforms.append(
        dt.transforms.WrapTorchTransform(
            v2.Normalize(mean=mean, std=std),
            source=target,
            target=target,
        )
    )
    return dt.transforms.Compose(*transforms)


def build_transforms(cfg, dataset):
    base_transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]
    train_transforms = [
        get_img_preprocessor(
            source='pixels',
            target='pixels',
            img_size=cfg.img_size,
            augmentation=cfg.get('augmentation'),
        )
    ]

    for col in cfg.data.dataset.keys_to_load:
        if col.startswith('pixels'):
            continue

        normalizer = get_column_normalizer(dataset, col, col)
        base_transforms.append(normalizer)
        train_transforms.append(normalizer)

    return (
        spt.data.transforms.Compose(*train_transforms),
        spt.data.transforms.Compose(*base_transforms),
    )


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm_aug')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cfg_cache_dir = dataset_cfg.pop('cache_dir', None)
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', cfg_cache_dir)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    train_transform, val_transform = build_transforms(cfg, dataset)

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        if 'proprio_encoder' in cfg.model:
            cfg.model.proprio_encoder.input_dim = dataset.get_dim('proprio')

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train_dataset = copy.copy(dataset)
    val_dataset = copy.copy(dataset)
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    train_set = spt.data.Subset(train_dataset, train_set.indices)
    val_set = spt.data.Subset(val_dataset, val_set.indices)

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

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

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=run_id or cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
