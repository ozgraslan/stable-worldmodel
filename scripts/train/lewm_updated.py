import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
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


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoints by training step."""

    def __init__(self, run_name, cfg, step_interval: int = None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.step_interval = step_interval
        self.last_saved_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if not trainer.is_global_zero:
            return

        step = trainer.global_step
        if step <= self.last_saved_step:
            return

        is_interval = self.step_interval and step % self.step_interval == 0
        is_final = trainer.max_steps > 0 and step >= trainer.max_steps
        if is_interval or is_final:
            self._save(pl_module.model, step)
            self.last_saved_step = step

    def _save(self, model, step):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_step_{step}.pt',
        )


def get_max_training_steps(cfg) -> int:
    max_steps = cfg.trainer.get('max_steps', None)
    if max_steps is None or max_steps <= 0:
        raise ValueError(
            'Set trainer.max_steps to a positive integer. '
            'LEWM training is controlled by iterations instead of epochs.'
        )
    return int(max_steps)


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


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
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
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        if 'proprio_encoder' in cfg.model:
            cfg.model.proprio_encoder.input_dim = dataset.get_dim('proprio')

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = get_max_training_steps(cfg)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'step',
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
        step_interval=cfg.get('save_step_interval', None),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename='step_{step}',
        every_n_train_steps=cfg.get('save_step_interval', None),
        save_last=True,
        save_top_k=-1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, checkpoint_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / 'last.ckpt'
    legacy_ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=(
            ckpt_path
            if ckpt_path.exists()
            else legacy_ckpt_path
            if legacy_ckpt_path.exists()
            else None
        ),
    )

    manager()
    return


if __name__ == '__main__':
    run()
