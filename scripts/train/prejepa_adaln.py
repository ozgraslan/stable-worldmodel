import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from functools import partial
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from torch.nn import functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class SaveCkptCallback(Callback):
    """Callback to save model checkpoints by training step."""

    def __init__(self, run_name, cfg, step_interval=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.step_interval = step_interval
        self.last_saved_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
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


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(batch, target='emb')

    embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    action_embedding = (
        batch['action_emb'][:, : cfg.wm.history_size]
        if 'action_emb' in batch
        else None
    )
    pred_embedding = self.model.predict(embedding, action_embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    non_action_keys = [
        key for key in self.model.extra_encoders if key != 'action'
    ]
    if cfg.wm.get('extra_fusion', 'feature') == 'token':
        pixel_tokens = batch['pixels_emb'].size(-2)
        batch['pixels_loss'] = F.mse_loss(
            pred_embedding[..., :pixel_tokens, :],
            target_embedding[..., :pixel_tokens, :],
        )
        for i, key in enumerate(non_action_keys):
            token_idx = pixel_tokens + i
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., token_idx, :],
                target_embedding[..., token_idx, :].detach(),
            )
    else:
        pixels_dim = batch['pixels_emb'].size(-1)
        batch['pixels_loss'] = F.mse_loss(
            pred_embedding[..., :pixels_dim],
            target_embedding[..., :pixels_dim],
        )
        start = pixels_dim
        for key in non_action_keys:
            dim = batch[f'{key}_emb'].size(-1)
            lo, hi = start, start + dim
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
            start = hi

    batch['loss'] = F.mse_loss(pred_embedding, target_embedding.detach())

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


def get_max_training_steps(cfg) -> int:
    max_steps = cfg.trainer.get('max_steps', None)
    if max_steps is None or max_steps <= 0:
        raise ValueError(
            'Set trainer.max_steps to a positive integer. '
            'PreJEPA AdaLN training is controlled by iterations instead of epochs.'
        )
    return int(max_steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa_adaln')
def run(cfg):
    # --- Dataset ---

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
            source='pixels', target='pixels', img_size=cfg.image_size
        ),
    ]

    with open_dict(cfg) as cfg:
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.data.dataset.frameskip
            )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train_loader = DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    get_max_training_steps(cfg)

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    print(
        f'Loaded encoder: {encoder.__class__.__name__} with config: {encoder.config}'
    )
    print(
        f'Encoder has get_vision_features: {hasattr(encoder, "get_vision_features")}'
    )
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, 'hidden_sizes')
    embed_dim = (
        encoder.config.hidden_sizes[-1]
        if is_cnn
        else encoder.config.hidden_size
    )
    image_representation = cfg.wm.get('image_representation', 'patches')
    if is_cnn or image_representation == 'cls':
        num_patches = 1
    elif image_representation == 'cls+reg':
        num_patches = 1 + getattr(encoder.config, 'num_register_tokens', 0)
    else:
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
    non_action_keys = [
        key for key in cfg.wm.get('encoding', {}) if key != 'action'
    ]
    if cfg.wm.get('extra_fusion', 'feature') == 'feature':
        embed_dim += sum(cfg.wm.encoding[key] for key in non_action_keys)

    if cfg.wm.get('extra_fusion', 'feature') == 'token':
        num_patches += len(non_action_keys)

    predictor_condition_dim = (
        cfg.predictor.get('hidden_dim', None) or embed_dim
    )

    with open_dict(cfg):
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.condition_dim = predictor_condition_dim
        cfg.model.predictor.num_patches = num_patches
        extra_encoder_modules = {}
        for key in cfg.wm.get('encoding', {}):
            if key == 'action':
                extra_encoder_modules[key] = {
                    '_target_': 'stable_worldmodel.wm.lewm.module.Embedder',
                    'input_dim': cfg.extra_dims[key],
                    'emb_dim': predictor_condition_dim,
                }
            else:
                extra_encoder_modules[key] = {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': cfg.extra_dims[key],
                    'emb_dim': (
                        embed_dim
                        if cfg.wm.get('extra_fusion', 'feature') == 'token'
                        else int(cfg.wm.encoding[key])
                    ),
                }

        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': extra_encoder_modules,
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename='step_{step}',
        every_n_train_steps=cfg.get('save_step_interval', None),
        save_last=True,
        save_top_k=-1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=run_id or cfg.output_model_name,
                cfg=cfg.model,
                step_interval=cfg.get('save_step_interval', None),
            ),
            checkpoint_callback,
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / 'last.ckpt'
    legacy_ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=(
            ckpt_path
            if ckpt_path.exists()
            else legacy_ckpt_path
            if legacy_ckpt_path.exists()
            else None
        ),
    )
    manager()


if __name__ == '__main__':
    run()
