"""LeWM over ViT patch tokens, with spatial and causal temporal attention."""

import torch
from einops import rearrange
from torch import nn

from .lewm import LeWM
from .module import Predictor


class LeWMPatch(LeWM):
    """Predict every projected ViT patch instead of the CLS embedding.

    Embeddings have shape ``(B, T, P, D)``. The shared projector processes
    both CLS and patch tokens; ``cls_emb`` retains the projected CLS embedding
    for SIGReg with shape ``(B, T, D)``. The shared LeWM rollout preserves the
    patch axis. Training, checkpointing, and goal scoring use the standard
    LEWM pipeline.
    """

    def encode(self, info):
        pixels = self.observation(info).to(
            next(self.encoder.parameters()).dtype
        )
        batch = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        tokens = output.last_hidden_state
        emb = self.projector(rearrange(tokens, 'bt p d -> (bt p) d'))
        emb = rearrange(emb, '(b t p) d -> b t p d', b=batch, p=tokens.size(1))
        info['cls_emb'] = emb[:, :, 0]
        info['emb'] = emb[:, :, 1:]
        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info


class PatchPredictor(Predictor):
    """Action-conditioned next-frame prediction for ``(B, T, P, D)``.

    Reuses LEWM's AdaLN blocks with one action embedding per frame broadcast
    to its patches. All patches in a frame attend to each other and all past
    frames, but never to future frames. Learned temporal and spatial position
    embeddings distinguish frames and patches. ``num_patches`` must match the
    encoder's image grid; shorter temporal windows are supported for rollout.
    """

    def __init__(self, *, num_patches, **kwargs):
        super().__init__(**kwargs)
        if num_patches < 1:
            raise ValueError('num_patches must be positive')
        self.num_patches = num_patches
        self.patch_pos_embedding = nn.Parameter(
            torch.randn(1, 1, num_patches, self.input_dim)
        )
        frame_ids = torch.arange(self.num_frames).repeat_interleave(
            num_patches
        )
        self.register_buffer(
            'attn_mask',
            frame_ids[:, None] >= frame_ids[None, :],
            persistent=False,
        )

    def forward(self, x, c):
        if x.ndim != 4:
            raise ValueError('patch embeddings must have shape (B, T, P, D)')
        batch, time, patches, dim = x.shape
        if not 1 <= time <= self.num_frames:
            raise ValueError(f'expected 1 to {self.num_frames} context frames')
        if patches != self.num_patches or dim != self.input_dim:
            raise ValueError(
                f'expected {self.num_patches} patches of dim {self.input_dim}, '
                f'got {patches} patches of dim {dim}'
            )
        if c.shape != (batch, time, self.input_dim):
            raise ValueError('action embeddings must have shape (B, T, D)')

        x = x + self.pos_embedding[:, :time, None] + self.patch_pos_embedding
        x = self.dropout(rearrange(x, 'b t p d -> b (t p) d'))
        c = c.unsqueeze(2).expand(-1, -1, patches, -1)
        c = rearrange(c, 'b t p d -> b (t p) d')
        length = time * patches
        x = self.transformer(x, c, attn_mask=self.attn_mask[:length, :length])
        return rearrange(x, 'b (t p) d -> b t p d', p=patches)


__all__ = ['LeWMPatch', 'PatchPredictor']
