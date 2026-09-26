"""Projection-free LeWM operating on pre-final-LayerNorm ViT tokens."""

from einops import rearrange

from .lewm import LeWM


class LeWMPreLN(LeWM):
    """LeWM variant whose latent space is the ViT's raw final residual stream.

    Hugging Face ``ViTModel.last_hidden_state`` has already passed through the
    model-level final LayerNorm.  The final entry in ``hidden_states`` is the
    output of the last transformer block before that LayerNorm.  Using it
    keeps CLS and patch tokens in the same, unprojected coordinate system.
    """

    def __init__(self, encoder, predictor, action_encoder, **kwargs):
        # Accept and discard projection entries so this variant remains easy
        # to instantiate from configs derived from the standard LeWM config.
        kwargs.pop('projector', None)
        kwargs.pop('pred_proj', None)
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
        )

    def encode(self, info, return_patch_tokens: bool = False):
        """Encode observations using tokens before the ViT's final LayerNorm.

        Args:
            info: Dictionary containing ``pixels`` and optionally ``action``.
            return_patch_tokens: Store the pre-LayerNorm patch tokens in
                ``info['patch_tokens']``. Disabled by default to avoid keeping
                the large patch-token tensor during ordinary training.
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        output = self.encoder(
            pixels,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
        )
        tokens = output.hidden_states[-1]
        cls_tokens = tokens[:, 0]
        info['emb'] = rearrange(
            cls_tokens, '(b t) d -> b t d', b=batch_size
        )

        if return_patch_tokens:
            info['patch_tokens'] = rearrange(
                tokens[:, 1:], '(b t) p d -> b t p d', b=batch_size
            )

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info


__all__ = ['LeWMPreLN']
