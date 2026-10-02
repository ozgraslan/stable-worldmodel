"""LeWM's state encoder and input mapping; dynamics and costs come from SWM."""

import torch
from einops import rearrange

from .lewm import LeWM


class StateLeWM(LeWM):
    """LeWM over normalized numeric observations instead of image patches.

    ``state_columns`` lists ordered vector-valued inputs to concatenate along
    the feature axis. With no columns, read a single ``state`` tensor.
    Prediction and action-history rollout use the shared LeWM implementation.
    """

    def __init__(self, *args, state_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_columns = list(state_columns or [])

    def observation(self, info):
        if self.state_columns:
            return torch.cat([info[key] for key in self.state_columns], dim=-1)
        return info['state']

    def encode(self, info):
        """Encode SWM-normalized observation columns instead of image patches."""
        state = self.observation(info).to(
            next(self.encoder.parameters()).dtype
        )
        batch = state.size(0)
        emb = self.projector(
            self.encoder(rearrange(state, 'b t d -> (b t) d'))
        )
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=batch)
        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info


__all__ = ['StateLeWM']
