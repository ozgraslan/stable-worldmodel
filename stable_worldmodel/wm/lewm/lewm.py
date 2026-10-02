import torch
from einops import rearrange
from torch import nn


class LeWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        rollout_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.rollout_layernorm = rollout_layernorm

    def observation(self, info):
        """Return model inputs with batch and time axes preserved."""
        return info['pixels']

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = self.observation(info).to(
            next(self.encoder.parameters()).dtype
        )
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, ..., D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        shape = preds.shape[:-1]
        preds = self.pred_proj(preds.reshape(-1, preds.size(-1)))
        preds = preds.reshape(*shape, -1)
        return preds

    def _normalize_rollout_prediction(self, prediction):
        # getattr keeps older whole-object checkpoints compatible.
        if getattr(self, 'rollout_layernorm', False):
            return nn.functional.layer_norm(prediction, (prediction.size(-1),))
        return prediction

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int | None = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, H, C, h, w) — H context frames (block timesteps)
        action_sequence: (B, S, T, action_dim) — strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) — executed action
            blocks between the context frames (required when H > 1)
         - S is the number of action plan samples
         - T is the planning horizon
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, ..., D);
        the first H entries are the encoded context frames.
        When ``rollout_layernorm`` is enabled, each predicted latent is
        normalized over its last feature axis without affine parameters,
        after ``pred_proj`` and before reuse in the next step. Context
        embeddings and ordinary ``predict`` calls are unchanged.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        H = self.observation(info).size(2)
        B, S, T = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        # action paired with context frame k is the block leaving it; the
        # current frame (k = H-1) pairs with the first candidate
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            initial_emb = _init['emb'].detach()
            info['emb'] = initial_emb.unsqueeze(1).expand(
                B, S, *initial_emb.shape[1:]
            )

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_past_flat = rearrange(act_past, 'b s ... -> (b s) ...')
        act_cand_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )  # (BS, H - 1 + T, A_emb); index k = block leaving frame k

        # rollout predictor autoregressively, one step per candidate
        # Each frame retains all latent axes, with its own grad_fn.
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors: (BS, ..., D)
        for t in range(T):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, ..., D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            prediction = self.predict(emb_trunc, act_trunc)[:, -1]
            emb_list.append(self._normalize_rollout_prediction(prediction))

        emb = torch.stack(emb_list, dim=1)  # (BS, H + T, ..., D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info


__all__ = ['LeWM']
