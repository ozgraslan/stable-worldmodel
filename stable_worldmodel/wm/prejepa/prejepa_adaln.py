import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class AdaLNPreJEPA(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        extra_encoders=None,
        decoder=None,
        history_size=3,
        num_pred=1,
        interpolate_pos_encoding=True,
        extra_fusion='feature',
        image_representation='patches',
    ):
        super().__init__()

        self.backbone = encoder
        self.predictor = predictor
        self.extra_encoders = extra_encoders or {}
        self.decoder = decoder
        self.history_size = history_size
        self.num_pred = num_pred

        self.interpolate_pos_encoding = interpolate_pos_encoding
        if extra_fusion not in {'feature', 'token'}:
            raise ValueError("extra_fusion must be either 'feature' or 'token'")
        self.extra_fusion = extra_fusion
        if image_representation not in {'cls', 'cls+reg', 'patches'}:
            raise ValueError(
                'image_representation must be one of '
                "'cls', 'cls+reg', or 'patches'"
            )
        self.image_representation = image_representation

    def encode(
        self,
        info,
        pixels_key='pixels',
        emb_keys=None,
        prefix=None,
        target='emb',
    ):
        assert target not in info, f'{target} key already in info_dict'
        emb_keys = self.extra_encoders.keys() if emb_keys is None else emb_keys
        prefix = prefix or ''

        with torch.no_grad():
            pixels_embed = self._encode_image(
                info[pixels_key].float()
            )  # (B, T, 3, H, W)

        # == improve the embedding
        n_patches = pixels_embed.shape[2]
        embedding = pixels_embed
        info[f'pixels_{target}'] = pixels_embed

        for key in emb_keys:
            extr_enc = self.extra_encoders[key]
            extra_input = info[f'{prefix}{key}'].float()  # (B, T, dim)
            extra_embed = extr_enc(
                extra_input
            )  # (B, T, dim) -> (B, T, emb_dim)
            info[f'{key}_{target}'] = extra_embed

            if key == 'action':
                continue

            if self.extra_fusion == 'token':
                # V-JEPA 2-AC style: one state token per timestep, alongside
                # (rather than feature-concatenated into) the visual patches.
                embedding = torch.cat(
                    [embedding, extra_embed.unsqueeze(2)], dim=2
                )
            else:
                # Legacy behavior: repeat state across patches and concatenate
                # it to every visual token's feature dimension.
                extra_tiled = repeat(
                    extra_embed.unsqueeze(2),
                    'b t 1 d -> b t p d',
                    p=n_patches,
                )
                embedding = torch.cat([embedding, extra_tiled], dim=3)

        info[target] = embedding  # (B, T, P, d)

        return info

    def _encode_image(self, pixels):
        # == pixels embedding
        B = pixels.shape[0]
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )

        if hasattr(self.backbone, 'get_vision_features'):
            # V-JEPA 2-AC applies the video-pretrained encoder to individual
            # images by repeating each image across one temporal tubelet.
            tubelet_size = getattr(self.backbone.config, 'tubelet_size', 2)
            pixel_clips = pixels.unsqueeze(1).repeat(
                1, tubelet_size, 1, 1, 1
            )
            pixels_embed = self.backbone.get_vision_features(
                pixel_clips
            )
        else:
            pixels_embed = self.backbone(pixels, **kwargs)

        if torch.is_tensor(pixels_embed):
            # V-JEPA 2 returns patch tokens directly as (B*T, P, D).
            pass
        elif hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            if self.image_representation == 'cls':
                # Keep the token dimension for the predictor's B,T,P,D API.
                pixels_embed = pixels_embed[:, :1, :]
            elif self.image_representation == 'cls+reg':
                num_register_tokens = getattr(
                    self.backbone.config, 'num_register_tokens', 0
                )
                pixels_embed = pixels_embed[
                    :, : 1 + num_register_tokens, :
                ]
            else:
                pixels_embed = pixels_embed[:, -256:, :]  # drop CLS/registers
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(
                1
            )  # (B*T, 1, emb_dim)

        pixels_embed = rearrange(
            pixels_embed.detach(), '(b t) p d -> b t p d', b=B
        )

        return pixels_embed

    def predict(self, embedding, condition=None):
        """predict next latent state
        Args:
            embedding: (B, T, P, d)
            condition: (B, T, d_cond) or (B, T, P, d_cond)
        Returns:
            preds: (B, T, P, d)
        """

        T, P = embedding.shape[1:3]
        embedding = rearrange(embedding, 'b t p d -> b (t p) d')
        if condition is not None:
            if condition.ndim == 3:
                condition = repeat(condition, 'b t d -> b (t p) d', p=P)
            elif condition.ndim == 4:
                condition = rearrange(condition, 'b t p d -> b (t p) d')
            else:
                raise ValueError(
                    'condition must have shape (B, T, D) or (B, T, P, D)'
                )
            preds = self.predictor(embedding, condition)
        else:
            preds = self.predictor(embedding)
        preds = rearrange(preds, 'b (t p) d -> b t p d', t=T)

        return preds

    def decode(self, info):
        assert 'pixels_emb' in info, 'pixels_emb not in info_dict'
        pixels_embed = info['pixels_emb']
        num_frames = pixels_embed.shape[1]

        pixels, diff = self.decoder(
            pixels_embed
        )  # (b*num_frames, 3, 224, 224)
        pixels = rearrange(pixels, '(b t) c h w -> b t c h w', t=num_frames)

        info['reconstructed_pixels'] = pixels
        info['reconstruction_diff'] = diff

        return info

    def split_embedding(self, embedding, extra_dims, pixel_tokens=None):
        split_embed = {}
        non_action_keys = [
            key for key in self.extra_encoders.keys() if key != 'action'
        ]

        if self.extra_fusion == 'token':
            if pixel_tokens is None:
                raise ValueError('pixel_tokens is required for token fusion')
            split_embed['pixels_emb'] = embedding[..., :pixel_tokens, :]
            for i, key in enumerate(non_action_keys):
                split_embed[f'{key}_emb'] = embedding[
                    ..., pixel_tokens + i, :
                ]
            return split_embed

        pixel_dim = embedding.shape[-1] - sum(extra_dims)

        # == pixels embedding
        split_embed['pixels_emb'] = embedding[..., :pixel_dim]

        # == extra embeddings
        start_dim = pixel_dim
        for i, key in enumerate(non_action_keys):
            dim = extra_dims[i]
            extra_emb = embedding[..., start_dim : start_dim + dim]
            split_embed[f'{key}_emb'] = extra_emb[
                :, :, :, 0
            ]  # all patches are the same
            start_dim += dim

        return split_embed

    def rollout(self, info, action_sequence):
        """Rollout the world model given an initial observation and a sequence of actions.

        Params:
        obs_start: n current observations (B, n, C, H, W)
        actions: current and predicted actions (B, n+t, action_dim)

        Returns:
        z_obs: dict with latent observations (B, n+t+1, n_patches, D)
        z: predicted latent states (B, n+t+1, n_patches, D)
        """

        assert 'pixels' in info, 'pixels not in info_dict'
        n_obs = info['pixels'].shape[2]
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        # == add action to info dict
        act_0 = action_sequence[:, :, :n_obs]
        info['action'] = act_0

        # check if we have already computed the initial embedding for this state
        if (
            hasattr(self, '_init_cached_info')
            and torch.equal(self._init_cached_info['id'], info['id'][:, 0])
            and torch.equal(
                self._init_cached_info['step_idx'], info['step_idx'][:, 0]
            )
        ):
            init_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._init_cached_info.items()
            }
        else:
            # prepare init_info_dict
            init_info_dict = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    init_info_dict[k] = info[k][:, 0]  # (B, 1, ...)

            init_info_dict = self.encode(
                init_info_dict,
                pixels_key='pixels',
                target='emb',
                emb_keys=emb_keys,
            )
            # repeat copy for each action candidate
            init_info_dict['emb'] = (
                init_info_dict['emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['emb'].ndim - 1)),
                )
                .clone()
            )
            init_info_dict['pixels_emb'] = (
                init_info_dict['pixels_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_sequence.shape[1],
                    *([-1] * (init_info_dict['pixels_emb'].ndim - 1)),
                )
            )

            for key in emb_keys:
                init_info_dict[f'{key}_emb'] = (
                    init_info_dict[f'{key}_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_sequence.shape[1],
                        *([-1] * (init_info_dict[f'{key}_emb'].ndim - 1)),
                    )
                )

            init_info_dict = {
                k: v.detach().clone() if torch.is_tensor(v) else v
                for k, v in init_info_dict.items()
            }
            self._init_cached_info = init_info_dict

        info['emb'] = init_info_dict['emb']
        info['pixels_emb'] = init_info_dict['pixels_emb']

        for key in emb_keys:
            info[f'{key}_emb'] = init_info_dict[f'{key}_emb']

        # number of step to predict
        act_pred = action_sequence[:, :, n_obs:]
        n_steps = act_pred.shape[2]

        # == initial embedding
        z = info['emb']
        B, N = z.shape[:2]

        if 'action' in self.extra_encoders:
            act_0_flat = rearrange(
                action_sequence[:, :, :n_obs], 'b n ... -> (b n) ...'
            )
            act_0_emb = self.extra_encoders['action'](act_0_flat)
            info['action_emb'] = rearrange(
                act_0_emb, '(b n) ... -> b n ...', b=B, n=N
            )

        # we flatten B and N to process all candidates in a single batch in the predictor
        z_flat = rearrange(z, 'b n ... -> (b n) ...').clone()
        act_pred_flat = rearrange(act_pred, 'b n ... -> (b n) ...')
        act_emb_flat = (
            rearrange(info['action_emb'], 'b n ... -> (b n) ...').clone()
            if 'action' in self.extra_encoders
            else None
        )

        for t in range(n_steps):
            # predict the next state
            pred_embed = self.predict(
                z_flat[:, -self.history_size :],
                (
                    act_emb_flat[:, -self.history_size :]
                    if act_emb_flat is not None
                    else None
                ),
            )[:, -1:]  # (B*N, 1, P, D)

            if act_emb_flat is not None:
                new_action = act_pred_flat[:, t : t + 1, :]
                new_action_emb = self.extra_encoders['action'](new_action)
                act_emb_flat = torch.cat(
                    [act_emb_flat, new_action_emb], dim=1
                )

            # append new embedding to the sequence
            z_flat = torch.cat([z_flat, pred_embed], dim=1)

        # predict the last state (n+t+1)
        pred_embed = self.predict(
            z_flat[:, -self.history_size :],
            (
                act_emb_flat[:, -self.history_size :]
                if act_emb_flat is not None
                else None
            ),
        )[:, -1:]  # (B, 1, P, D)
        z_flat = torch.cat([z_flat, pred_embed], dim=1)
        z = rearrange(z_flat, '(b n) ... -> b n ...', b=B, n=N)
        # == update info dict with predicted embeddings
        info['predicted_embedding'] = z

        extra_dims = []
        for key in emb_keys:
            if f'{key}_emb' not in info:
                raise ValueError(f'{key}_emb not in info dict')
            extra_dims.append(info[f'{key}_emb'].shape[-1])

        splitted_embed = self.split_embedding(
            z,
            extra_dims,
            pixel_tokens=info['pixels_emb'].shape[-2],
        )
        info.update({f'predicted_{k}': v for k, v in splitted_embed.items()})

        return info

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost for planning. Should be overridden for custom costs."""
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']
        cost = 0.0

        for key in emb_keys + ['pixels']:
            preds = info_dict[f'predicted_{key}_emb']
            goal = info_dict[f'{key}_goal_emb']
            cost = cost + F.mse_loss(
                preds[:, :, -1:], goal, reduction='none'
            ).mean(dim=tuple(range(2, preds.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert 'action' in info_dict, 'action key must be in info_dict'
        assert 'pixels' in info_dict, 'pixels key must be in info_dict'

        # == non action embeddings keys
        emb_keys = [k for k in self.extra_encoders.keys() if k != 'action']

        # == get the goal embedding

        # check if we have already computed the goal embedding for this goal
        if (
            hasattr(self, '_goal_cached_info')
            and torch.equal(
                self._goal_cached_info['id'], info_dict['id'][:, 0]
            )
            and torch.equal(
                self._goal_cached_info['step_idx'], info_dict['step_idx'][:, 0]
            )
        ):
            goal_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in self._goal_cached_info.items()
            }

        else:
            # prepare goal_info_dict
            goal_info_dict = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    # goal is the same across samples so we will only embed it once
                    goal_info_dict[k] = info_dict[k][:, 0]  # (B, ...)
            goal_info_dict = self.encode(
                goal_info_dict,
                target='goal_emb',
                pixels_key='goal',
                prefix='goal_',
                emb_keys=emb_keys,
            )

            goal_info_dict['goal_emb'] = (
                goal_info_dict['goal_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_candidates.shape[1],
                    *([-1] * (goal_info_dict['goal_emb'].ndim - 1)),
                )
            )

            goal_info_dict['pixels_goal_emb'] = (
                goal_info_dict['pixels_goal_emb']
                .unsqueeze(1)
                .expand(
                    -1,
                    action_candidates.shape[1],
                    *([-1] * (goal_info_dict['pixels_goal_emb'].ndim - 1)),
                )
            )

            for key in emb_keys:
                goal_info_dict[f'{key}_goal_emb'] = (
                    goal_info_dict[f'{key}_goal_emb']
                    .unsqueeze(1)
                    .expand(
                        -1,
                        action_candidates.shape[1],
                        *([-1] * (goal_info_dict[f'{key}_goal_emb'].ndim - 1)),
                    )
                )

            goal_info_dict = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in goal_info_dict.items()
            }
            self._goal_cached_info = goal_info_dict

        info_dict['goal_emb'] = goal_info_dict['goal_emb']
        info_dict['pixels_goal_emb'] = goal_info_dict['pixels_goal_emb']

        for key in emb_keys:
            info_dict[f'{key}_goal_emb'] = goal_info_dict[f'{key}_goal_emb']

        # == run world model
        info_dict = self.rollout(info_dict, action_candidates)

        # cost = 0.0

        # for key in emb_keys + ["pixels"]:
        #     preds = info_dict[f"predicted_{key}_embed"]
        #     goal = info_dict[f"{key}_goal_embed"]
        #     cost = cost + F.mse_loss(preds[:, :, -1:], goal, reduction="none").mean(dim=tuple(range(2, preds.ndim)))

        return self.criterion(info_dict, action_candidates)


__all__ = ['AdaLNPreJEPA']
