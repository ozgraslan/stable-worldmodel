"""Behavior cloning from numeric observation histories."""

from collections.abc import Sequence

import torch
from torch import nn

from .lewm.module import Transformer


class StateBC(nn.Module):
    """Predict an action chunk from normalized state history.

    Inputs are dictionaries with ordered ``state_columns`` of shape
    ``(B, history_size, column_dim)``. Columns default to a single ``state``.
    History must already be sampled at ``frameskip`` intervals, oldest first.
    For history 3 and frameskip 5, inputs represent t-10, t-5, t and outputs
    represent actions t through t+4 (distinct commands, not action repeats).

    ``architecture='mlp'`` flattens history into an MLP (the default).
    ``architecture='transformer'`` embeds each state as a temporal token,
    adds learned positions, and applies LeWM's causal Transformer. The final
    token predicts the complete action chunk. ``transformer`` configures its
    embed_dim, depth, heads, dim_head, mlp_dim, and dropout; ``hidden_dims``
    applies only to the MLP. Neither architecture consumes a goal.

    Normalization, startup padding, and action execution belong to the data
    and policy adapters. This module neither mutates inputs nor buffers state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_columns: Sequence[str] = ('state',),
        history_size: int = 3,
        frameskip: int = 5,
        hidden_dims: Sequence[int] = (256, 256),
        architecture: str = 'mlp',
        transformer: dict | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ('state_dim', state_dim),
            ('action_dim', action_dim),
            ('history_size', history_size),
            ('frameskip', frameskip),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f'{name} must be a positive integer')
        columns = tuple(state_columns)
        if (
            not columns
            or any(not isinstance(key, str) or not key for key in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(
                'state_columns must contain unique nonempty names'
            )
        hidden_dims = tuple(hidden_dims)
        if any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
            for dim in hidden_dims
        ):
            raise ValueError('hidden_dims must contain positive integers')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_columns = columns
        self.history_size = history_size
        self.frameskip = frameskip

        if architecture not in ('mlp', 'transformer'):
            raise ValueError("architecture must be 'mlp' or 'transformer'")
        self.architecture = architecture
        if architecture == 'transformer':
            self.network = _StateTransformer(
                state_dim,
                history_size,
                frameskip * action_dim,
                **(transformer or {}),
            )
        else:
            layers = []
            in_dim = history_size * state_dim
            for out_dim in hidden_dims:
                layers.extend((nn.Linear(in_dim, out_dim), nn.GELU()))
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, frameskip * action_dim))
            self.network = nn.Sequential(*layers)

    def _inputs(self, info: dict[str, torch.Tensor]) -> torch.Tensor:
        observations = []
        batch_size = None
        for key in self.state_columns:
            obs = info[key]
            if obs.ndim != 3 or obs.shape[1] != self.history_size:
                raise ValueError(
                    f'{key} must have shape (B, {self.history_size}, D)'
                )
            if batch_size is not None and obs.shape[0] != batch_size:
                raise ValueError('All state columns must share a batch size')
            batch_size = obs.shape[0]
            observations.append(obs)
        state = torch.cat(observations, dim=-1)
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f'Expected {self.state_dim} state features, '
                f'got {state.shape[-1]}'
            )
        inputs = state.flatten(1)
        return inputs.to(dtype=next(self.network.parameters()).dtype)

    def forward(self, info: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return normalized actions with shape (B, frameskip, action_dim)."""
        actions = self.network(self._inputs(info))
        return actions.reshape(-1, self.frameskip, self.action_dim)

    def get_action(
        self,
        info: dict[str, torch.Tensor],
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the first action, or up to one predicted action chunk.

        Matches SWM's Actionable shape convention. Warm-start prefixes are
        unsupported because this policy does not roll out latent dynamics.
        Use ``horizon=frameskip`` to retrieve the entire chunk for execution.
        """
        if prefix_actions is not None:
            raise ValueError('StateBC does not support prefix_actions')
        if (
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or not 1 <= horizon <= self.frameskip
        ):
            raise ValueError(f'horizon must be between 1 and {self.frameskip}')
        actions = self(info)
        return actions[:, 0] if horizon == 1 else actions[:, :horizon]


class _StateTransformer(nn.Module):
    """Temporal action head built from the repository's LeWM Transformer."""

    def __init__(
        self,
        state_dim: int,
        history_size: int,
        output_dim: int,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ('embed_dim', embed_dim),
            ('depth', depth),
            ('heads', heads),
            ('dim_head', dim_head),
            ('mlp_dim', mlp_dim),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f'{name} must be a positive integer')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.state_dim = state_dim
        self.history_size = history_size
        self.state_encoder = nn.Linear(state_dim, embed_dim)
        self.pos_embedding = nn.Parameter(
            0.02 * torch.randn(1, history_size, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.action_head = nn.Linear(embed_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = inputs.reshape(-1, self.history_size, self.state_dim)
        tokens = self.state_encoder(state)
        tokens = self.dropout(tokens + self.pos_embedding.to(tokens.dtype))
        features = self.transformer(tokens)
        return self.action_head(features[:, -1])


__all__ = ['StateBC']
