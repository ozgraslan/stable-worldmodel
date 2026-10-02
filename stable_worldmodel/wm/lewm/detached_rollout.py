"""Autoregressive LeWM predictions with gradients only on the final step."""

from contextlib import nullcontext

import torch

from .lewm import LeWM


def detached_rollout(
    model: LeWM,
    initial_emb: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
    history_size: int | None = None,
) -> torch.Tensor:
    """Return future embeddings (B, horizon, D) without backprop through time.

    ``initial_emb`` contains H observed embeddings. ``actions`` contains
    exactly H - 1 + horizon action blocks: action k leaves observation k.
    Blocks must already use the training normalization and frame grouping.

    All initial embeddings are detached. The first horizon - 1 steps run
    under no_grad; only the final step builds a graph, including its action
    encoder. The caller must detach supervision targets as well.

    This function preserves model train/eval mode. In training, prefix calls
    still use dropout and update BatchNorm statistics, as ordinary forwards
    do. Under an outer no_grad/inference_mode, the final step has no graph.
    """
    if initial_emb.ndim != 3 or actions.ndim != 3:
        raise ValueError('Expected (batch, time, features) tensors.')
    if horizon < 1:
        raise ValueError('horizon must be positive.')
    history_size = (
        getattr(model.predictor, 'num_frames', 3)
        if history_size is None
        else history_size
    )
    context_len = initial_emb.size(1)
    if context_len < 1 or history_size < 1:
        raise ValueError('Context and history_size must be nonempty.')
    if actions.shape[:2] != (
        initial_emb.size(0),
        context_len - 1 + horizon,
    ):
        raise ValueError('Expected H - 1 + horizon aligned action blocks.')

    history = list(initial_emb.detach().unbind(dim=1))
    predictions = []
    for step in range(horizon):
        end = context_len + step
        start = max(0, end - history_size)
        # nullcontext respects an enclosing validation no_grad context.
        grad_context = (
            nullcontext() if step == horizon - 1 else torch.no_grad()
        )
        with grad_context:
            context = torch.stack(history[start:], dim=1).detach()
            action_emb = model.action_encoder(actions[:, start:end])
            prediction = model.predict(context, action_emb)[:, -1]
            prediction = model._normalize_rollout_prediction(prediction)
        predictions.append(prediction)
        history.append(prediction.detach())
    return torch.stack(predictions, dim=1)
