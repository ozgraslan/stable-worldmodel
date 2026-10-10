"""Action-grid diagnostics using the same rollout as model-based planning."""

import torch
from torch import nn
from torch.nn import functional as F

from .evaluator import ShootingCostEvaluator


class EndpointEnergy(nn.Module):
    """Mean endpoint error, optionally layer-normalizing each latent vector.

    ``l1`` reproduces the V-JEPA2 notebook's energy definition; ``mse``
    measures the mean squared error used to train LeWM. Normalization is
    optional because native LeWM planning compares unnormalized latents.
    """

    def __init__(self, metric='l1', normalize=False):
        super().__init__()
        if metric not in ('l1', 'mse'):
            raise ValueError('metric must be l1 or mse')
        self.metric = metric
        self.normalize = normalize

    def forward(self, info):
        pred = info['predicted_emb'][:, :, -1]
        goal = info['goal_emb'][:, None, -1]
        if self.normalize:
            pred = F.layer_norm(pred, (pred.shape[-1],))
            goal = F.layer_norm(goal, (goal.shape[-1],))
        error = pred - goal
        error = error.abs() if self.metric == 'l1' else error.square()
        return error.flatten(2).mean(-1)


def xy_action_grid(samples=5, low=-0.075, high=0.075):
    """Return metric XY deltas ``(Y*X, 2)`` in heatmap row order.

    Rows vary Y, columns vary X. Units are meters per model prediction.
    """
    if samples < 2 or not low < high:
        raise ValueError('Need samples >= 2 and low < high')
    axis = torch.linspace(low, high, samples)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return axis, torch.stack([x.flatten(), y.flatten()], dim=-1)


@torch.inference_mode()
def evaluate_action_sequences(model, info, actions, objective=None, chunk=64):
    """Score model-space sequences ``(S, horizon, packed_action_dim)``.

    ``info`` has batch/time axes ``(1, T, ...)`` and contains current and
    ``goal*`` observations plus optional executed ``action_history`` blocks.
    Adds the candidate axis and uses SWM's goal encoding and rollout. Fresh
    tensors per chunk prevent model-side mutation from leaking across runs.
    """
    if actions.ndim != 3 or not actions.size(0) or chunk < 1:
        raise ValueError('Need nonempty (S, H, A) actions and chunk >= 1')
    if any(v.size(0) != 1 for v in info.values()):
        raise ValueError('Energy landscapes require a single context')
    evaluator = ShootingCostEvaluator(
        model, objective if objective is not None else EndpointEnergy()
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    base = {k: v.to(device=device, dtype=dtype) for k, v in info.items()}
    # Encode the goal once and reuse it across candidate chunks.
    goal_info = {k: v.unsqueeze(1) for k, v in base.items()}
    goal_emb = evaluator.encode_goal(model, goal_info)
    costs = []
    for candidate in actions.split(chunk):
        count = candidate.size(0)
        batch = {
            k: v.unsqueeze(1).expand(-1, count, *v.shape[1:]).clone()
            for k, v in base.items()
        }
        batch['goal_emb'] = goal_emb
        cost = evaluator.get_cost(
            batch, candidate.to(device=device, dtype=dtype).unsqueeze(0)
        )[0]
        if not torch.isfinite(cost).all():
            raise ValueError('Non-finite landscape energy; check inputs/stats')
        costs.append(cost.float().cpu())
    return torch.cat(costs)


class NotebookDynamics(nn.Module):
    """Normalize context and every prediction, reusing native LeWM rollout.

    The notebook feeds normalized predictions back into its predictor. Applying
    layer norm only when scoring changes multi-step behavior.
    """

    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize

    @property
    def predictor(self):
        return self.model.predictor

    @property
    def action_encoder(self):
        return self.model.action_encoder

    def observation(self, info):
        return self.model.observation(info)

    def encode(self, info):
        info = self.model.encode(info)
        if self.normalize:
            info['emb'] = F.layer_norm(info['emb'], (info['emb'].shape[-1],))
        return info

    def predict(self, info):
        info = self.model.predict(info)
        if self.normalize:
            prediction = info['preds']
            info['preds'] = F.layer_norm(prediction, (prediction.shape[-1],))
        return info

    def rollout(self, info, actions):
        from stable_worldmodel.wm.lewm.lewm import LeWM

        return LeWM.rollout(self, info, actions)


def pack_delta_sequences(deltas, scaler, action_block, controller_scale):
    """Map per-model-step metric XY deltas to trained controller chunks.

    A delta is split evenly across the physical steps of a chunk. Scaling
    back and summing those commands recovers exactly the sampled delta.
    """
    if action_block < 1 or controller_scale <= 0:
        raise ValueError('Need positive action_block and controller_scale')
    commands = deltas / (action_block * controller_scale)
    if (commands.abs() > 1 + 1e-6).any():
        raise ValueError('Sampled delta exceeds the controller chunk bounds')
    normalized = scaler.transform(commands)
    return (
        normalized.unsqueeze(-2)
        .expand(*normalized.shape[:-1], action_block, 2)
        .flatten(-2)
    )
