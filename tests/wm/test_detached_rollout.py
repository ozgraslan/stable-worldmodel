"""Forward parity and strict gradient boundaries for detached rollouts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_worldmodel.wm.lewm import LeWM, StateLeWM
from stable_worldmodel.wm.lewm.detached_rollout import detached_rollout


class RecordingPredictor(nn.Module):
    def __init__(self, history_size):
        super().__init__()
        self.num_frames = history_size
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(self, embeddings, actions):
        result = self.scale * (embeddings + actions).cumsum(dim=1)
        self.calls.append((embeddings, actions, result))
        return result


def make_model(history_size):
    return LeWM(
        encoder=nn.Linear(2, 2),
        predictor=RecordingPredictor(history_size),
        action_encoder=nn.Linear(2, 2, bias=False),
    )


@pytest.mark.parametrize(
    'context_len,history_size,horizon',
    [(1, 3, 5), (3, 3, 1), (3, 3, 5), (2, 4, 3)],
)
@pytest.mark.parametrize('rollout_layernorm', [False, True])
def test_matches_planning(
    context_len, history_size, horizon, rollout_layernorm
):
    torch.manual_seed(4)
    model = make_model(history_size).eval()
    model.rollout_layernorm = rollout_layernorm
    initial = torch.randn(2, context_len, 2)
    actions = torch.randn(2, context_len - 1 + horizon, 2)
    actual = detached_rollout(model, initial, actions, horizon, history_size)
    info = {
        'pixels': torch.empty(2, 1, context_len, 3, 1, 1),
        'emb': initial[:, None],
        'action_history': actions[:, None, : context_len - 1],
    }
    expected = model.rollout(
        info, actions[:, None, context_len - 1 :], history_size
    )['predicted_emb'][:, 0, context_len:]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    'history_size,horizon', [(1, 3), (3, 2), (3, 5), (3, 1)]
)
def test_only_endpoint_computation_has_gradients(history_size, horizon):
    model = make_model(history_size)
    initial = torch.randn(2, history_size, 2, requires_grad=True)
    actions = torch.randn(2, history_size - 1 + horizon, 2, requires_grad=True)
    predictions = detached_rollout(model, initial, actions, horizon)
    predictions[:, -1].square().mean().backward()

    assert initial.grad is None
    for embeddings, encoded_actions, result in model.predictor.calls[:-1]:
        assert not embeddings.requires_grad
        assert not encoded_actions.requires_grad
        assert not result.requires_grad
    embeddings, encoded_actions, result = model.predictor.calls[-1]
    assert not embeddings.requires_grad
    assert encoded_actions.requires_grad and result.requires_grad
    assert model.predictor.scale.grad.abs() > 0
    assert model.action_encoder.weight.grad.abs().sum() > 0
    # Only actions in the final sliding context can receive direct gradients.
    assert torch.count_nonzero(actions.grad[:, : horizon - 1]) == 0
    assert model.encoder.weight.grad is None


def test_respects_outer_no_grad_and_model_mode():
    model = make_model(3).eval()
    with torch.no_grad():
        prediction = detached_rollout(
            model, torch.randn(2, 3, 2), torch.randn(2, 5, 2), 3
        )
    assert not prediction.requires_grad
    assert not model.training


def test_action_blocks_are_aligned_and_predictions_are_fed_back():
    model = make_model(2)
    initial = torch.randn(2, 2, 2)
    actions = torch.randn(2, 4, 2)
    prediction = detached_rollout(model, initial, actions, 3)
    for step, (context, encoded_actions, _) in enumerate(
        model.predictor.calls
    ):
        torch.testing.assert_close(
            encoded_actions, model.action_encoder(actions[:, step : step + 2])
        )
        if step:
            torch.testing.assert_close(context[:, -1], prediction[:, step - 1])


def make_training_config():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            'wm': {'history_size': 3, 'num_preds': 1},
            'rollout': {
                'horizon': 3,
                'sample_horizon': False,
                'weight': 1.0,
                'teacher_weight': 0.0,
            },
            'loss': {'sigreg': {'weight': 0.0}},
        }
    )


def test_rollout_loss_detaches_observations_and_target():
    from scripts.train.detached_lewm import detached_lejepa_forward

    model = StateLeWM(
        encoder=nn.Linear(2, 2),
        predictor=RecordingPredictor(3),
        action_encoder=nn.Linear(2, 2),
    )
    logged = {}
    module = SimpleNamespace(
        model=model,
        sigreg=lambda emb: emb.square().mean(),
        log_dict=lambda metrics, **kwargs: logged.update(metrics),
    )
    states = torch.randn(2, 6, 2, requires_grad=True)
    batch = {'state': states, 'action': torch.randn(2, 6, 2)}
    result = detached_lejepa_forward(
        module, batch, 'val', make_training_config()
    )
    result['loss'].backward()
    assert states.grad is None
    assert model.encoder.weight.grad is None
    assert model.predictor.scale.grad.abs() > 0
    assert model.action_encoder.weight.grad.abs().sum() > 0
    assert 'emb' not in batch
    assert 'val/rollout_error_3' in logged


def test_teacher_branch_still_trains_encoder():
    from scripts.train.detached_lewm import detached_lejepa_forward

    model = StateLeWM(
        encoder=nn.Linear(2, 2),
        predictor=RecordingPredictor(3),
        action_encoder=nn.Linear(2, 2),
    )
    module = SimpleNamespace(
        model=model,
        sigreg=lambda emb: emb.square().mean(),
        log_dict=lambda *args, **kwargs: None,
    )
    cfg = make_training_config()
    cfg.rollout.teacher_weight = 1.0
    result = detached_lejepa_forward(
        module,
        {'state': torch.randn(2, 6, 2), 'action': torch.randn(2, 6, 2)},
        'train',
        cfg,
    )
    result['loss'].backward()
    assert model.encoder.weight.grad.abs().sum() > 0


def test_rejects_misaligned_actions():
    with pytest.raises(ValueError, match='aligned action blocks'):
        detached_rollout(
            make_model(3), torch.randn(2, 3, 2), torch.randn(2, 3, 2), 3
        )


def test_episode_split_has_no_overlap():
    from scripts.train.detached_lewm import split_rollout_episodes

    dataset = SimpleNamespace(
        clip_indices=[(ep, start) for ep in range(4) for start in range(5)]
    )
    train, val = split_rollout_episodes(
        dataset, 0.75, torch.Generator().manual_seed(7)
    )
    train_episodes = {dataset.clip_indices[i][0] for i in train.indices}
    val_episodes = {dataset.clip_indices[i][0] for i in val.indices}
    assert not train_episodes & val_episodes
    assert len(train) + len(val) == 20


def test_cpu_training_exports_planning_compatible_model(tmp_path, monkeypatch):
    from pathlib import Path

    import hydra
    import lance
    import pyarrow as pa
    import stable_pretraining as spt

    from scripts.train.detached_lewm import run
    from stable_worldmodel.wm.utils import load_pretrained

    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    monkeypatch.setattr(spt.get_config(), 'cache_dir', str(tmp_path))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset_path = tmp_path / 'toy.lance'
        lance.write_dataset(
            pa.table(
                {
                    'episode_idx': [ep for ep in range(4) for _ in range(10)],
                    'step_idx': list(range(10)) * 4,
                    'action': [[float(i % 3) / 2] for i in range(40)],
                    'state': [[float(i), float(i % 3)] for i in range(40)],
                }
            ),
            str(dataset_path),
        )
        config_dir = (
            Path(__file__).resolve().parents[2] / 'scripts/train/config'
        )
        with hydra.initialize_config_dir(
            version_base=None, config_dir=str(config_dir)
        ):
            cfg = hydra.compose(
                config_name='detached_state_lewm',
                overrides=[
                    '+trainer.limit_train_batches=1',
                    '+trainer.limit_val_batches=1',
                    '+trainer.enable_progress_bar=false',
                    '+trainer.enable_model_summary=false',
                ],
            )
        cfg.subdir = 'detached_test'
        cfg.output_model_name = 'detached_test'
        cfg.data.state_columns = ['state']
        cfg.data.state_dim = 2
        cfg.data.dataset.name = str(dataset_path)
        cfg.data.dataset.frameskip = 2
        cfg.wm.history_size = 2
        cfg.rollout.horizon = 2
        cfg.rollout.sample_horizon = False
        cfg.embed_dim = 4
        cfg.model.encoder.hidden_dim = 8
        cfg.model.predictor.depth = 1
        cfg.model.predictor.heads = 1
        cfg.model.predictor.dim_head = 4
        cfg.model.predictor.mlp_dim = 8
        cfg.model.projector.hidden_dim = 8
        cfg.model.pred_proj.hidden_dim = 8
        cfg.loss.sigreg.kwargs.knots = 3
        cfg.loss.sigreg.kwargs.num_proj = 4
        cfg.loader.batch_size = 2
        cfg.loader.num_workers = 0
        cfg.loader.persistent_workers = False
        cfg.loader.prefetch_factor = None
        cfg.loader.pin_memory = False
        cfg.trainer.accelerator = 'cpu'
        cfg.trainer.devices = 1
        cfg.trainer.precision = '32-true'
        cfg.trainer.max_epochs = 1
        run.__wrapped__(cfg)
        checkpoint = tmp_path / 'checkpoints/detached_test/weights_epoch_1.pt'
        model = load_pretrained(str(checkpoint)).eval()
        assert isinstance(model, StateLeWM)
        assert cfg.model.action_encoder.input_dim == 2
        info = {
            'state': torch.randn(1, 1, 2, 2),
            'action_history': torch.randn(1, 1, 1, 2),
        }
        output = model.rollout(info, torch.randn(1, 1, 2, 2))
        assert output['predicted_emb'].shape == (1, 1, 4, 4)
        assert torch.isfinite(output['predicted_emb']).all()
    finally:
        torch.set_num_threads(old_threads)
