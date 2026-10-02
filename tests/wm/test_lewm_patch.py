"""Patch prediction, temporal causality, and LEWM pipeline integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.wm.lewm import LeWMPatch, PatchPredictor
from stable_worldmodel.wm.lewm.module import MLP
from stable_worldmodel.wm.loss import SIGReg


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.cls_proj = nn.Linear(8, 8)

    def forward(self, pixels, **kwargs):
        assert kwargs['interpolate_pos_encoding']
        patches = self.conv(pixels).flatten(2).transpose(1, 2)
        cls = self.cls_proj(patches.mean(dim=1, keepdim=True))
        return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], 1))


def make_predictor(num_frames=3):
    predictor = PatchPredictor(
        num_frames=num_frames,
        num_patches=4,
        input_dim=8,
        hidden_dim=12,
        output_dim=8,
        depth=2,
        heads=2,
        dim_head=4,
        mlp_dim=16,
    )
    # AdaLN gates initialize to zero. Activate them to exercise attention.
    for block in predictor.transformer.layers:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    return predictor


def make_model():
    return LeWMPatch(
        encoder=PatchEncoder(),
        predictor=make_predictor(),
        action_encoder=nn.Linear(2, 8),
        projector=MLP(8, 16, 8, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(8, 16, 8, norm_fn=nn.BatchNorm1d),
    )


def test_encode_preserves_cls_and_projects_each_patch():
    model = make_model().eval()
    pixels = torch.randn(2, 3, 3, 8, 8)
    action = torch.randn(2, 3, 2)
    out = model.encode({'pixels': pixels, 'action': action})
    patches = model.encoder(
        pixels.flatten(0, 1), interpolate_pos_encoding=True
    )
    expected = model.projector(
        patches.last_hidden_state.reshape(-1, 8)
    ).reshape(2, 3, 5, 8)
    torch.testing.assert_close(out['emb'], expected[:, :, 1:])
    torch.testing.assert_close(out['cls_emb'], expected[:, :, 0])
    torch.testing.assert_close(out['act_emb'], model.action_encoder(action))
    assert out['emb'].requires_grad
    assert 'act_emb' not in model.encode({'pixels': pixels})


@pytest.mark.parametrize('context', [1, 2, 3])
def test_predict_supports_short_windows_and_batchnorm(context):
    model = make_model()
    out = model.predict(
        torch.randn(2, context, 4, 8), torch.randn(2, context, 8)
    )
    assert out.shape == (2, context, 4, 8)


def test_attention_is_causal_in_time_but_full_within_frames():
    torch.manual_seed(2)
    predictor = make_predictor().eval()
    emb = torch.randn(2, 3, 4, 8)
    actions = torch.randn(2, 3, 8)
    expected = predictor(emb, actions)
    future = emb.clone()
    future[:, 2] = torch.randn_like(future[:, 2]) * 10
    future_actions = actions.clone()
    future_actions[:, 2] += 10
    actual = predictor(future, future_actions)
    torch.testing.assert_close(actual[:, :2], expected[:, :2])
    assert not torch.allclose(actual[:, 2], expected[:, 2])

    # A later patch in the SAME frame must affect its first patch.
    same_frame = emb.clone()
    same_frame[:, 0, -1] += torch.arange(8) * 5
    actual = predictor(same_frame, actions)
    assert not torch.allclose(actual[:, 0, 0], expected[:, 0, 0])
    torch.testing.assert_close(
        predictor(emb[:, :2], actions[:, :2]), expected[:, :2]
    )


@pytest.mark.parametrize(
    'emb_shape,action_shape,message',
    [
        ((2, 3, 8), (2, 3, 8), 'patch embeddings'),
        ((2, 4, 4, 8), (2, 4, 8), 'context frames'),
        ((2, 3, 5, 8), (2, 3, 8), 'patches'),
        ((2, 3, 4, 7), (2, 3, 8), 'dim'),
        ((2, 3, 4, 8), (2, 2, 8), 'action embeddings'),
    ],
)
def test_predictor_rejects_mismatched_inputs(emb_shape, action_shape, message):
    with pytest.raises(ValueError, match=message):
        make_predictor()(torch.randn(emb_shape), torch.randn(action_shape))


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('context', [1, 3])
def test_rollout_and_goal_cost_preserve_patches_and_action_gradients(
    cached, context
):
    torch.manual_seed(4)
    model = make_model().eval()
    pixels = torch.randn(2, context, 3, 8, 8)
    initial = model.encode({'pixels': pixels})['emb'].detach()
    candidates = torch.randn(2, 3, 4, 2, requires_grad=True)
    past = torch.randn(2, 3, context - 1, 2)
    info = {
        'pixels': pixels[:, None].expand(-1, 3, -1, -1, -1, -1),
        'action_history': past,
        'goal': torch.randn(2, 3, 1, 3, 8, 8),
    }
    if cached:
        info['emb'] = initial[:, None].expand(-1, 3, -1, -1, -1)
    evaluator = ShootingCostEvaluator(model, GoalMSE(reduction='mean'))
    cost = evaluator.get_cost(info, candidates)
    assert cost.shape == (2, 3)
    assert info['predicted_emb'].shape == (2, 3, context + 4, 4, 8)
    torch.testing.assert_close(
        info['predicted_emb'][:, :, :context],
        initial[:, None].expand(-1, 3, -1, -1, -1),
    )
    assert info['goal_emb'].shape == (2, 1, 4, 8)
    # Independently verify past/candidate pairing through the sliding window.
    frames = list(initial[:, None].expand(-1, 3, -1, -1, -1).unbind(2))
    actions = torch.cat([past, candidates.detach()], dim=2).flatten(0, 1)
    for step in range(4):
        lo = max(0, context + step - 3)
        window = torch.stack(frames[lo:], dim=2).flatten(0, 1)
        predicted = model.predict(
            window, model.action_encoder(actions[:, lo : context + step])
        )[:, -1].reshape(2, 3, 4, 8)
        frames.append(predicted)
    torch.testing.assert_close(
        info['predicted_emb'], torch.stack(frames, dim=2)
    )
    cost.sum().backward()
    assert torch.isfinite(candidates.grad).all()
    assert candidates.grad.abs().sum() > 0
    assert model.encoder.conv.weight.grad is None


@pytest.mark.parametrize('patches', [False, True])
def test_training_regularizes_cls_across_batch(patches):
    pytest.importorskip('stable_pretraining')
    from scripts.train.lewm import lejepa_forward

    model = make_model()
    if not patches:
        # Exercise the unchanged CLS loss shape with a small toy model.
        model = SimpleNamespace(
            encode=lambda batch: {
                'emb': batch['emb'],
                'act_emb': batch['action'],
            },
            predict=lambda emb, action: emb + action,
        )
    recorded = []
    regularizer = SIGReg(knots=5, num_proj=8)

    def sigreg(emb):
        recorded.append(emb)
        return regularizer(emb)

    module = SimpleNamespace(
        model=model, sigreg=sigreg, log_dict=lambda *args, **kwargs: None
    )
    cfg = SimpleNamespace(
        wm=SimpleNamespace(history_size=3, num_preds=1),
        loss=SimpleNamespace(sigreg=SimpleNamespace(weight=0.09)),
    )
    batch = {
        'pixels': torch.randn(2, 4, 3, 8, 8),
        'action': torch.randn(2, 4, 2 if patches else 8),
        'emb': torch.randn(2, 4, 8, requires_grad=True),
    }
    output = lejepa_forward(module, batch, 'train', cfg)
    expected = output['cls_emb' if patches else 'emb'].transpose(0, 1)
    assert recorded[0].shape == (4, 2, 8)
    torch.testing.assert_close(recorded[0], expected)
    prediction = model.predict(output['emb'][:, :3], output['act_emb'][:, :3])
    torch.testing.assert_close(
        output['pred_loss'],
        (prediction - output['emb'][:, 1:]).square().mean(),
    )
    torch.testing.assert_close(
        output['loss'], output['pred_loss'] + 0.09 * output['sigreg_loss']
    )
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    if patches:
        for component in (
            model.encoder,
            model.predictor,
            model.action_encoder,
        ):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in component.parameters()
            )


def test_cls_regularizer_backpropagates_to_encoder():
    model = make_model()
    output = model.encode({'pixels': torch.randn(2, 4, 3, 8, 8)})
    output['cls_emb'].retain_grad()
    output['emb'].retain_grad()
    loss = SIGReg(knots=5, num_proj=8)(output['cls_emb'].transpose(0, 1))
    loss.backward()
    assert output['cls_emb'].grad.abs().sum() > 0
    assert output['emb'].grad is None
    assert model.encoder.conv.weight.grad.abs().sum() > 0
    assert model.encoder.cls_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('config_name', ['lewm_patch', 'vision_lewm_patch'])
def test_config_instantiation_and_checkpoint_roundtrip(
    tmp_path, monkeypatch, config_name
):
    pytest.importorskip('stable_pretraining')
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Register the ViT dimension resolver used by the training configs.
    import scripts.train.lewm  # noqa: F401
    from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    config_dir = Path(__file__).resolve().parents[2] / 'scripts/train/config'
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    assert (
        cfg.model.predictor.num_patches
        == (cfg.img_size // cfg.patch_size) ** 2
    )
    assert cfg.model.predictor.input_dim == cfg.embed_dim
    # Instantiate the actual configured ViT, with a tiny grid and predictor.
    cfg.img_size = 16
    cfg.patch_size = 8
    cfg.encoder_scale = 'tiny'
    cfg.model.predictor.depth = 1
    cfg.model.predictor.heads = 2
    cfg.model.predictor.dim_head = 4
    cfg.model.predictor.mlp_dim = 16
    cfg.model.projector.hidden_dim = 16
    cfg.model.pred_proj.hidden_dim = 16
    cfg.model.action_encoder.input_dim = 2
    model = instantiate(cfg.model).eval()
    info = {
        'pixels': torch.randn(2, 3, 3, 16, 16),
        'action': torch.randn(2, 3, 2),
    }
    encoded = model.encode(dict(info))
    expected = model.predict(encoded['emb'], encoded['act_emb'])
    assert expected.shape == (2, 3, 4, 192)
    save_pretrained(model, 'patch_test', cfg.model, cache_dir=str(tmp_path))
    restored = load_pretrained('patch_test', cache_dir=str(tmp_path)).eval()
    encoded = restored.encode(dict(info))
    actual = restored.predict(encoded['emb'], encoded['act_emb'])
    torch.testing.assert_close(actual, expected)
    assert OmegaConf.to_container(cfg.model, resolve=True)[
        '_target_'
    ].endswith('.LeWMPatch')
