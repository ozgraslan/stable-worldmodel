import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_worldmodel.wm.diagnostics import encoder_gradient_diagnostics
from stable_worldmodel.wm.lewm import LeWMState
from stable_worldmodel.wm.loss import SIGReg


@pytest.mark.parametrize(
    ('sigreg_vector', 'expected_norm', 'expected_cosine'),
    [
        ([6.0, 8.0], 10.0, 1.0),
        ([-3.0, -4.0], 5.0, -1.0),
        ([-4.0, 3.0], 5.0, 0.0),
        ([0.0, 5.0], 5.0, 0.8),
    ],
)
def test_known_gradient_norms_and_cosine(
    sigreg_vector, expected_norm, expected_cosine
):
    parameter = nn.Parameter(torch.ones(2))
    pred_loss = (parameter * torch.tensor([3.0, 4.0])).sum()
    sigreg_loss = (parameter * torch.tensor(sigreg_vector)).sum()
    metrics = encoder_gradient_diagnostics(
        pred_loss, sigreg_loss, [parameter], sigreg_weight=0.2
    )

    assert metrics['pred_grad_norm'].item() == pytest.approx(5.0)
    assert metrics['sigreg_grad_norm'].item() == pytest.approx(expected_norm)
    assert metrics['relative_grad_magnitude'].item() == pytest.approx(
        0.2 * expected_norm / (5.0 + 1e-8)
    )
    assert metrics['grad_cosine_similarity'].item() == pytest.approx(
        expected_cosine
    )
    assert all(not value.requires_grad for value in metrics.values())


def test_unused_parameters_preserve_alignment_and_backward():
    shared = nn.Parameter(torch.ones(2))
    pred_only = nn.Parameter(torch.ones(1))
    sigreg_only = nn.Parameter(torch.ones(1))
    unused = nn.Parameter(torch.ones(1))
    frozen = nn.Parameter(torch.ones(1), requires_grad=False)
    parameters = [shared, pred_only, sigreg_only, unused, frozen]
    pred_loss = (
        shared * torch.tensor([3.0, 4.0])
    ).sum() + 12 * pred_only.sum()
    sigreg_loss = 5 * shared[1] + 8 * sigreg_only.sum()
    shared.grad = torch.tensor([7.0, 9.0])

    metrics = encoder_gradient_diagnostics(
        pred_loss, sigreg_loss, parameters, sigreg_weight=0.2
    )
    pred_vector = torch.tensor([3.0, 4.0, 12.0, 0.0, 0.0])
    sigreg_vector = torch.tensor([0.0, 5.0, 0.0, 8.0, 0.0])
    torch.testing.assert_close(metrics['pred_grad_norm'], pred_vector.norm())
    torch.testing.assert_close(
        metrics['sigreg_grad_norm'], sigreg_vector.norm()
    )
    torch.testing.assert_close(
        metrics['grad_cosine_similarity'],
        torch.nn.functional.cosine_similarity(
            pred_vector, sigreg_vector, dim=0
        ),
    )
    torch.testing.assert_close(shared.grad, torch.tensor([7.0, 9.0]))
    assert all(p.grad is None for p in parameters[1:])

    (pred_loss + 0.2 * sigreg_loss).backward()
    torch.testing.assert_close(shared.grad, torch.tensor([10.0, 14.0]))
    torch.testing.assert_close(pred_only.grad, torch.tensor([12.0]))
    torch.testing.assert_close(sigreg_only.grad, torch.tensor([1.6]))
    assert unused.grad is None and frozen.grad is None


@pytest.mark.parametrize('constant_loss', [False, True])
@pytest.mark.parametrize('zero_loss', ['prediction', 'sigreg', 'both'])
def test_zero_norms_are_safe(constant_loss, zero_loss):
    parameter = nn.Parameter(torch.ones(2))
    zero = torch.tensor(0.0) if constant_loss else parameter.sum() * 0
    pred_loss = zero if zero_loss != 'sigreg' else parameter.sum()
    sigreg_loss = zero if zero_loss != 'prediction' else parameter.sum()
    metrics = encoder_gradient_diagnostics(
        pred_loss, sigreg_loss, [parameter], sigreg_weight=0.2
    )

    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics['grad_cosine_similarity'].item() == 0
    if zero_loss != 'sigreg':
        assert metrics['pred_grad_norm'].item() == 0
    if zero_loss != 'prediction':
        assert metrics['sigreg_grad_norm'].item() == 0
        assert metrics['relative_grad_magnitude'].item() == 0


def test_no_trainable_encoder_parameters():
    parameter = nn.Parameter(torch.ones(2), requires_grad=False)
    loss = torch.ones((), requires_grad=True)
    metrics = encoder_gradient_diagnostics(loss, loss, [parameter], 0.2)
    assert all(value.item() == 0 for value in metrics.values())


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_gradients_are_reduced_in_fp32(dtype):
    parameter = nn.Parameter(torch.ones(2, dtype=dtype))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = (parameter * parameter.new_tensor([300.0, 400.0])).sum()
        metrics = encoder_gradient_diagnostics(loss, loss, [parameter], 0.2)

    assert metrics['pred_grad_norm'].dtype == torch.float32
    assert metrics['pred_grad_norm'].item() == pytest.approx(500.0)
    assert metrics['sigreg_grad_norm'].item() == pytest.approx(500.0)
    assert metrics['grad_cosine_similarity'].item() == pytest.approx(1.0)
    assert parameter.grad is None
    loss.backward()
    torch.testing.assert_close(
        parameter.grad, parameter.new_tensor([300.0, 400.0])
    )


@pytest.fixture
def training_forward():
    pytest.importorskip('stable_pretraining')
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_training_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.lejepa_forward


class _Predictor(nn.Module):
    def forward(self, emb, act_emb):
        return emb + act_emb


@pytest.mark.parametrize('mixed_precision', [False, True])
def test_training_logging_schedule_and_optimizer_parity(
    training_forward, mixed_precision
):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            'wm': {'history_size': 2, 'num_preds': 1},
            'loss': {'sigreg': {'weight': 0.2}},
        }
    )
    batch = {'state': torch.randn(4, 3, 2), 'action': torch.randn(4, 3, 1)}
    results = []
    for interval in (0, 100):
        torch.manual_seed(0)
        model = LeWMState(
            encoder=nn.Linear(2, 2),
            predictor=_Predictor(),
            action_encoder=nn.Linear(1, 2),
        )
        logs = []
        module = SimpleNamespace(
            model=model,
            sigreg=SIGReg(knots=3, num_proj=4),
            global_step=0,
            log_dict=lambda values, logs=logs, **kwargs: logs.append(values),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg.gradient_diagnostics_interval = interval
        # Repeated step 0 checks that accumulation doesn't duplicate logging.
        for step in (0, 0, 99, 100):
            module.global_step = step
            optimizer.zero_grad()
            with torch.autocast(
                'cpu', dtype=torch.bfloat16, enabled=mixed_precision
            ):
                output = training_forward(module, dict(batch), 'fit', cfg)
            output['loss'].backward()
            optimizer.step()
        with torch.no_grad():
            training_forward(module, dict(batch), 'validate', cfg)
        diagnostic_logs = [
            values for values in logs if 'fit/pred_grad_norm' in values
        ]
        assert len(diagnostic_logs) == (2 if interval else 0)
        if interval:
            assert set(diagnostic_logs[0]) == {
                'fit/pred_grad_norm',
                'fit/sigreg_grad_norm',
                'fit/relative_grad_magnitude',
                'fit/grad_cosine_similarity',
            }
        results.append([p.detach().clone() for p in model.parameters()])

    for disabled, enabled in zip(*results):
        torch.testing.assert_close(disabled, enabled, rtol=0, atol=0)
