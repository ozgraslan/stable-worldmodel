from collections.abc import Iterable

import torch


def encoder_gradient_diagnostics(
    pred_loss: torch.Tensor,
    sigreg_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    sigreg_weight: float,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare unscaled loss gradients without changing parameter ``.grad``.

    Retain the forward graph for the usual backward pass. Unused parameters
    contribute zeros at their original positions in both gradient vectors.
    Reduce in FP32 outside autocast, including when training uses FP16/BF16.
    A zero norm gives a cosine similarity of zero.
    """
    parameters = tuple(p for p in parameters if p.requires_grad)

    def gradients(loss):
        if not parameters or not loss.requires_grad:
            return (None,) * len(parameters)
        return torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    # Use the original losses, before Lightning's backward loss scaling.
    pred_grads = gradients(pred_loss)
    sigreg_grads = gradients(sigreg_loss)
    with (
        torch.no_grad(),
        torch.autocast(device_type=pred_loss.device.type, enabled=False),
    ):
        pred_sq = pred_loss.new_zeros((), dtype=torch.float32)
        sigreg_sq = torch.zeros_like(pred_sq)
        dot = torch.zeros_like(pred_sq)
        for pred_grad, sigreg_grad in zip(pred_grads, sigreg_grads):
            if pred_grad is not None:
                pred_grad = pred_grad.float()
                pred_sq += pred_grad.square().sum()
            if sigreg_grad is not None:
                sigreg_grad = sigreg_grad.float()
                sigreg_sq += sigreg_grad.square().sum()
            if pred_grad is not None and sigreg_grad is not None:
                dot += (pred_grad * sigreg_grad).sum()

        pred_norm = pred_sq.sqrt()
        sigreg_norm = sigreg_sq.sqrt()
        norm_product = pred_norm * sigreg_norm
        denominator = torch.where(
            norm_product > 0, norm_product, torch.ones_like(norm_product)
        )
        cosine = (dot / denominator).clamp(-1, 1)
        return {
            'pred_grad_norm': pred_norm,
            'sigreg_grad_norm': sigreg_norm,
            'relative_grad_magnitude': (
                sigreg_weight * sigreg_norm / (pred_norm + eps)
            ),
            'grad_cosine_similarity': cosine,
        }
