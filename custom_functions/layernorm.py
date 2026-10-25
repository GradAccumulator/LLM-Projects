import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from configs import runtime as rt
from utils import nn_utils, dev_utils


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        needs_grad: bool,
        x: Tensor,
        gamma: Tensor,
        beta: Tensor,
        normalized_dims: tuple[int, ...],
        eps: float,
    ):
        mean: Tensor = x.mean(dim=normalized_dims, keepdim=True)
        v = x - mean
        var: Tensor = (v**2).to(x.dtype).mean(dim=normalized_dims, keepdim=True)
        inv_std = (var + eps).rsqrt().to(x.dtype)

        x_hat = (x - mean) * inv_std

        out = x_hat * gamma
        if beta is not None:
            out = out + beta

        nn_utils.save_for_backward(needs_grad, ctx, v, inv_std, gamma, beta, x_hat)
        ctx.normalized_dims = normalized_dims
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        v, inv_std, gamma, beta, x_hat = nn_utils.saved_tensors(ctx, grad_output.dtype)
        sum_dims = tuple(range(-len(v.shape), -len(ctx.normalized_dims)))

        grad_x_hat = grad_output * gamma
        grad_gamma = (grad_output * x_hat).sum(sum_dims)
        grad_beta = None
        if beta is not None:
            grad_beta = grad_output.sum(sum_dims)

        grad_x = inv_std * (
            grad_x_hat
            - grad_x_hat.mean(dim=ctx.normalized_dims, keepdim=True)
            - (inv_std.square() * v) * (v * grad_x_hat).mean(dim=ctx.normalized_dims, keepdim=True)
        )
        return None, grad_x, grad_gamma, grad_beta, None, None
