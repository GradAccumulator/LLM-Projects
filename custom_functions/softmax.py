import torch
from torch import Tensor

from utils import nn_utils


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad: bool, x: Tensor, temperature: float, dim: int):
        if temperature != 1:
            x = x / temperature

        x -= x.max(dim=dim, keepdim=True).values
        exp_x = x.exp()
        y = exp_x / exp_x.sum(dim=dim, keepdim=True)
        y = y.to(x.dtype)

        nn_utils.save_for_backward(needs_grad, ctx, y)
        ctx.temperature = temperature
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (y,) = nn_utils.saved_tensors(ctx, grad_output.dtype)
        out = y * (grad_output - (grad_output * y).sum(dim=ctx.dim, keepdim=True))
        if ctx.temperature != 1:
            return None, out / ctx.temperature, None, None
        return None, out, None, None
