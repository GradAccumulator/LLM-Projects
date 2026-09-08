import torch, torch.nn as nn
from torch import Tensor

from utils import nn_utils


class _MulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad:bool, a: Tensor, b: Tensor):
        nn_utils.save_for_backward(needs_grad, ctx, a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = nn_utils.saved_tensors(ctx, grad_output.dtype)
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * b
        if ctx.needs_input_grad[1]:
            grad_b = grad_output * a

        return None, grad_a, grad_b


def mul(a: Tensor, b: Tensor) -> Tensor:
    return _MulFunction.apply(torch.is_grad_enabled(), a, b)
