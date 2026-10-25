import torch, torch.nn as nn
from torch import Tensor
from utils import nn_utils, dev_utils


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad: bool, x: Tensor, p: int | float) -> Tensor:
        ctx.p = p
        if ctx.p == 0:
            return x
        mask = torch.rand_like(x, dtype=torch.bfloat16) > ctx.p
        mask /= 1 - ctx.p
        mask = mask.to(x.dtype)
        nn_utils.save_for_backward(needs_grad, ctx, mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        if ctx.p == 0:
            return None, grad_output, None
        (mask,) = nn_utils.saved_tensors(ctx, grad_output.dtype)
        return None, grad_output * mask, None
