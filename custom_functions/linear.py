import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from configs import runtime as rt
from utils import nn_utils, dev_utils


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad: bool, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        nn_utils.save_for_backward(needs_grad, ctx, x, weight)
        ctx.use_bias = bias is not None

        out = x @ weight.T
        if ctx.use_bias:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        x, weight = nn_utils.saved_tensors(ctx, grad_output.dtype)
        # x.shape = (B,T,in), grad_output.shape = (B,T,out)
        # weight.shape = (out,in)

        grad_x = grad_output @ weight

        grad_weight = grad_output.view(-1, weight.size(0)).T @ x.view(-1, weight.size(1))

        grad_bias = grad_output.view(-1, weight.size(0)).sum(dim=0) if ctx.use_bias else None

        return None, grad_x, grad_weight, grad_bias

