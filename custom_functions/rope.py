import torch
from torch import Tensor

from utils import nn_utils


class RotateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad: bool, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        nn_utils.save_for_backward(needs_grad, ctx, sin, cos)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd

        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        sin, cos = nn_utils.saved_tensors(ctx, grad_output.dtype)
        grad_even, grad_odd = grad_output[..., 0::2], grad_output[..., 1::2]

        grad_x_even = grad_even * cos + grad_odd * sin
        grad_x_odd = grad_odd * cos - grad_even * sin

        grad_x = torch.empty_like(grad_output)
        grad_x[..., 0::2] = grad_x_even
        grad_x[..., 1::2] = grad_x_odd

        return None, grad_x, None, None
