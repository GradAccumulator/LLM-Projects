import torch
from torch import Tensor

from utils import nn_utils


class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad: bool, x: Tensor):
        sig = torch.sigmoid(x)
        nn_utils.save_for_backward(needs_grad, ctx, x, sig)
        return x * sig
        #    x
        # ---------
        # 1 + e^(-x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, sig = nn_utils.saved_tensors(ctx, grad_output.dtype)

        return None, grad_output * ((x + 1) * sig - x * (sig**2))
