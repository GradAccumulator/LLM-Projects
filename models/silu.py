import torch, torch.nn as nn
from torch import Tensor

from utils import nn_utils, dev_utils


class _SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, needs_grad:bool, x: Tensor):
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


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return _SiLUFunction.apply(torch.is_grad_enabled(), x)
