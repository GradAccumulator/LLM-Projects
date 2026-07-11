import torch, torch.nn as nn
from torch     import Tensor

from utils     import nn_utils

class _MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a:Tensor, b:Tensor):
        #a.shape = (..., a, c), b.shape = (..., c, b), out.shape = (..., a,b)
        nn_utils.save_for_backward(ctx, a, b)

        return a@b
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        a, b   = nn_utils.dequantize(ctx)
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output@b.transpose(-1, -2)
        
        if ctx.needs_input_grad[1]:
            grad_b = a.transpose(-1, -2)@grad_output

        return grad_a, grad_b

def matmul(a:Tensor, b:Tensor) -> Tensor:
    return _MatmulFunction.apply(a, b)