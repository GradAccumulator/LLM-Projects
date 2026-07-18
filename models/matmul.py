import torch, torch.nn as nn
from torch     import Tensor
from configs   import runtime as rt

from utils     import nn_utils

class _MatmulFunction(torch.autograd.Function):
    @staticmethod
    def _forward_debug(a, b):
        func_name = "_MatmulFunction._forward_debug()"
        if a.dtype != b.dtype:
            raise ValueError(
                f"<{func_name}> 입력의 a.dtype, b.dtype이 다릅니다. a: {a.dtype}, b: {b.dtype}"
            )

    @staticmethod
    def forward(ctx, a:Tensor, b:Tensor):
        #a.shape = (..., a, c), b.shape = (..., c, b), out.shape = (..., a,b)
        if rt.DEBUG_CHECKS:
            _MatmulFunction._forward_debug(a, b)
        nn_utils.save_for_backward(ctx, a, b)

        return a@b
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        a, b   = nn_utils.dequantize(ctx, grad_output.dtype)
        grad_a = grad_b = None

        if grad_output.dtype != b.dtype:
            breakpoint()
        if ctx.needs_input_grad[0]:
            grad_a = grad_output@b.transpose(-1, -2)
        
        if ctx.needs_input_grad[1]:
            grad_b = a.transpose(-1, -2)@grad_output

        return grad_a, grad_b

def matmul(a:Tensor, b:Tensor) -> Tensor:
    return _MatmulFunction.apply(a, b)