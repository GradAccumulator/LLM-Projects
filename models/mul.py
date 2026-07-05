import torch, torch.nn as nn
from torch      import Tensor
from torch.amp  import custom_fwd, custom_bwd

from utils      import nn_utils

class _MulFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, a:Tensor, b:Tensor):
        nn_utils.save_for_backward(ctx, a, b)
        return a*b
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output:Tensor):
        a,b    = nn_utils.dequantize(ctx)
        grad_a = grad_b = None
        
        if ctx.needs_input_grad[0]:
            grad_a = grad_output*b
        if ctx.needs_input_grad[1]:
            grad_b = grad_output*a
        
        return grad_a,grad_b

def mul(a:Tensor, b:Tensor) -> Tensor:
    return _MulFunction.apply(a, b)