import torch, torch.nn as nn
from torch      import Tensor

from utils      import nn_utils, dev_utils

class _SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor):
        sig = torch.sigmoid(x)
        nn_utils.save_for_backward(ctx, x, sig)
        return x*sig
        #    x
        # ---------
        # 1 + e^(-x)
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        x,sig = nn_utils.dequantize(ctx, grad_output.dtype)
        
        return grad_output * ((x+1) * sig - x * (sig**2))
    


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:Tensor)->Tensor:
        return _SiLUFunction.apply(x)