import torch, torch.nn as nn
from torch     import Tensor
from utils     import nn_utils, dev_utils

class _DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, p:int|float)->Tensor:
        ctx.p = p
        if ctx.p == 0:
            return x
        mask = torch.rand_like(x, dtype=torch.bfloat16) > ctx.p
        mask /= (1-ctx.p)
        mask = mask.to(x.dtype)
        nn_utils.save_for_backward(ctx, mask)
        return x*mask
    
    @staticmethod
    def backward(ctx, grad_output:Tensor)->tuple[Tensor, None]:
        if ctx.p == 0:
            return grad_output, None
        mask, = nn_utils.dequantize(ctx, grad_output.dtype)
        return grad_output*mask, None

class Dropout(nn.Module):
    def __init__(self, p:int|float):
        super().__init__()
        dev_utils.type_check(
            ("p", p, int|float),
            func_name="Dropout.__init__()"
        )
        if not 0<=p<1:
            raise ValueError("<Dropout.__init__()> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다.")
        self._p = p
        
    def forward(self, x:Tensor)->Tensor:
        return _DropoutFunction.apply(x, self.p*self.training)
    
    @property
    def p(self): return self._p