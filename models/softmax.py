import torch, torch.nn as nn
from torch      import Tensor
from torch.amp  import custom_fwd, custom_bwd

from utils      import dev_utils,nn_utils

class _SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(
        ctx, 
        x:Tensor, 
        temperature:float,
        dim:int
    ):
        if temperature != 1:
            x = x / temperature
        x = x - x.max(dim=dim, keepdim=True).values
        exp_x = x.exp()
        y = exp_x/exp_x.sum(dim=dim, keepdim=True)
        nn_utils.save_for_backward(ctx, y)
        ctx.temperature = temperature
        return y
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output:Tensor):
        y, = nn_utils.dequantize(ctx) 
        out = (y * (grad_output 
            - (grad_output.unsqueeze(-2)@y.unsqueeze(-1)).squeeze(-1)
            )
        )
        if ctx.temperature != 1:
            return out/ctx.temperature
        return out,None,None

class Softmax(nn.Module):
    def __init__(self, dim:int=-1, temperature:int|float=1.0):
        super().__init__()
        dev_utils.type_check(
            ("dim"          , dim           , int),
            ("temperature"  , temperature   , float|int),
            func_name="Softmax.__init__()"
        )
        if temperature<=0:
            raise ValueError("<Softmax.__init__()> Softmax의 temperature은 양수여야 합니다.")
        
        self._dim         = dim
        self._temperature = temperature
    
    def forward(self, x:Tensor) -> Tensor:
        return _SoftmaxFunction.apply(x, self.temperature, self.dim)
    
    @property
    def dim(self): return self._dim
    @property
    def temperature(self): return self._temperature