import torch, torch.nn as nn
from torch      import Tensor
from omegaconf  import DictConfig

from configs    import runtime as rt
from utils      import nn_utils, dev_utils

class _LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, weight:Tensor, bias:Tensor|None)->Tensor:
        nn_utils.save_for_backward(ctx, x, weight)
        ctx.use_bias = bias is not None

        out = x@weight.T
        if ctx.use_bias:
            out = out + bias
        return out
    
    @staticmethod
    def backward(ctx, grad_output:Tensor)->tuple[Tensor, Tensor, Tensor|None]:
        x, weight = nn_utils.dequantize(ctx)
        #x.shape = (B,T,in), grad_output.shape = (B,T,out)
        #weight.shape = (out,in)
        
        grad_x = grad_output @ weight

        grad_weight = grad_output.view(-1, weight.size(0)).T @ x.view(-1, weight.size(1))

        grad_bias = grad_output.view(-1, weight.size(0)).sum(dim=0) if ctx.use_bias else None

        return grad_x, grad_weight, grad_bias

class Linear(nn.Module):
    def __init__(
        self, 
        in_features     :int, 
        out_features    :int,
        *, 
        use_bias        :bool =True,
        init_cfg        :DictConfig|dict =None, 
    ):
        '''```
        init_cfg = {
            "weight": {
                "method":...
            },
            "bias": {
                "method":...
            }
        }
        ```'''
        
        super().__init__()
        dev_utils.type_check(
            ("in_features"  , in_features   , int),
            ("out_features" , out_features  , int),
            ("use_bias"     , use_bias      , bool),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            func_name="Linear.__init__()"
        )
        if in_features  <= 0:
            raise ValueError("<Linear.__init__()> in_features는 양의 정수여야 합니다.")
        if out_features <= 0:
            raise ValueError("<Linear.__init__()> out_features는 양의 정수여야 합니다.")
        
        self._use_bias = use_bias
        
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "weight": {
                "method":"normal",
                "std":0.02
            },
            "bias": {
                "method":"zeros"
            }
        })
        dev_utils.check_dictconfig(
            init_cfg,
            ("weight", "bias", "weight.method", "bias.method"),
            "Linear.__init__()"
        )
        
        self._weight = nn.Parameter(
            nn_utils.init_tensor(out_features, in_features, init_cfg=init_cfg.weight)
        )
        if self.use_bias:
            self._bias = nn.Parameter(
                nn_utils.init_tensor(out_features, init_cfg=init_cfg.bias)
            )
    
    def forward_debug(self, x:Tensor):
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"<Linear.forward()> 입력 텐서의 마지막 차원의 크기가 부적절합니다."
                f"예상한 크기: {self.in_features}, 실제 크기:{x.size(-1)}"
            )
    
    def forward(self, x:Tensor) -> Tensor:
        if rt.DEBUG_CHECKS:
            self.forward_debug(x)
        
        return _LinearFunction.apply(x, self.weight, self.bias)
    
    @property
    def bias(self): return self._bias if self.use_bias else None
    @property
    def weight(self): return self._weight
    @property
    def use_bias(self): return self._use_bias
    @property
    def in_features(self): return self.weight.size(1)
    @property
    def out_features(self): return self.weight.size(0)
    