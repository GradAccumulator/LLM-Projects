import torch, torch.nn as nn
from torch      import Tensor
from omegaconf  import DictConfig

from configs    import runtime as rt
from utils      import nn_utils, dev_utils

class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        x               :Tensor,
        gamma           :Tensor,
        beta            :Tensor,
        normalized_dims :tuple[int,...], 
        eps             :float
    ):
        mean:Tensor = x.mean(dim=normalized_dims, keepdim=True)
        v           = x - mean
        var :Tensor = (v**2).to(x.dtype).mean(dim=normalized_dims, keepdim=True)
        inv_std     = (var+eps).rsqrt().to(x.dtype)
        
        x_hat = (x-mean)*inv_std
        
        out = x_hat*gamma
        if beta is not None:
            out = out + beta
        
        nn_utils.save_for_backward(ctx, v, inv_std, gamma, beta, x_hat)
        ctx.normalized_dims = normalized_dims
        return out
    
    @staticmethod
    def backward(ctx, grad_output:Tensor):
        v,inv_std,gamma,beta,x_hat = nn_utils.dequantize(ctx, grad_output.dtype)
        sum_dims = tuple(range(-len(v.shape), -len(ctx.normalized_dims)))

        grad_x_hat = grad_output * gamma
        grad_gamma = (grad_output * x_hat).sum(sum_dims)
        grad_beta  = None
        if beta is not None:
            grad_beta = grad_output.sum(sum_dims)
        
        grad_x = inv_std * (
            grad_x_hat
            - grad_x_hat.mean(dim=ctx.normalized_dims, keepdim=True)
            - (inv_std.square() * v)
            * (v * grad_x_hat).mean(dim=ctx.normalized_dims, keepdim=True)
        )
        return grad_x,grad_gamma,grad_beta,None,None

class LayerNorm(nn.Module):
    def __init__(
        self, 
        *normalized_shape:int, 
        eps:float=1e-5,
        bias:bool=True,
        init_cfg:DictConfig|dict=None, 
    ):
        '''```
        init_cfg = { 
            "gamma": { 
                "method":... 
            },
            "beta": {
                "method":...
            }
        }
        ```'''
        super().__init__()
        dev_utils.type_check(
            ("bias"         , bias          , bool),
            ("eps"          , eps           , float),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            func_name="LayerNorm.__init__()"
        )
        for i,value in enumerate(normalized_shape):
            if not isinstance(value, int):
                raise TypeError(
                    "<LayerNorm.__init__()> normalized_shape의 타입이 부적절합니다. normalized_shape의 모든 값은 int여야 합니다. "
                    f"{i}번째 값의 타입:{type(value)}"
                )
            if value<=0:
                raise ValueError(
                    "<LayerNorm.__init__()> normalized_shape의 모든 값은 양의 정수여야 합니다. "
                    f"{i}번째 값: {value}"
                )
        if len(normalized_shape) == 0:
            raise ValueError("<LayerNorm.__init__()> normalized_shape는 최소 1개 이상이어야 합니다.")
        if eps<=0:
            raise ValueError("<LayerNorm.__init__()> eps는 반드시 양수여야 합니다.")
        
        self._eps = eps
        self._use_bias = bias
        self._normalized_dims = tuple(range(-len(normalized_shape), 0))
        
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "gamma": {
                "method":"ones"
            },
            "beta": {
                "method":"zeros"
            }
        })
        dev_utils.check_dictconfig(
            init_cfg,
            ("gamma", "beta", "gamma.method", "beta.method"),
            "LayerNorm.__init__()"
        )
        
        self._gamma = nn.Parameter(
            nn_utils.init_tensor(*normalized_shape, init_cfg=init_cfg.gamma)
        )
        if self.use_bias:
            self._beta = nn.Parameter(
                nn_utils.init_tensor(*normalized_shape, init_cfg=init_cfg.beta)
            )
    
    def forward_debug(self, x:Tensor):
        #x.shape == (B, T, D), self.normalized_shape == (D,)
        if self.normalized_shape != x.shape[-len(self.normalized_shape):]:
            raise ValueError(
                f"<LayerNorm.forward()> 입력 텐서의 shape는 {tuple(self.normalized_shape)}로 끝나야 합니다. "
                f"(현재: {tuple(x.shape)})"
            )
    
    def forward(self, x:Tensor)->Tensor:
        #x.shape == (B, T, D)
        if rt.DEBUG_CHECKS:
            self.forward_debug(x)
        
        return _LayerNormFunction.apply(
            x,
            self.gamma,
            self.beta,
            self.normalized_dims,
            self.eps
        )
    
    @property
    def eps(self): return self._eps
    @property
    def beta(self): return self._beta if self.use_bias else None
    @property
    def gamma(self): return self._gamma
    @property
    def use_bias(self): return self._use_bias
    @property
    def normalized_dims(self): return self._normalized_dims
    @property
    def normalized_shape(self): return self.gamma.shape