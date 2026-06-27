import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import nn_utils, dev_utils

class LayerNorm(nn.Module):
    def __init__(
        self, 
        *normalized_shape:int, 
        init_cfg:DictConfig|dict=None, 
        eps:float=1e-5
    ):
        '''```
        init_cfg = { 
            "alpha": { 
                "method":... 
            },
            "beta": {
                "method":...
            }
        }
        ```'''
        
        super().__init__()
        dev_utils.type_check(
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            ("eps"          , eps           , float)
            ,func_name="LayerNorm.__init__()"
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
        self._normalized_dims = tuple(range(-len(normalized_shape), 0))
        
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "alpha": {
                "method":"ones"
            },
            "beta": {
                "method":"zeros"
            }
        })
        
        self._alpha = nn.Parameter(
            nn_utils.init_tensor(*normalized_shape, init_cfg=init_cfg.alpha)
        )
        self._beta = nn.Parameter(
            nn_utils.init_tensor(*normalized_shape, init_cfg=init_cfg.beta)
        )
    
    def forward(self, x:Tensor)->Tensor:
        if self.normalized_shape != x.shape[-len(self.normalized_shape):]:
            raise ValueError(
                f"<LayerNorm.forward()> 입력 텐서의 shape는 {tuple(self.normalized_shape)}로 끝나야 합니다. "
                f"(현재: {tuple(x.shape)})"
            )
        
        mean = x.mean(dim=self.normalized_dims, keepdim=True)
        var = x.var(dim=self.normalized_dims, keepdim=True, unbiased=False)
        
        x_hat = (x-mean)/(var+self.eps).sqrt()
        
        return self.alpha * x_hat + self.beta
    
    @property
    def alpha(self):return self._alpha
    @property
    def beta(self):return self._beta
    @property
    def eps(self):return self._eps
    @property
    def normalized_shape(self):return self.alpha.shape
    @property
    def normalized_dims(self):return self._normalized_dims