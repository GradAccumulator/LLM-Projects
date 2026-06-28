import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from utils import nn_utils, dev_utils
from configs import runtime

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int,*, init_cfg:DictConfig|dict=None, use_bias:bool=True):
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
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            ("use_bias"     , use_bias      , bool)
            ,func_name="Linear.__init__()"
        )
        if in_features <= 0:
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
            ("weight", "bias", "method"),
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
        if runtime.DEBUG_CHECKS:
            self.forward_debug(x)
        
        out = x@self.weight.T
        if self.use_bias:
            if not hasattr(self, "_bias"):
                raise ValueError(
                    "<Linear.forward()> use_bias=True로 설정되어있지만 bias 텐서가 존재하지 않습니다."
                )
            out = out + self.bias
        return out
    
    @property
    def in_features(self): return self.weight.size(1)
    @property
    def out_features(self): return self.weight.size(0)
    @property
    def use_bias(self): return self._use_bias
    @property
    def weight(self): return self._weight
    @property
    def bias(self): return self._bias if self.use_bias else None
    