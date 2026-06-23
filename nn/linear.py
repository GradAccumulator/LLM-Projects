import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import nn_utils, dev_utils

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int,*, init_cfg:DictConfig, use_bias:bool=True):
        super().__init__()
        dev_utils.type_check(
            ("in_features"  , in_features   , int),
            ("out_features" , out_features  , int),
            ("init_cfg"     , init_cfg      , DictConfig),
            ("use_bias"     , use_bias      , bool)
            ,func_name="Linear.__init__()"
        )
        
        self._use_bias:bool
        self.register_buffer('_use_bias', use_bias)
        
        self._weight = nn.Parameter(
            nn_utils.init_tensor(in_features, out_features, init_cfg.weight)
        )
        if self.use_bias:
            self._bias = nn.Parameter(
                nn_utils.init_tensor(out_features, init_cfg.bias)
            )
    
    def forward(self, x:Tensor) -> Tensor:
        out = x@self.weight.T
        if self.use_bias:
            out += self.bias
        return out
    
    @property
    def in_features(self): return self.weight.size(0)
    @property
    def out_features(self): return self.weight.size(1)
    @property
    def use_bias(self): return self._use_bias
    @property
    def weight(self): return self._weight
    @property
    def bias(self): return self._bias
    