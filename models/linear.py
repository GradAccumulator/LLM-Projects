import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import nn_utils, dev_utils

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int,*, cfg:DictConfig, use_bias:bool=True):
        super().__init__()
        dev_utils.type_check(
            ("in_features"  , in_features   , int),
            ("out_features" , out_features  , int),
            ("cfg"          , cfg           , DictConfig),
            ("use_bias"     , use_bias      , bool),
        )
        self._in_features   = in_features
        self._out_features  = out_features
        self._use_bias      = use_bias
        self._cfg = cfg
        
        self._weight = nn.Parameter(
            nn_utils.init_tensor(self.in_features, self.out_features, self.cfg.init.linear.weight)
        )
        if self.use_bias:
            self._bias = nn.Parameter(
                nn_utils.init_tensor(self.out_features, self.cfg.init.linear.bias)
            )
    
    def forward(self, x:Tensor) -> Tensor:
        out = x@self.weight.T
        if self.use_bias:
            out += self.bias
        return out
    
    @property
    def in_features(self): return self._in_features
    @property
    def out_features(self): return self._out_features
    @property
    def use_bias(self): return self._use_bias
    @property
    def cfg(self): return self._cfg
    @property
    def weight(self): return self._weight
    @property
    def bias(self): return self._bias
    