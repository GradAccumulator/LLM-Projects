import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import nn_utils, dev_utils

class RoPE(nn.Module):
    def __init__(self, RoPE_cfg:DictConfig|dict):
        super().__init__()
        dev_utils.type_check(
            ("RoPE_cfg", RoPE_cfg, DictConfig|dict)
            ,func_name="RoPE.__init__()"
        )
        
        self.cfg = RoPE_cfg
        self.cached_sin:Tensor
        self.cached_cos:Tensor
    
    def compute_sin_cos(self, T:int, D:int, device:torch.device, dtype:torch.dtype) -> tuple[Tensor, Tensor]:
        need_new_cache = (
            not hasattr(self, "cached_sin")
            or tuple(self.cached_sin.shape) != (T, D // 2)
            or self.cached_sin.device != device
            or self.cached_sin.dtype != dtype
        )
        if need_new_cache:
            angles = torch.arange(T, device=device,dtype=torch.float32)[:, None] \
                * self.base**(
                    -2 * torch.arange(D//2, device=device,dtype=torch.float32) / D
                )[None, :]
            
            sin = angles.sin().to(dtype)
            cos = angles.cos().to(dtype)
            
            self.register_buffer("cached_sin", sin, persistent=False)
            self.register_buffer("cached_cos", cos, persistent=False)
        else:
            sin,cos = self.cached_sin, self.cached_cos
        
        return sin,cos
    
    def _rotate(self, x:Tensor, sin:Tensor, cos:Tensor) -> Tensor:
        x_odd, x_even = x[...,1::2], x[...,0::2]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        
        out = torch.empty_like(x)
        out[...,1::2] = x_rot_odd
        out[...,0::2] = x_rot_even
        
        return out
    
    def forward(self, Q:Tensor, K:Tensor)->Tensor:
        #Q, K.shape == (B, H, T, D)
        T = Q.size(2)
        D = Q.size(3)
        
        if D%2!=0:
            raise ValueError("RoPE의 입력으로 주어지는 Q 행렬의 마지막 차원의 크기는 짝수여야 합니다")
        
        sin,cos = self.compute_sin_cos(T, D, Q.device, Q.dtype)
        
        Q_rot = self._rotate(Q, sin, cos)
        K_rot = self._rotate(K, sin, cos)
        
        return Q_rot, K_rot
    
    
    @property
    def base(self)->int|float: return self.cfg['base']
    @base.setter
    def base(self, x:int|float):
        dev_utils.type_check(
            ("x", x, int|float)
            ,func_name="RoPE.base.setter()"
        )
        self.cfg['base'] = x
        
        if hasattr(self, "cached_sin"):
            self.__delattr__("cached_sin")
            self.__delattr__("cached_cos")