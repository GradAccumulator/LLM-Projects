import torch, torch.nn as nn
from torch      import Tensor
from torch.amp  import custom_fwd, custom_bwd

from configs    import runtime as rt
from utils      import dev_utils, nn_utils

class _RotateFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x:Tensor, sin:Tensor, cos:Tensor) -> Tensor:
        nn_utils.save_for_backward(ctx, sin, cos)
        x_even, x_odd = x[...,0::2], x[...,1::2]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        
        out = torch.empty_like(x)
        out[...,0::2] = x_rot_even
        out[...,1::2] = x_rot_odd
        
        return out
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output:Tensor):
        sin,cos = nn_utils.dequantize(ctx)
        grad_even, grad_odd = grad_output[...,0::2], grad_output[...,1::2]

        grad_x_even = grad_even*cos + grad_odd*sin
        grad_x_odd  = grad_odd*cos - grad_even*sin

        grad_x = torch.empty_like(grad_output)
        grad_x[...,0::2] = grad_x_even
        grad_x[...,1::2] = grad_x_odd
        
        return grad_x,None,None

class RoPE(nn.Module):
    def __init__(self, base:float|int=None):
        super().__init__()
        if base is None:
            base = 10000
        dev_utils.type_check(
            ("base", base, float|int),
            func_name="RoPE.__init__()"
        )
        if base <= 0:
            raise ValueError("<RoPE.__init__()> RoPE의 base는 양수여야 합니다.")
        
        self._base = base
        
        self.cached_sin:Tensor
        self.cached_cos:Tensor

    @staticmethod
    def compute_sin_cos(T:int, D:int, device:torch.device, dtype:torch.dtype, base:int|float) -> tuple[Tensor, Tensor]:
        angles = torch.arange(T, device=device,dtype=torch.float32)[:, None] \
            * base**(
                -2 * torch.arange(D//2, device=device,dtype=torch.float32) / D
            )[None, :]
        
        sin = angles.sin().to(dtype)
        cos = angles.cos().to(dtype)
        return sin, cos
    
    def _compute_sin_cos_safe(
        self, 
        T:int, 
        D:int, 
        device:torch.device, 
        dtype:torch.dtype, 
        cached_sin:Tensor=None, 
        cached_cos:Tensor=None
    ) -> tuple[Tensor, Tensor]:
        cached_sin_cos_is_given = cached_sin is not None and cached_cos is not None
        if cached_sin is None and hasattr(self, "cached_sin"):
            cached_sin = self.cached_sin
        if cached_cos is None and hasattr(self, "cached_cos"):
            cached_cos = self.cached_cos
        
        need_new_cache = (
            cached_sin.shape != (T, D // 2)
            or cached_sin.device != device
            or cached_sin.dtype != dtype
        )
        if need_new_cache:
            if cached_sin_cos_is_given:
                raise ValueError(
                    "RoPE에 주어진 cached_sin, cached_cos의 shape, device, dtype이 입력 텐서와 맞지 않습니다."
                    f"\n예상 shape=(..., {cached_sin.size(0)}, {cached_sin.size(1)}), device={cached_sin.device}, dtype={cached_sin.dtype}"
                    f"\n현재 shape=(..., {T}, {D//2}), device={device}, dtype={dtype}"
                )

            sin, cos = self.compute_sin_cos(T, D, device, dtype, self.base)
            self.register_buffer("cached_sin", sin, persistent=False)
            self.register_buffer("cached_cos", cos, persistent=False)
        else:
            sin,cos = cached_sin, cached_cos
        
        return sin,cos
    
    def _compute_sin_cos_fast(
        self, 
        T:int, 
        D:int, 
        device:torch.device, 
        dtype:torch.dtype, 
        cached_sin:Tensor=None, 
        cached_cos:Tensor=None
    ) -> tuple[Tensor, Tensor]:
        if cached_sin is None:
            if hasattr(self, "cached_sin"):
                cached_sin = self.cached_sin
                cached_cos = self.cached_cos
            else:
                cached_sin,cached_cos = self.compute_sin_cos(T, D, device, dtype, self.base)
                self.register_buffer("cached_sin", cached_sin, persistent=False)
                self.register_buffer("cached_cos", cached_cos, persistent=False)
        return cached_sin, cached_cos
        
    
    def _rotate(self, x:Tensor, sin:Tensor, cos:Tensor) -> Tensor:
        return _RotateFunction.apply(x, sin, cos)
    
    def forward(self, Q:Tensor, K:Tensor, cached_sin:Tensor=None, cached_cos:Tensor=None)->tuple[Tensor, Tensor]:
        #Q, K.shape == (B, H, T, D)
        T = Q.size(2)
        D = Q.size(3)
        
        if D%2!=0:
            raise ValueError(f"RoPE의 입력으로 주어지는 Q 행렬의 마지막 차원의 크기는 짝수여야 합니다. 현재: {D}")
        
        if rt.DEBUG_CHECKS:
            sin,cos = self._compute_sin_cos_safe(
                T, D, 
                Q.device, 
                Q.dtype,
                cached_sin,
                cached_cos
            )
        else:
            sin,cos = self._compute_sin_cos_fast(
                T, D, 
                Q.device, 
                Q.dtype,
                cached_sin,
                cached_cos
            )
        
        Q_rot = self._rotate(Q, sin, cos)
        K_rot = self._rotate(K, sin, cos)
        
        return Q_rot, K_rot
    
    
    @property
    def base(self)->int|float: return self._base
    @base.setter
    def base(self, x:int|float):
        dev_utils.type_check(
            ("x", x, int|float)
            ,func_name="RoPE.base.setter()"
        )
        self._base = x
        
        if hasattr(self, "cached_sin"):
            self.__delattr__("cached_sin")
            self.__delattr__("cached_cos")