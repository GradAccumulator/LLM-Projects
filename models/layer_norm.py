import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from configs import runtime as rt
from utils import nn_utils, dev_utils
from custom_functions import LayerNormFunction


class LayerNorm(nn.Module):
    def __init__(
        self,
        *normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        init_cfg: DictConfig | dict = None,
    ):
        """```
        init_cfg = {
            "gamma": {
                "method":...
            },
            "beta": {
                "method":...
            }
        }
        ```"""
        super().__init__()
        func_name = "LayerNorm.__init__()"
        dev_utils.type_check(
            ("bias", bias, bool),
            ("eps", eps, float),
            ("init_cfg", init_cfg, DictConfig | dict | None),
            func_name=func_name,
        )
        for i, value in enumerate(normalized_shape):
            if not isinstance(value, int):
                raise TypeError(
                    f"<{func_name}> normalized_shape의 타입이 부적절합니다. normalized_shape의 모든 값은 int여야 합니다. "
                    f"{i}번째 값의 타입:{type(value)}"
                )
            if value <= 0:
                raise ValueError(
                    f"<{func_name}> normalized_shape의 모든 값은 양의 정수여야 합니다. " f"{i}번째 값: {value}"
                )
        if len(normalized_shape) == 0:
            raise ValueError(f"<{func_name}> normalized_shape는 최소 1개 이상이어야 합니다.")
        if eps <= 0:
            raise ValueError(f"<{func_name}> eps는 반드시 양수여야 합니다.")

        self._eps = eps
        self._use_bias = bias
        self._normalized_dims = tuple(range(-len(normalized_shape), 0))

        init_cfg = dev_utils.make_dictconfig(
            init_cfg, default={"gamma": {"method": "ones"}, "beta": {"method": "zeros"}}
        )
        dev_utils.check_dictconfig(
            init_cfg,
            ("gamma", "beta", "gamma.method", "beta.method"),
            func_name=func_name,
        )

        self._gamma = nn_utils.init_parameter(*normalized_shape, init_cfg=init_cfg.gamma)
        if self.use_bias:
            self._beta = nn_utils.init_parameter(*normalized_shape, init_cfg=init_cfg.beta)

    def forward_debug(self, x: Tensor):
        # x.shape == (B, T, D), self.normalized_shape == (D,)
        func_name = "LayerNorm.forward()"
        if len(self.normalized_shape) > x.ndim:
            raise ValueError(
                f"<{func_name}> 입력 텐서의 shape의 ndim은 {self.normalized_shape.ndim} 이상이어야 합니다."
                f"(현재: {x.ndim})"
            )
        if self.normalized_shape != x.shape[-len(self.normalized_shape) :]:
            raise ValueError(
                f"<{func_name}> 입력 텐서의 shape는 {tuple(self.normalized_shape)}로 끝나야 합니다. "
                f"(현재: {tuple(x.shape)})"
            )

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, T, D)
        if rt.DEBUG_CHECKS:
            self.forward_debug(x)

        return LayerNormFunction.apply(
            torch.is_grad_enabled(), x, self.gamma, self.beta, self.normalized_dims, self.eps
        )

    @property
    def eps(self):
        return self._eps

    @property
    def beta(self):
        return self._beta if self.use_bias else None

    @property
    def gamma(self):
        return self._gamma

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def normalized_dims(self):
        return self._normalized_dims

    @property
    def normalized_shape(self):
        return self.gamma.shape
