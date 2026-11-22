import torch, torch.nn as nn
from torch import Tensor
from utils import dev_utils

from custom_functions import DropoutFunction


class Dropout(nn.Module):
    def __init__(self, p: int | float, custom:bool = True):
        super().__init__()
        func_name = "Dropout.__init__()"
        dev_utils.type_check(("p", p, int | float), func_name=func_name)
        if not 0 <= p < 1:
            raise ValueError(f"<{func_name}> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다.")

        self._p = p
        self._custom = custom
        if not custom:
            self._dropout = nn.Dropout(p=p)

    def forward(self, x: Tensor) -> Tensor:
        if self.custom:
            return DropoutFunction.apply(torch.is_grad_enabled(), x, self.p * self.training)
        else:
            return self._dropout(x)

    @property
    def p(self):
        return self._p
    
    @property
    def custom(self) -> bool:
        return self._custom
