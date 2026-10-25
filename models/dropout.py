import torch, torch.nn as nn
from torch import Tensor
from utils import dev_utils

from custom_functions import DropoutFunction


class Dropout(nn.Module):
    def __init__(self, p: int | float):
        super().__init__()
        func_name = "Dropout.__init__()"
        dev_utils.type_check(("p", p, int | float), func_name=func_name)
        if not 0 <= p < 1:
            raise ValueError(f"<{func_name}> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다.")

        self._p = p

    def forward(self, x: Tensor) -> Tensor:
        return DropoutFunction.apply(torch.is_grad_enabled(), x, self.p * self.training)

    @property
    def p(self):
        return self._p
