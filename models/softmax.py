import torch, torch.nn as nn
from torch import Tensor

from utils import dev_utils
from custom_functions import SoftmaxFunction


class Softmax(nn.Module):
    def __init__(self, dim: int = -1, temperature: int | float = 1.0):
        super().__init__()
        dev_utils.type_check(
            ("dim", dim, int),
            ("temperature", temperature, float | int),
            func_name="Softmax.__init__()",
        )
        if temperature <= 0:
            raise ValueError("<Softmax.__init__()> Softmax의 temperature은 양수여야 합니다.")

        self._dim = dim
        self._temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        return SoftmaxFunction.apply(torch.is_grad_enabled(), x, self.temperature, self.dim)

    @property
    def dim(self):
        return self._dim

    @property
    def temperature(self):
        return self._temperature
