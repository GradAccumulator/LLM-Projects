import torch, torch.nn as nn
from torch import Tensor

from custom_functions import SiLUFunction


class SiLU(nn.Module):
    def __init__(self, custom:bool = True):
        super().__init__()
        self.custom = custom

    def forward(self, x: Tensor) -> Tensor:
        if self.custom:
            return SiLUFunction.apply(torch.is_grad_enabled(), x)
        return nn.functional.silu(x)
