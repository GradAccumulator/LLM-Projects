import torch, torch.nn as nn
from torch import Tensor

from custom_functions import SiLUFunction


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return SiLUFunction.apply(torch.is_grad_enabled(), x)
