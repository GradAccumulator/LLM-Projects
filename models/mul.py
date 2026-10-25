import torch
from torch import Tensor

from custom_functions import MulFunction

def mul(a: Tensor, b: Tensor) -> Tensor:
    return MulFunction.apply(torch.is_grad_enabled(), a, b)
