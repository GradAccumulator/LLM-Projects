import torch
from torch import Tensor

from custom_functions import MatmulFunction


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulFunction.apply(torch.is_grad_enabled(), a, b)
