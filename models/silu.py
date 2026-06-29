import torch, torch.nn as nn
from torch import Tensor

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:Tensor)->Tensor:
        return x/(1+(-x).exp())