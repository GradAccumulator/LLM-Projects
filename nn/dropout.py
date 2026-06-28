import torch, torch.nn as nn
from torch import Tensor

class Dropout(nn.Module):
    def __init__(self, p:float):
        super().__init__()
        if not 0<=p<1:
            raise ValueError("<Dropout.__init__()> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다.")
        self._p = p
        
    def forward(self, x:Tensor)->Tensor:
        if self.training and self.p > 0:
            mask = torch.rand_like(x, dtype=torch.float16) > self.p
            x = x*mask/(1-self.p)
        return x
    
    @property
    def p(self):return self._p