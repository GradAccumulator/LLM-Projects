import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import nn_utils, dev_utils

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, init_cfg:DictConfig|dict):
        super().__init__()
        dev_utils.type_check(
            ('num_embeddings', num_embeddings, int),
            ('embedding_dim' , embedding_dim , int),
            ('init_cfg'      , init_cfg      , DictConfig|dict)
            ,func_name="Embedding.__init__()"
        )
        if num_embeddings <= 0:
            raise ValueError("<Embedding.__init__()> num_embeddings는 양의 정수여야 합니다.")
        if embedding_dim <= 0:
            raise ValueError("<Embedding.__init__()> embedding_dim은 양의 정수여야 합니다.")
        
        self._weight = nn.Parameter(
            nn_utils.init_tensor(num_embeddings, embedding_dim, init_cfg=init_cfg)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return self.weight[x.long()]
    
    @property
    def num_embeddings(self): return self.weight.size(0)
    @property
    def embedding_dim(self): return self.weight.size(1)
    @property
    def weight(self): return self._weight