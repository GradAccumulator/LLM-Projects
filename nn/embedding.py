import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import nn_utils, dev_utils

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, cfg:DictConfig):
        super().__init__()
        dev_utils(
            ('num_embeddings', num_embeddings, int),
            ('embedding_dim' , embedding_dim , int),
            ('cfg'           , cfg           , DictConfig)
        )
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._init_cfg = cfg.init.embedding
        
        self._weight = nn.Parameter(
            nn_utils.init_tensor(self.num_embeddings, self.embedding_dim, init_cfg=self.init_cfg)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return self.weight[x.long()]
    
    @property
    def num_embeddings(self): return self._num_embeddings
    @property
    def embedding_dim(self): return self._embedding_dim
    @property
    def init_cfg(self): return self._init_cfg
    @property
    def weight(self): return self._weight