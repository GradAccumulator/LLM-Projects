import torch, torch.nn as nn
from torch     import Tensor
from omegaconf import DictConfig

from utils     import nn_utils, dev_utils

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings  :int, 
        embed_dim       :int, 
        init_cfg        :DictConfig|dict=None
    ):
        '''```
        init_cfg = {
            "method":...
        }
        ```'''
        
        super().__init__()
        dev_utils.type_check(
            ('num_embeddings', num_embeddings, int),
            ('embed_dim'     , embed_dim     , int),
            ('init_cfg'      , init_cfg      , DictConfig|dict|None)
            ,func_name="Embedding.__init__()"
        )
        if num_embeddings <= 0:
            raise ValueError("<Embedding.__init__()> num_embeddings는 양의 정수여야 합니다.")
        if embed_dim <= 0:
            raise ValueError("<Embedding.__init__()> embed_dim은 양의 정수여야 합니다.")
        
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "method":"normal",
            "std":0.02
        })
        
        dev_utils.check_dictconfig(
            init_cfg,
            ("method",),
            "Embedding.__init__()"
        )
        
        self._weight = nn.Parameter(
            nn_utils.init_tensor(num_embeddings, embed_dim, init_cfg=init_cfg)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        #x.shape == (B, T)
        return self.weight[x.long()]
    
    @property
    def weight(self): return self._weight
    @property
    def embed_dim(self): return self.weight.size(1)
    @property
    def num_embeddings(self): return self.weight.size(0)