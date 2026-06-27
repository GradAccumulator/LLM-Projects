import torch, torch.nn as nn, math
from torch import Tensor
from omegaconf import DictConfig
from ..utils import dev_utils
from .linear import Linear
from .rope import RoPE
from .softmax import Softmax
from .dropout import Dropout

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        embed_dim   :int, 
        num_heads   :int, 
        dropout     :float|int, 
        init_cfg    :DictConfig|dict=None,
        bias        :bool       =True,
        use_RoPE    :bool       =True,
        RoPE_base   :int|float  =None
    ):
        '''```
        init_cfg = {
            "qkv_linear": {
                "weight": {
                    "method":...
                },
                "bias": {
                    "method":...
                }
            },
            "output_linear": {
                "weight": {
                    "method":...
                },
                "bias": {
                    "method":...
                }
            }
        }
        ```'''
        
        super().__init__()
        dev_utils.type_check(
            ("embed_dim"    , embed_dim     , int),
            ("num_heads"    , num_heads     , int),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            ("dropout"      , dropout       , float|int),
            ('bias'         , bias          , bool),
            ('use_RoPE'     , use_RoPE      , bool),
            ('RoPE_base'    , RoPE_base     , int|float|None)
            ,func_name="MultiHeadAttention.__init__()"
        )
        if embed_dim%num_heads != 0:
            raise ValueError("<MultiHeadAttention.__init__()> embed_dim은 num_heads와 나누어 떨어져야 합니다.")
        if not 0<=dropout<1:
            raise ValueError("<MultiHeadAttention.__init__()> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다.")
        
        self._num_heads = num_heads
        self._d_head = embed_dim//num_heads

        init_cfg = dev_utils.make_dictconfig(init_cfg,default={
                "qkv_linear": None,
                "output_linear": None
            }
        )
        
        self.qkv_linear = Linear(embed_dim, embed_dim*3, init_cfg=init_cfg.qkv_linear, use_bias=bias)
        self.out_linear = Linear(embed_dim, embed_dim, init_cfg=init_cfg.output_linear, use_bias=bias)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)
        self.cached_mask:Tensor
        
        self._use_RoPE = use_RoPE
        if use_RoPE:
            self.RoPE = RoPE(RoPE_base)
            
    def _qkv_projection(self, x:Tensor)->tuple[int,int, Tensor, Tensor, Tensor]:
        #x.shape == (B,T,D)
        B = x.size(0)
        T = x.size(1)
        QKV:Tensor = self.qkv_linear(x)
        #QKV.shape == (B,T,D*3)
        Q,K,V = QKV.chunk(3, dim=-1)
        #Q,K,V shape == (B,T,D)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1,2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1,2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1,2)
        #Q, K, V shape == (B, H, T, D)
        
        return B,T, Q,K,V
    
    def _apply_mask(self, scores:Tensor, T:int, device:torch.device)->Tensor:
        if not hasattr(self, 'cached_mask') or self.cached_mask.shape != (T,T):
            self.register_buffer(
                'cached_mask',
                torch.triu(
                    torch.ones(T,T, device=device, dtype=torch.bool), 
                    diagonal=1
                ),
                persistent=False
            )
        scores = scores.masked_fill(self.cached_mask, -float('inf'))
        
        return scores
    
    def _attention(self, scores:Tensor, V:Tensor, B:int, T:int)->Tensor:
        #scores.shape == (B,H,T,T), V.shape == (B,H,T,D)
        weights = self.softmax(scores)
        drop = self.dropout(weights)
        out = drop@V
        out = out.transpose(1, 2).reshape(B,T,self.embed_dim)
        
        return out
    
    def forward(self, x:Tensor) -> Tensor:
        device = x.device
        
        B,T, Q,K,V = self._qkv_projection(x)

        if self.use_RoPE:
            Q,K = self.RoPE(Q, K)
        
        scores = Q@K.transpose(-1, -2)/math.sqrt(self.d_head)
        
        scores = self._apply_mask(scores, T, device)
        
        out = self._attention(scores, V, B, T)
        out = self.out_linear(out)
        
        return out
    
    @property
    def embed_dim(self): return self.qkv_linear.in_features
    @property
    def d_head(self): return self._d_head
    @property
    def num_heads(self): return self._num_heads
    @property
    def use_RoPE(self):return self._use_RoPE