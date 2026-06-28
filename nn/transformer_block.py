import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from utils import dev_utils
from .attention import MultiHeadAttention
from .dropout import Dropout
from .layer_norm import LayerNorm
from .ffn import FFN
from typing import Literal
from configs import runtime

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim    : int,
        num_heads    : int,
        ffn_dim      : int,
        dropout      : float|int,
        init_cfg     : DictConfig|dict=None,
        attn_dropout : float|int=None,
        norm_eps     : float = 1e-5,
        bias         : bool = True,
        use_RoPE     : bool = True,
        RoPE_base    : int|float=None,
        ffn          : Literal['swiglu', 'mlp']='swiglu',
        activation   : str = "silu",
    ):
        super().__init__()
        dev_utils.type_check(
            ("embed_dim"    , embed_dim     , int),
            ("num_heads"    , num_heads     , int),
            ("ffn_dim"      , ffn_dim       , int),
            ("dropout"      , dropout       , float|int),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            ("attn_dropout" , attn_dropout  , float|None|int),
            ("norm_eps"     , norm_eps      , float),
            ("bias"         , bias          , bool),
            ("use_RoPE"     , use_RoPE      , bool),
            ("RoPE_base"    , RoPE_base     , int|float|None),
            ("activation"   , activation    , str)
            ,func_name="TransformerBlock.__init__()"
        )
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "layer_norm":None,
            "attention":None,
            "ffn":None
        })
        dev_utils.check_dictconfig(
            init_cfg,
            ("layer_norm", "attention", "ffn"),
            "TransformerBlock.__init__()"
        )
        if attn_dropout is None:
            attn_dropout = dropout
        
        self.ln1 = LayerNorm(
            embed_dim,
            init_cfg    =init_cfg.layer_norm,
            eps         =norm_eps
        )
        self.attention = MultiHeadAttention(
            embed_dim, 
            num_heads, 
            attn_dropout,
            init_cfg    =init_cfg.attention, 
            bias        =bias, 
            use_RoPE    =use_RoPE, 
            RoPE_base   =RoPE_base
        )
        self._dropout = Dropout(dropout)
        
        self.ln2 = LayerNorm(
            embed_dim,
            init_cfg    =init_cfg.layer_norm,
            eps         =norm_eps
        )
        self.ffn = FFN(
            embed_dim,
            ffn_dim,
            ffn,
            activation,
            init_cfg    =init_cfg.ffn,
            use_bias    =bias
        )
    
    def forward(
        self, 
        x:Tensor, 
        mask:Tensor=None, 
        cached_sin:Tensor=None, 
        cached_cos:Tensor=None
    )->Tensor:
        x = x + self.dropout(
            self.attention(
                self.ln1(x), 
                mask=mask, 
                cached_sin=cached_sin, 
                cached_cos=cached_cos
            )
        )
        x = x + self.dropout(
            self.ffn(self.ln2(x))
        )
        return x

    @property
    def embed_dim(self): return self.ln1.normalized_shape[-1]
    @property
    def num_heads(self): return self.attention.num_heads
    @property
    def ffn_dim(self):return self.ffn.ffn_dim
    @property
    def use_bias(self): return self.attention.use_bias and self.ffn.use_bias
    @property
    def use_RoPE(self): return self.attention.use_RoPE
    @property
    def RoPE_base(self): return self.attention.RoPE_base
    @property
    def act_fn(self): return self.ffn.act_fn
    @property
    def activation_name(self): return self.ffn.activation_name
    @property
    def ffn_type(self): return self.ffn.ffn_type
    @property
    def attn_dropout(self): return self.attention.dropout
    @property
    def attn_dropout_p(self): return self.attention.dropout_p
    @property
    def dropout(self): return self._dropout
    @property
    def dropout_p(self): return self._dropout.p
    @property
    def norm_eps(self): return self.ln1.eps