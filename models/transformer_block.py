import torch, torch.nn as nn
from torch       import Tensor
from typing      import Literal
from omegaconf   import DictConfig

from .ffn        import FFN
from .dropout    import Dropout
from utils       import dev_utils
from .layer_norm import LayerNorm
from .attention  import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim    : int,
        num_heads    : int,
        ffn_dim      : int,
        dropout      : float|int,
        attn_dropout : float|int=None,
        norm_eps     : float = 1e-5,
        ffn_bias     : bool = False,
        attn_bias    : bool = False,
        norm_bias    : bool = True,
        use_RoPE     : bool = True,
        RoPE_base    : int|float=None,
        ffn          : Literal['swiglu', 'mlp']='swiglu',
        activation   : str = "silu",
        init_cfg     : DictConfig|dict=None,
    ):
        '''```
        init_cfg = {
            "layer_norm": {
                "alpha": {
                    "method":...
                },
                "beta": {
                    "method":...
                }
            },
            "attention": {
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
            },
            "ffn": {
                "linear1": {
                    "weight": {
                        "method":...
                    },
                    "bias": {
                        "method":...
                    }
                },
                "linear2": {
                    "weight": {
                        "method":...
                    },
                    "bias": {
                        "method":...
                    }
                }
            }
        }
        ```'''
        super().__init__()
        dev_utils.type_check(
            ("activation"   , activation    , str),
            ("embed_dim"    , embed_dim     , int),
            ("num_heads"    , num_heads     , int),
            ("ffn_dim"      , ffn_dim       , int),
            ("ffn_bias"     , ffn_bias      , bool),
            ("attn_bias"    , attn_bias     , bool),
            ("norm_bias"    , norm_bias     , bool),
            ("use_RoPE"     , use_RoPE      , bool),
            ("norm_eps"     , norm_eps      , float),
            ("dropout"      , dropout       , float|int),
            ("RoPE_base"    , RoPE_base     , float|int|None),
            ("attn_dropout" , attn_dropout  , float|int|None),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            func_name="TransformerBlock.__init__()"
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
            eps         =norm_eps,
            bias        =norm_bias,
            init_cfg    =init_cfg.layer_norm,
        )
        self.attention = MultiHeadAttention(
            embed_dim, 
            num_heads, 
            attn_dropout,
            bias        =attn_bias, 
            use_RoPE    =use_RoPE, 
            RoPE_base   =RoPE_base,
            init_cfg    =init_cfg.attention,
        )
        self._dropout = Dropout(dropout)
        
        self.ln2 = LayerNorm(
            embed_dim,
            bias        =norm_bias,
            eps         =norm_eps,
            init_cfg    =init_cfg.layer_norm,
        )
        
        self.ffn = FFN(
            embed_dim,
            ffn_dim,
            ffn,
            activation,
            init_cfg    =init_cfg.ffn,
            use_bias    =ffn_bias
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
    def act_fn(self): return self.ffn.act_fn
    @property
    def ffn_dim(self):return self.ffn.ffn_dim
    @property
    def dropout(self): return self._dropout
    @property
    def norm_eps(self): return self.ln1.eps
    @property
    def ffn_type(self): return self.ffn.ffn_type
    @property
    def use_bias(self): return self.attention.use_bias and self.ffn.use_bias
    @property
    def use_RoPE(self): return self.attention.use_RoPE
    @property
    def RoPE_base(self): return self.attention.RoPE_base
    @property
    def embed_dim(self): return self.ln1.normalized_shape[-1]
    @property
    def num_heads(self): return self.attention.num_heads
    @property
    def dropout_p(self): return self._dropout.p
    @property
    def attn_dropout(self): return self.attention.dropout
    @property
    def attn_dropout_p(self): return self.attention.dropout_p
    @property
    def activation_name(self): return self.ffn.activation_name