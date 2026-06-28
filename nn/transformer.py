import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from utils import dev_utils
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .attention import MultiHeadAttention
from .rope import RoPE
from configs import runtime
from typing import Literal
from utils import nn_utils

class Transformer(nn.Module):
    def __init__(
        self,
        num_layers  : int,
        embed_dim   : int,
        num_heads   : int,
        ffn_dim     : int,
        vocab_size  : int,
        dropout     : float|int,
        init_cfg    : DictConfig|dict=None,
        attn_dropout: float|int = None,
        norm_eps    : float     = 1e-5,
        bias        : bool      = True,
        use_RoPE    : bool      = True,
        RoPE_base   : int|float = None,
        ffn         : str       = "swiglu",
        activation  : str       = "silu",
        max_seq_len : int       = None
    ):
        '''```
        init_cfg = {
            "embedding": {
                "method":...
            },
            (learnable positional embedding을 사용할 경우)
            "pos_embedding": {
                "method":...
            },
            "transformer_block": {
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
            ("num_layers"   , num_layers    , int),
            ("embed_dim"    , embed_dim     , int),
            ("num_heads"    , num_heads     , int),
            ("ffn_dim"      , ffn_dim       , int),
            ("vocab_size"   , vocab_size    , int),
            ("dropout"      , dropout       , float|int),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            ("attn_dropout" , attn_dropout  , float|int|None),
            ("norm_eps"     , norm_eps      , float),
            ("bias"         , bias          , bool),
            ("use_RoPE"     , use_RoPE      , bool),
            ("RoPE_base"    , RoPE_base     , int|float|None),
            ("activation"   , activation    , str),
            ("ffn"          , ffn           , str),
            ("max_seq_len"  , max_seq_len   , int|None)
            ,func_name="Transformer.__init__()"
        )
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "embedding":None,
            "transformer_block":None
        })
        dev_utils.check_dictconfig(
            init_cfg,
            ("embedding", "transformer_block"),
            "Transformer.__init__()"
        )

        self.embedding = Embedding(
            vocab_size,
            embed_dim,
            init_cfg=init_cfg.embedding
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(
                embed_dim   =embed_dim,
                num_heads   =num_heads,
                ffn_dim     =ffn_dim,
                dropout     =dropout,
                init_cfg    =init_cfg.transformer_block,
                attn_dropout=attn_dropout,
                norm_eps    =norm_eps,
                bias        =bias,
                use_RoPE    =use_RoPE,
                RoPE_base   =RoPE_base,
                ffn         =ffn,
                activation  =activation
            ) for _ in range(num_layers)]
        )
        if not use_RoPE:
            if max_seq_len is None:
                raise ValueError("<Transformer.__init__()> max_seq_len은 learnable positional embedding을 사용할 경우 필수입니다.")
            self._pos_embedding = Embedding(
                max_seq_len,
                embed_dim,
                init_cfg=init_cfg.pos_embedding
            )
        
        self.cached_causal_mask:Tensor

    
    def make_cached_tensors(self, x:Tensor):
        '''x.shape == (B, T, D)'''
        T = x.size(1)
        device = x.device
        dtype = x.dtype
        
        if not hasattr(self, 'cached_causal_mask') or self.cached_causal_mask.shape != (T,T) :
            self.register_buffer(
                'cached_causal_mask',
                MultiHeadAttention.make_causal_mask(T, device),
                persistent=False
            )
        
        need_new_cached_sin_cos = (
            not hasattr(self, 'cached_sin')
            or self.cached_sin.shape != (T, self.embed_dim // 2)
        )
        if runtime.DEBUG_CHECKS:
            need_new_cached_sin_cos = need_new_cached_sin_cos or (
                not hasattr(self, 'cached_cos')
                or self.cached_cos.shape != (T, self.embed_dim // 2)
            )
        if need_new_cached_sin_cos:
            if self.use_RoPE:
                rope:RoPE = self.transformer_blocks[0].attention.RoPE
                sin, cos = rope.compute_sin_cos(T, self.d_head, device, dtype, rope.base)
                self.register_buffer("cached_sin", sin, persistent=False)
                self.register_buffer("cached_cos", cos, persistent=False)
    
    def forward(self, x:Tensor) -> Tensor:
        #x.shape == (B, T)

        x = self.embedding(x)
        if not self.use_RoPE:
            pos_ids = torch.arange(x.size(1), device=x.device)
            x = x + self._pos_embedding(pos_ids)[None, :, :]
        self.make_cached_tensors(x)
        for i,block in enumerate(self.transformer_blocks):
            x = block(
                x,
                mask=self.cached_causal_mask,
                cached_sin=self.cached_sin,
                cached_cos=self.cached_cos
            )
        return x@self.embedding.weight.T
    
    @property
    def num_layers(self): return len(self.transformer_blocks)
    @property
    def embed_dim(self): return self.embedding.embed_dim
    @property
    def num_heads(self): return self.transformer_blocks[0].num_heads
    @property
    def d_head(self): return self.embed_dim // self.num_heads
    @property
    def ffn_dim(self): return self.transformer_blocks[0].ffn.ffn_dim
    @property
    def vocab_size(self): return self.embedding.vocab_size
    @property
    def dropout_p(self): return self.transformer_blocks[0].dropout_p
    @property
    def attn_dropout_p(self): return self.transformer_blocks[0].attn_dropout_p
    @property
    def norm_eps(self): return self.transformer_blocks[0].norm_eps
    @property
    def use_bias(self): return self.transformer_blocks[0].use_bias
    @property
    def use_RoPE(self): return self.transformer_blocks[0].use_RoPE
    @property
    def RoPE_base(self): return self.transformer_blocks[0].RoPE_base
    @property
    def ffn(self): return self.transformer_blocks[0].ffn_type
    @property
    def act_fn(self): return self.transformer_blocks[0].act_fn
    @property
    def positional_embedding(self): return "RoPE" if self.use_RoPE else "learnable"