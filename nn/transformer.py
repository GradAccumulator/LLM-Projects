import torch, torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from ..utils import dev_utils
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .attention import MultiHeadAttention
import warnings

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
        attn_dropout: float|int=None,
        norm_eps    : float = 1e-5,
        bias        : bool = True,
        use_RoPE    : bool = True,
        RoPE_base   : int|float=None,
        ffn         : str = "swiglu",
        activation  : str = "silu"
    ):
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
            ("ffn"          , ffn           , str)
            ,func_name="Transformer.__init__()"
        )
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            "embedding":None,
            "transformer_block":None
        })

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
        
        self.cached_causal_mask:Tensor
    
    def forward(self, x:Tensor) -> Tensor:
        #x.shape == (B, T)
        T = x.size(1)
        if not (hasattr(self, 'cached_causal_mask') and self.cached_causal_mask.shape == (T,T)): 
            self.register_buffer(
                'cached_causal_mask',
                MultiHeadAttention.make_causal_mask(T, x.device),
                persistent=False
            )

        x = self.embedding(x)
        for i,block in enumerate(self.transformer_blocks):
            x = block(
                x,
                mask=self.cached_causal_mask
            )
        return x@self.embedding.weight.T
    
    @property
    def num_layers(self): return len(self.transformer_blocks)
    @property
    def embed_dim(self): return self.embedding.embed_dim
    @property
    def num_heads(self): return self.transformer_blocks[0].num_heads
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