import torch, torch.nn as nn
from torch import Tensor
from typing import Literal
from typing import overload
from omegaconf import DictConfig

from .rope import RoPE
from .matmul import matmul
from configs import runtime as rt
from utils import dev_utils
from .layer_norm import LayerNorm
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .attention import MultiHeadAttention
from .kv_cache import KVCache


class Transformer(nn.Module):
    @overload
    def __init__(self, cfg: DictConfig | dict):
        """```
        cfg = {
            "model": {
                "vocab_size": int,
                "max_seq_len": int,
                "num_layers": int,
                "embed_dim": int,
                "ffn_dim": int,
                "dropout": float|int,
                "bias": bool,
                "attention": {
                    "use_rope":bool,
                    "positional_embedding": Literal["rope", "learnable"],
                    "dropout": float|int,
                    "RoPE": {
                        "base": int|float
                    },
                    "num_kv_heads":int,
                    "num_q_heads":int,
                }
            },
            "init" : {
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
            }
        }
        ```"""
        ...

    @overload
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_kv_heads: int,
        embed_dim: int,
        ffn_dim: int,
        num_q_heads: int | None = None,
        dropout: float | int = 0.0,
        attn_dropout: float | int = None,
        max_seq_len: int = None,
        norm_eps: float = 1e-5,
        ffn_bias: bool = False,
        attn_bias: bool = False,
        norm_bias: bool = True,
        use_RoPE: bool = True,
        RoPE_base: int | float = None,
        ffn: str = "swiglu",
        activation: str = "silu",
        init_cfg: DictConfig | dict = None,
    ):
        """```
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
        ```"""
        ...

    def __init__(
        self,
        vocab_size: int,
        num_layers: int = None,
        num_kv_heads: int = None,
        num_q_heads: int = None,
        embed_dim: int = None,
        ffn_dim: int = None,
        dropout: float | int = 0.0,
        attn_dropout: float | int = None,
        max_seq_len: int = None,
        norm_eps: float = 1e-5,
        ffn_bias: bool = False,
        attn_bias: bool = False,
        norm_bias: bool = True,
        use_RoPE: bool = True,
        RoPE_base: int | float = None,
        ffn: str = "swiglu",
        activation: str = "silu",
        init_cfg: DictConfig | dict = None,
    ):
        """```
        init_cfg = {
            "embedding": {
                "method":...
            },
            (learnable positional embedding을 사용할 경우)
            "pos_embedding": {
                "method":...
            },
            "final_layer_norm": {
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
        ```"""
        super().__init__()

        if isinstance(vocab_size, DictConfig):
            cfg = vocab_size
            model_cfg = cfg.model
            init_cfg = cfg.init

            vocab_size = model_cfg.vocab_size
            num_layers = model_cfg.num_layers
            num_kv_heads = model_cfg.attention.num_kv_heads
            num_q_heads = model_cfg.attention.num_q_heads
            embed_dim = model_cfg.embed_dim
            dropout = model_cfg.dropout
            max_seq_len = model_cfg.max_seq_len

            ffn = model_cfg.ffn.type
            activation = model_cfg.ffn.activation
            ffn_dim = model_cfg.ffn.dim
            ffn_bias = model_cfg.ffn.bias

            use_RoPE = model_cfg.attention.use_rope
            attn_dropout = model_cfg.attention.dropout
            RoPE_base = model_cfg.attention.rope.base
            attn_bias = model_cfg.attention.bias

            norm_eps = model_cfg.layernorm.eps
            norm_bias = model_cfg.layernorm.bias

        func_name = "Transformer.__init__()"
        dev_utils.type_check(
            ("num_layers", num_layers, int),
            ("num_kv_heads", num_kv_heads, int),
            ("num_q_heads", num_q_heads, int | None),
            ("vocab_size", vocab_size, int),
            ("norm_eps", norm_eps, float),
            ("ffn_bias", ffn_bias, bool),
            ("attn_bias", attn_bias, bool),
            ("norm_bias", norm_bias, bool),
            ("use_RoPE", use_RoPE, bool),
            ("activation", activation, str),
            ("ffn", ffn, str),
            ("embed_dim", embed_dim, int | None),
            ("ffn_dim", ffn_dim, int | None),
            ("max_seq_len", max_seq_len, int | None),
            ("dropout", dropout, float | int),
            ("RoPE_base", RoPE_base, float | int | None),
            ("attn_dropout", attn_dropout, float | int | None),
            ("init_cfg", init_cfg, DictConfig | dict | None),
            func_name=func_name,
        )
        init_cfg = dev_utils.make_dictconfig(
            init_cfg,
            default={
                "embedding": None,
                "transformer_block": None,
                "final_layer_norm": None,
            },
        )
        dev_utils.check_dictconfig(
            init_cfg,
            ("embedding", "transformer_block", "final_layer_norm"),
            func_name=func_name,
        )
        if embed_dim is None:
            if num_q_heads is not None:
                embed_dim = num_q_heads * 64
            else:
                embed_dim = num_kv_heads * 64
        if ffn_dim is None:
            if ffn.lower() == "swiglu":
                ffn_dim = int(embed_dim * 8 / 3)
            else:
                ffn_dim = embed_dim * 4

        self.embedding = Embedding(vocab_size, embed_dim, init_cfg=init_cfg.embedding)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_kv_heads=num_kv_heads,
                    num_q_heads=num_q_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    norm_eps=norm_eps,
                    ffn_bias=ffn_bias,
                    attn_bias=attn_bias,
                    norm_bias=norm_bias,
                    use_RoPE=use_RoPE,
                    RoPE_base=RoPE_base,
                    ffn=ffn,
                    activation=activation,
                    init_cfg=init_cfg.transformer_block,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = LayerNorm(
            self.embed_dim,
            eps=self.norm_eps,
            bias=norm_bias,
            init_cfg=init_cfg.final_layer_norm,
        )

        if not use_RoPE:
            if max_seq_len is None:
                raise ValueError(
                    "<Transformer.__init__()> max_seq_len은 learnable positional embedding을 사용할 경우 필수입니다."
                )
            self._pos_embedding = Embedding(max_seq_len, embed_dim, init_cfg=init_cfg.pos_embedding)
        self._max_seq_len = max_seq_len

        self.cached_causal_mask: Tensor
        self.k_cache = KVCache(self.num_layers, self.max_seq_len, self.d_head, self.transformer_blocks[0].attention.use_gqa, self.num_kv_heads)
        self.v_cache = KVCache(self.num_layers, self.max_seq_len, self.d_head, self.transformer_blocks[0].attention.use_gqa, self.num_kv_heads)
        self.kv_cache_idx = 0

    def make_cached_tensors(self, x: Tensor):
        """x.shape == (B, T, D)"""
        T = x.size(1) + self.kv_cache_idx*torch.is_inference_mode_enabled()
        device = x.device
        dtype = torch.get_autocast_dtype(device.type)

        if not hasattr(self, "cached_causal_mask") or self.cached_causal_mask.shape != (T, T):
            self.register_buffer(
                "cached_causal_mask",
                MultiHeadAttention.make_causal_mask(T, device),
                persistent=False,
            )

        need_new_cached_sin_cos = not hasattr(self, "cached_sin") or (
            self.cached_sin.shape != (T, self.d_head // 2) and self.cached_sin.size(-2) < T
        )
        if need_new_cached_sin_cos:
            if self.use_RoPE:
                rope: RoPE = self.transformer_blocks[0].attention.RoPE
                sin, cos = RoPE.compute_sin_cos(T, self.d_head, device, dtype, rope.base)
                self.register_buffer("cached_sin", sin, persistent=False)
                self.register_buffer("cached_cos", cos, persistent=False)

        if torch.is_inference_mode_enabled():
            if self.k_cache.is_empty():
                self.k_cache.allocate(device=x.device, dtype=x.dtype)
            if self.v_cache.is_empty():
                self.v_cache.allocate(device=x.device, dtype=x.dtype)
        elif not self.k_cache.is_empty() or not self.v_cache.is_empty():
            self.empty_kv_cache()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, T)
        x = self.embedding(x)
        if not self.use_RoPE:
            pos_ids = torch.arange(x.size(1), device=x.device)
            x = x + self._pos_embedding(pos_ids)[None, :, :]
        self.make_cached_tensors(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(
                x,
                mask=self.cached_causal_mask,
                cached_sin=self.cached_sin,
                cached_cos=self.cached_cos,
                k_cache=self.k_cache[i],
                v_cache=self.v_cache[i],
                start_idx=self.kv_cache_idx,
            )
        x = self.final_layer_norm(x)

        if torch.is_inference_mode_enabled():
            self.kv_cache_idx += x.size(1)

        return matmul(x, self.embedding.weight.T)

    def empty_kv_cache(self):
        self.k_cache.empty_cache()
        self.v_cache.empty_cache()
        self.kv_cache_idx: int = 0

    def allocate_kv_cache(self):
        self.k_cache.allocate()
        self.v_cache.allocate()

    def expand_kv_cache(self, max_seq_len: int = 0):
        func_name = "Transformer.expand_kv_cache()"
        if self.positional_embedding == "learnable":
            raise ValueError(f"<{func_name}> 위치 임베딩 방식이 learnable이라면 kv cache의 크기를 늘릴 수 없습니다.")
        if max_seq_len < 0 or not isinstance(max_seq_len, int):
            raise ValueError(f"<{func_name}> max_seq_len은 [0,inf)의 정수여야 합니다. 현재: {max_seq_len}")
        if max_seq_len <= self.max_seq_len:
            raise ValueError(f"<{func_name}> 새로 설정되는 max_seq_len은 반드시 이전의 max_seq_len보다 커야 합니다.")
        if max_seq_len == 0:
            max_seq_len = self.max_seq_len * 1.5

        self.max_seq_len = max_seq_len
        self.k_cache.expand(self.max_seq_len, self.kv_cache_idx)
        self.v_cache.expand(self.max_seq_len, self.kv_cache_idx)

    @property
    def ffn(self):
        return self.transformer_blocks[0].ffn_type

    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def act_fn(self):
        return self.transformer_blocks[0].act_fn

    @property
    def d_head(self):
        return self.transformer_blocks[0].attention.d_head

    @property
    def ffn_dim(self):
        return self.transformer_blocks[0].ffn.ffn_dim

    @property
    def norm_eps(self):
        return self.transformer_blocks[0].norm_eps

    @property
    def use_bias(self):
        return self.transformer_blocks[0].use_bias

    @property
    def use_RoPE(self):
        return self.transformer_blocks[0].use_RoPE

    @property
    def embed_dim(self):
        return self.embedding.embed_dim

    @property
    def num_kv_heads(self):
        return self.transformer_blocks[0].num_kv_heads

    @property
    def num_q_heads(self):
        return self.transformer_blocks[0].num_q_heads

    @property
    def dropout_p(self):
        return self.transformer_blocks[0].dropout_p

    @property
    def RoPE_base(self):
        return self.transformer_blocks[0].RoPE_base

    @property
    def vocab_size(self):
        return self.embedding.vocab_size

    @property
    def num_layers(self):
        return len(self.transformer_blocks)

    @property
    def attn_dropout_p(self):
        return self.transformer_blocks[0].attn_dropout_p

    @property
    def positional_embedding(self):
        return "RoPE" if self.use_RoPE else "learnable"

    @property
    def max_seq_len(self):
        return self._max_seq_len
