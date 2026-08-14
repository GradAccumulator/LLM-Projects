import torch, torch.nn as nn, math
from torch import Tensor
from omegaconf import DictConfig

from .rope import RoPE
from .matmul import matmul
from .linear import Linear
from .softmax import Softmax
from .dropout import Dropout
from utils import dev_utils
from configs import runtime as rt


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_kv_heads: int,
        dropout: float | int,
        init_cfg: DictConfig | dict | None = None,
        bias: bool = False,
        use_RoPE: bool = True,
        RoPE_base: int | float | None = None,
        num_q_heads: int | None = None,
    ):
        """```
        init_cfg = {
            "qkv_linear": {
                "weight": {
                    "method":...
                },
                "bias": {
                    "method":...
                }
            },
            "out_linear": {
                "weight": {
                    "method":...
                },
                "bias": {
                    "method":...
                }
            }
        }
        ```"""
        super().__init__()
        func_name = "MultiHeadAttention.__init__()"
        dev_utils.type_check(
            ("embed_dim", embed_dim, int),
            ("num_kv_heads", num_kv_heads, int),
            ("num_q_heads", num_q_heads, int | None),
            ("bias", bias, bool),
            ("use_RoPE", use_RoPE, bool),
            ("dropout", dropout, float | int),
            ("RoPE_base", RoPE_base, float | int | None),
            ("init_cfg", init_cfg, DictConfig | dict | None),
            func_name=func_name,
        )
        self._use_gqa = num_q_heads is not None and num_q_heads != num_kv_heads
        if not self.use_gqa:
            if embed_dim % num_kv_heads != 0:
                raise ValueError(f"<{func_name}> embed_dim은 num_kv_heads와 나누어 떨어져야 합니다.")
            num_q_heads = num_kv_heads
        elif self.use_gqa and (embed_dim % num_q_heads != 0):
                raise ValueError(f"<{func_name}> embed_dim은 num_q_heads와 나누어 떨어져야 합니다.")
        
        if not 0 <= dropout < 1:
            raise ValueError(f"<{func_name}> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다.")

        self._num_kv_heads = num_kv_heads
        self._num_q_heads = num_q_heads
        self._d_head = embed_dim // num_q_heads

        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(f"<{func_name}> num_q_heads값이 num_kv_heads값으로 나누어 떨어지지 않습니다.")
        if self.num_q_heads < self.num_kv_heads:
            raise ValueError(f"<{func_name}> num_kv_heads값은 num_q_heads값보다 클 수 없습니다.")

        init_cfg = dev_utils.make_dictconfig(init_cfg, default={"qkv_linear": None, "out_linear": None})
        dev_utils.check_dictconfig(
            init_cfg,
            ("qkv_linear", "out_linear"),
            func_name=func_name,
        )
        if not self.use_gqa:
            self.qkv_linear = Linear(embed_dim, embed_dim * 3, init_cfg=init_cfg.qkv_linear, use_bias=bias)
        else:
            self.kv_linear = Linear(
                embed_dim,
                self.d_head * self.num_kv_heads * 2,
                init_cfg=init_cfg.qkv_linear,
                use_bias=bias,
            )
            self.q_linear = Linear(
                embed_dim,
                embed_dim,
                init_cfg=init_cfg.qkv_linear,
                use_bias=bias,
            )
        self.out_linear = Linear(embed_dim, embed_dim, init_cfg=init_cfg.out_linear, use_bias=bias)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(dropout)
        self._use_bias = bias
        self.cached_mask: Tensor

        self._use_RoPE = use_RoPE
        if use_RoPE:
            self.RoPE = RoPE(RoPE_base)

    def _qkv_projection(self, x: Tensor) -> tuple[int, Tensor, Tensor, Tensor]:
        """return B, Q,K,V"""
        # x.shape == (B,T,D)
        B = x.size(0)
        T = x.size(1)
        if not self.use_gqa:
            QKV: Tensor = self.qkv_linear(x)
            Q, K, V = QKV.chunk(3, dim=-1)

            Q = Q.reshape(B, T, self.num_q_heads, self.d_head).transpose(1, 2)
            K = K.reshape(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
            V = V.reshape(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        else:
            K, V = self.kv_linear(x).chunk(2, dim=-1)
            Q: Tensor = self.q_linear(x)
            Q = (
                Q.reshape(B, T, self.num_q_heads, self.d_head)
                .transpose(1, 2)
                .reshape(
                    B,
                    self.num_kv_heads,
                    self.num_q_heads // self.num_kv_heads,
                    T,
                    self.d_head,
                )
            )
            
            K = K.reshape(B, T, self.num_kv_heads, self.d_head).transpose(1, 2).unsqueeze(2)
            V = V.reshape(B, T, self.num_kv_heads, self.d_head).transpose(1, 2).unsqueeze(2)

        return B, Q, K, V

    @staticmethod
    def make_causal_mask(T: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _apply_mask(self, scores: Tensor, T: int, device: torch.device, mask: Tensor = None) -> Tensor:
        # scores.shape == (B,H,T,T)
        need_new_mask = (mask is None) and not (hasattr(self, "cached_mask") and self.cached_mask.shape == (T, T))
        if need_new_mask:
            self.register_buffer("cached_mask", self.make_causal_mask(T, device), persistent=False)
        scores = scores.masked_fill(mask if mask is not None else self.cached_mask, -float("inf"))

        return scores

    def _attention(self, scores: Tensor, V: Tensor, B: int, T: int) -> Tensor:
        # scores.shape == (B,H,T,T), V.shape == (B,H,T,D)
        weights = self.softmax(scores)
        drop = self.dropout(weights)
        out = matmul(drop, V)
        if self.use_gqa:
            # out.shape == (B,H_q//H_kv,H_kv, T, D)
            out = out.reshape(B, self.num_q_heads, T, self.d_head)
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        return out

    def forward_debug(self, x: Tensor, mask: Tensor, T: int, k_cache:Tensor, v_cache:Tensor, start_idx:int) -> Tensor:
        # x.shape == (B, T, D)
        # mask.shape == (T, T)
        func_name = "MultiHeadAttention.forward()"
        if mask is not None and mask.shape != (T, T):
            raise ValueError(
                f"mask의 shape이 입력 텐서에 맞지 않습니다."
                f"\nmask.shape: {tuple(mask.shape)}, 입력 텐서 shape: {tuple(x.shape)}, 필요한 mask shape: {(T,T)}"
            )
        if x.ndim != 3:
            raise ValueError(
                f"<{func_name}> 입력 x는 (B,T,D) 형태의 3차원 텐서여야 합니다. 현재 x.shape={tuple(x.shape)}"
            )
        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"<{func_name}> 입력 x의 마지막 차원은 embed_dim={self.embed_dim}이어야 합니다. "
                f"현재 x.shape={tuple(x.shape)}"
            )
        if torch.is_inference_mode_enabled():
            if k_cache is None:
                raise ValueError(
                    f"<{func_name}> inference mode가 활성화 돼있을 때는 k_cache가 필요합니다."
                )
            if k_cache.shape[:-2] != (1,1,self.num_kv_heads) or k_cache.size(-1) != self.embed_dim:
                raise ValueError(
                    f"<{func_name}> k_cache의 shape가 부적절합니다."
                    "\n예상 shape= (1, 1, H_kv, T_max, D), "
                    f"현재 shape= {k_cache.shape}"
                )
            if v_cache is None:
                raise ValueError(
                    f"<{func_name}> inference mode가 활성화 돼있을 때는 v_cache가 필요합니다."
                )
            if v_cache.shape[:-2] != (1,1,self.num_kv_heads) or v_cache.size(-1) != self.embed_dim:
                raise ValueError(
                    f"<{func_name}> v_cache의 shape가 부적절합니다."
                    "\n예상 shape= (1, 1, H_kv, T_max, D), "
                    f"현재 shape= {v_cache.shape}"
                )
            if start_idx < 0 or not isinstance(start_idx, int):
                raise ValueError(
                    f"<{func_name}> start_idx는 0 이상의 정수여야 합니다."
                )

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        cached_sin: Tensor = None,
        cached_cos: Tensor = None,
        k_cache: Tensor = None,
        v_cache: Tensor = None,
        start_idx:int = 0,
    ) -> Tensor:
        # x.shape == (B, T, D)
        #k_cache,v_cache.sahpe == (1, 1, H_kv, T_max, D)
        device = x.device
        T = x.size(1)
        if rt.DEBUG_CHECKS:
            self.forward_debug(x, mask, T, k_cache, v_cache, start_idx)

        B, Q, K, V = self._qkv_projection(x)
        # Q,K,V shape == (B, H, T, D)

        if self.use_RoPE:
            Q, K = self.RoPE(Q, K, cached_sin=cached_sin, cached_cos=cached_cos)

        if torch.is_inference_mode_enabled():
            k_cache[:, :, :, start_idx:start_idx+1, :] = K
            v_cache[:, :, :, start_idx:start_idx+1, :] = V

            K = k_cache[:, :, :, :start_idx+1, :]
            V = v_cache[:, :, :, :start_idx+1, :]

        scores = matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head)
        # scores.shape == (B,H,T_q,T_k)

        scores = self._apply_mask(scores, T, device, mask=mask)

        out = self._attention(scores, V, B, T)
        # out.shape == (B, T, D)
        out = self.out_linear(out)
        # out.shape == (B, T, D)

        return out

    @property
    def d_head(self):
        return self._d_head

    @property
    def use_RoPE(self):
        return self._use_RoPE

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def num_kv_heads(self):
        return self._num_kv_heads

    @property
    def dropout_p(self):
        return self.dropout.p

    @property
    def embed_dim(self):
        if not self.use_gqa:
            return self.qkv_linear.in_features
        return self.kv_linear.in_features

    @property
    def RoPE_base(self):
        if self.use_RoPE:
            return self.RoPE.base
        else:
            return None

    @property
    def num_q_heads(self):
        return self._num_q_heads

    @property
    def use_gqa(self):
        return self._use_gqa
