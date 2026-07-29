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
        num_heads: int,
        dropout: float | int,
        init_cfg: DictConfig | dict = None,
        bias: bool = False,
        use_RoPE: bool = True,
        RoPE_base: int | float = None,
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
            ("num_heads", num_heads, int),
            ("bias", bias, bool),
            ("use_RoPE", use_RoPE, bool),
            ("dropout", dropout, float | int),
            ("RoPE_base", RoPE_base, float | int | None),
            ("init_cfg", init_cfg, DictConfig | dict | None),
            func_name=func_name,
        )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"<{func_name}> embed_dim은 num_heads와 나누어 떨어져야 합니다."
            )
        if not 0 <= dropout < 1:
            raise ValueError(
                f"<{func_name}> dropout p는 반드시 [0, 1) 범위의 실수여야 합니다."
            )

        self._num_heads = num_heads
        self._d_head = embed_dim // num_heads

        init_cfg = dev_utils.make_dictconfig(
            init_cfg, default={"qkv_linear": None, "out_linear": None}
        )
        dev_utils.check_dictconfig(
            init_cfg,
            ("qkv_linear", "out_linear"),
            func_name=func_name,
        )

        self.qkv_linear = Linear(
            embed_dim, embed_dim * 3, init_cfg=init_cfg.qkv_linear, use_bias=bias
        )
        self.out_linear = Linear(
            embed_dim, embed_dim, init_cfg=init_cfg.out_linear, use_bias=bias
        )
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
        QKV: Tensor = self.qkv_linear(x)
        # QKV.shape == (B,T,D*3)
        Q, K, V = QKV.chunk(3, dim=-1)
        # Q,K,V shape == (B,T,D)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        # Q, K, V shape == (B, H, T, D)

        return B, Q, K, V

    @staticmethod
    def make_causal_mask(T: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _apply_mask(
        self, scores: Tensor, T: int, device: torch.device, mask: Tensor = None
    ) -> Tensor:
        # scores.shape == (B,H,T,T)
        need_new_mask = (mask is None) and not (
            hasattr(self, "cached_mask") and self.cached_mask.shape == (T, T)
        )
        if need_new_mask:
            self.register_buffer(
                "cached_mask", self.make_causal_mask(T, device), persistent=False
            )
        scores = scores.masked_fill(
            mask if mask is not None else self.cached_mask, -float("inf")
        )

        return scores

    def _attention(self, scores: Tensor, V: Tensor, B: int, T: int) -> Tensor:
        # scores.shape == (B,H,T,T), V.shape == (B,H,T,D)
        weights = self.softmax(scores)
        drop = self.dropout(weights)
        out = matmul(drop, V)
        # out.shape == (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        # out.shape == (B, T, D)

        return out

    def forward_debug(self, x: Tensor, mask: Tensor, T: int) -> Tensor:
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

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        cached_sin: Tensor = None,
        cached_cos: Tensor = None,
    ) -> Tensor:
        # x.shape == (B, T, D)
        device = x.device
        T = x.size(1)
        if rt.DEBUG_CHECKS:
            self.forward_debug(x, mask, T)

        B, Q, K, V = self._qkv_projection(x)
        # Q,K,V shape == (B, H, T, D)

        if self.use_RoPE:
            Q, K = self.RoPE(Q, K, cached_sin=cached_sin, cached_cos=cached_cos)

        scores = matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head)
        # scores.shape == (B,H,T,T)

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
    def num_heads(self):
        return self._num_heads

    @property
    def dropout_p(self):
        return self.dropout.p

    @property
    def embed_dim(self):
        return self.qkv_linear.in_features

    @property
    def RoPE_base(self):
        if self.use_RoPE:
            return self.RoPE.base
        else:
            return None
