import torch
from torch import Tensor, nn


class Cache:
    def __init__(self, cache: Tensor, use_gqa: bool):
        self.cache = cache
        self.use_gqa = use_gqa

    def __getitem__(self, keys):
        if isinstance(keys, tuple) and len(keys) == 2 and keys[0] == Ellipsis:
            if self.use_gqa:
                return self.cache[:, :, :, keys[1], :]
            else:
                return self.cache[:, :, keys[1], :]
        return self.cache[keys]

    def __setitem__(self, keys, value):
        if isinstance(keys, tuple) and len(keys) == 2 and keys[0] == Ellipsis:
            if self.use_gqa:
                self.cache[:, :, :, keys[1], :] = value
            else:
                self.cache[:, :, keys[1], :] = value
        else:
            self.cache[keys] = value


class KVCache(nn.Module):
    def __init__(self, num_layers: int, max_seq_len: int, d_head: int, use_gqa: bool, num_kv_heads: int):
        super().__init__()
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.d_head = d_head
        self.use_gqa = use_gqa
        self.num_kv_heads = num_kv_heads

        self.empty_cache()

    def empty_cache(self):
        self.cache = torch.empty(0, device="meta")

    def allocate(self, device, dtype):
        cache_size = (
            (self.num_layers, 1, self.num_kv_heads, 1, self.max_seq_len, self.d_head)
            if self.use_gqa
            else (self.num_layers, 1, self.num_kv_heads, self.max_seq_len, self.d_head)
        )
        del self.cache
        self.register_buffer("cache", torch.empty(*cache_size, device=device, dtype=dtype), persistent=False)

    def expand(self, max_seq_len: int, start_idx: int):
        cache = self.cache

        self.max_seq_len = max_seq_len
        self.allocate_kv_cache()
        self[..., :start_idx] = cache

    def __getitem__(self, keys):
        if  isinstance(keys, tuple):
            if len(keys) == 2 and keys[0] == Ellipsis:
                if self.use_gqa:
                    return self.cache[:, :, :, :, keys[1], :]
                else:
                    return self.cache[:, :, :, keys[1], :]
        elif self.is_empty():
            return self.cache
        else:
            return Cache(self.cache[keys], use_gqa=self.use_gqa)
        return self.cache[keys]

    def __setitem__(self, keys, value):
        if len(keys) == 2 and keys[0] == Ellipsis:
            if self.use_gqa:
                self.cache[:, :, :, :, keys[1], :] = value
            else:
                self.cache[:, :, :, keys[1], :] = value
        elif len(keys) == 1 and self.is_empty():
            raise ValueError(f"<KVCache.__setitem__()> kv cache가 비어있습니다.")
        else:
            self.cache[keys] = value

    def is_empty(self):
        return self.cache.device.type == "meta"
