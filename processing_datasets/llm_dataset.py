import torch
import torch.utils.data as data
from pathlib import Path

from utils import dev_utils


class LLMDataset(data.Dataset):
    def __init__(
        self,
        seq_len: int,
        datasets_dir: str | Path,
        dataset_name: str,
        total_tokens: int = 0,
        dataset_type: str = "train",
        bin_dtype: torch.dtype = torch.uint16,
    ):
        func_name = "LLMDataset.__init__()"
        dev_utils.type_check(
            ("seq_len", seq_len, int),
            ("total_tokens", total_tokens, int),
            ("datasets_dir", datasets_dir, str | Path),
            ("dataset_name", dataset_name, str),
            ("dataset_type", dataset_type, str),
            ("bin_dtype", bin_dtype, torch.dtype),
            func_name=func_name,
        )
        self._total_tokens = total_tokens
        self._seq_len = seq_len
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self._bin_dtype = bin_dtype
        if not isinstance(datasets_dir, Path):
            datasets_dir = Path(datasets_dir)

        self._dataset_path = (
            datasets_dir / "llm" / dataset_name / "tokenized" / (dataset_type + ".bin")
        )
        element_size = torch.empty((), dtype=bin_dtype).element_size()
        self._num_tokens = self.dataset_path.stat().st_size // element_size
        self._tokens = torch.from_file(
            str(self.dataset_path),
            shared=False,
            size=self._num_tokens,
            dtype=self._bin_dtype,
        )

        if self.total_tokens == 0:
            self._total_tokens = self.num_tokens

    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len + 1
        dataset = self._tokens[start:end]
        x, y = dataset[:-1], dataset[1:]
        return x, y

    def __len__(self):
        return (self.total_tokens - 1) // self.seq_len

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def name(self):
        return self._dataset_name

    @property
    def type(self):
        return self._dataset_type

    @property
    def bin_dtype(self):
        return self._bin_dtype

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def total_tokens(self):
        return self._total_tokens
