import sentencepiece as spm
import numpy as np
import os

class Tokenizer:
    def __init__(self, model_name:str, num_threads:int=os.cpu_count()-1):
        self._processor = spm.SentencePieceProcessor(
            model_file=fr"tokenizer\{model_name}\{model_name}.model"
        )
        self.num_threads = num_threads
    
    def encode(self, x, return_type=None):
        if issubclass(return_type, np.generic):
            return np.asarray(self._processor.encode(x, num_threads=self.num_threads), dtype=return_type)
        return self._processor.encode(x, return_type=return_type, num_threads=self.num_threads)

    def decode(self, x):
        return self._processor.decode(x, num_threads=self.num_threads)

    @property
    def vocab_size(self) -> int:
        return self._processor.vocab_size()
    @property
    def eos_id(self) -> int:
        return self._processor.piece_to_id('<eos>')
    @property
    def pad_id(self) -> int:
        return self._processor.pad_id()
    @property
    def unk_id(self) -> int:
        return self._processor.unk_id()