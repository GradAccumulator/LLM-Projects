import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path:str):
        self._processor = spm.SentencePieceProcessor(
            model_file=fr"tokenizer\{model_path}"
        )
    
    def encode(self, x):
        return self._processor.encode(x)

    def decode(self, x):
        return self._processor.decode(x)

    @property
    def vocab_size(self) -> int:
        return self._processor.vocab_size()
    @property
    def eos_id(self) -> int:
        return self._processor.eos_id()
    @property
    def pad_id(self) -> int:
        return self._processor.pad_id()
    @property
    def bos_id(self) -> int:
        return self._processor.bos_id()
    @property
    def unk_id(self) -> int:
        return self._processor.unk_id()