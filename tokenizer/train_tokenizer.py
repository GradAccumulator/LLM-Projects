import sentencepiece as spm
from pathlib import Path
from utils.dev_utils import num_to_str

DATASET_NAME = 'fineweb2_korean'
INPUT_PATH = Path(__file__).parent.parent/'datasets'/'llm'/DATASET_NAME/'processed'/'train.txt'

vocab_size = 32000
model_name = f"{num_to_str(vocab_size).lower()}_fineweb2_korean"
spm.SentencePieceTrainer.train(
    input=INPUT_PATH,
    model_prefix=f"tokenizer/{model_name}/{model_name}",
    vocab_size=vocab_size,
    model_type="bpe",
    input_sentence_size=20_000_000,
    shuffle_input_sentence=True,

    pad_id=0,
    unk_id=2,
    bos_id=-1,
    eos_id=-1,

    user_defined_symbols=["<eos>"],
)