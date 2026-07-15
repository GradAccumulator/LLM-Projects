import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input=r"C:\Users\LEEJUHYOUNG\Documents\LLM Projects\datasets\llm\fineweb2_korean\processed\train.txt",
    model_prefix="tokenizer/32k_fineweb2_korean",
    vocab_size=32000,
    model_type="bpe",
    input_sentence_size=10_000_000,
    shuffle_input_sentence=True,

    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,

    pad_piece="<pad>",
    bos_piece="<bos>",
    eos_piece="<eos>",
    unk_piece="<unk>",
)