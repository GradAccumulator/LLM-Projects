import sentencepiece as spm

model_name = "32k_fineweb2_korean"
spm.SentencePieceTrainer.train(
    input=r"C:\Users\LEEJUHYOUNG\Documents\LLM Projects\datasets\llm\fineweb2_korean\processed\train.txt",
    model_prefix=f"tokenizer/{model_name}/{model_name}",
    vocab_size=32000,
    model_type="bpe",
    input_sentence_size=20_000_000,
    shuffle_input_sentence=True,

    pad_id=0,
    unk_id=2,
    bos_id=-1,
    eos_id=-1,

    user_defined_symbols=["<eos>"],
)