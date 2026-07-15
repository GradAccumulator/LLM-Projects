from tokenizer import Tokenizer
import hydra

B = 1_000_000_000
M = 1_000_000
CHUNK_SIZE = 500*M

tokenizer = Tokenizer(model_path="32k_fineweb2_korean")

def tokenize_one_chunk(max_seq_len:int):
    


@hydra.main(version_base=None, config_path="../configs", config_name="model1_v32k_s1024")
def main(cfg):
    max_seq_len:int = cfg.model.max_seq_len




if __name__ == "__main__":
    main()