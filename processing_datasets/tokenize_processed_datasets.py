from tokenizer import Tokenizer
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

B = 1_000_000_000
M = 1_000_000
K = 1_000
CHUNK_SIZE = 500*M

BASE_DIR = Path(
    r'C:\Users\LEEJUHYOUNG\Documents\LLM Projects'
    r'\datasets\llm\fineweb2_korean'
)
DATASET_TYPE = 'train'
SAVE_DIR = BASE_DIR/"tokenized"/DATASET_TYPE
DATASET_PATH = BASE_DIR/"processed"/(DATASET_TYPE+".txt")

tokenizer = Tokenizer(model_name="32k_fineweb2_korean")

def num_to_str(num):
    if num>=B:
        return f"{num/B:.2f}B"
    elif num>=M:
        return f"{num/M:.2f}M"
    elif num>=K:
        return f"{num/K:.2f}K"
    return f"{num}"

def save_one_chunk(file_idx:int, chunk:NDArray[np.uint16]):
    file_name = f"{file_idx:04d}.bin"
    chunk.tofile(SAVE_DIR/file_name)
    processed_tokens = (file_idx+1)*CHUNK_SIZE
    print(f"[처리 완료] {file_idx+1}번째 청크 처리 완료, 현재까지 {num_to_str(processed_tokens)} 토큰 처리 완료")

def main() -> None:
    file_idx = 0
    write_pos = 0
    progress_segments = 5
    next_progress_segment = 1
    token_buffer = np.empty(CHUNK_SIZE, dtype=np.uint16)
    print("[시작] 토크나이징 시작")
    with DATASET_PATH.open("r", encoding='utf-8') as f:
        print(f"[처리 시작] 1번째 청크({num_to_str(CHUNK_SIZE)} 토큰) 처리 시작")
        for line in f:
            line_token_ids:NDArray[np.uint16] = tokenizer.encode(line, return_type=np.uint16)

            while len(line_token_ids)>=(remaining_capacity:=CHUNK_SIZE - write_pos):
                token_buffer[write_pos:] = line_token_ids[:remaining_capacity]

                save_one_chunk(file_idx, token_buffer)
                file_idx += 1
                print(f"[처리 시작] {file_idx+1}번째 청크({num_to_str(CHUNK_SIZE)} 토큰) 처리 시작")

                line_token_ids = line_token_ids[remaining_capacity:]
                write_pos = 0
                next_progress_segment = 1
            
            token_buffer[write_pos:write_pos+len(line_token_ids)] = line_token_ids
            write_pos += len(line_token_ids)
            if next_progress_segment != 0 and write_pos>=CHUNK_SIZE*(next_progress_segment/progress_segments):
                print(f"[처리중] 현재까지 {num_to_str(CHUNK_SIZE*file_idx + write_pos)} 토큰 처리 완료")
                next_progress_segment+=1

    if write_pos  > 0:
        save_one_chunk(file_idx, token_buffer[:write_pos])

if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    main()