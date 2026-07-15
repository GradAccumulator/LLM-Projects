import pandas as pd
from multiprocessing import Pool
import shutil
from pathlib import Path

BASE_DIR = Path(
    r'C:\Users\LEEJUHYOUNG\Documents\LLM Projects'
    r'\datasets\llm\fineweb2_korean'
)

DATASET_TYPE = 'test'
TRAIN_FILE_GROUPS = [(0,1,2), (3,4)]
TEST_FILE_GROUPS = [(0,)]

def process_file(args:tuple[Path, Path]):
    input_path,output_path = args

    texts = pd.read_parquet(input_path, columns=['text'])['text']

    print(f"[추출] {input_path.name}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text)
            f.write("<eos>")

def merge_files(output_paths:list[Path], destination_path:Path):
    with destination_path.open("a", encoding="utf-8") as destination:
        for output_path in output_paths:
            with output_path.open("r", encoding="utf-8") as source:
                shutil.copyfileobj(source, destination)

def main():
    file_groups = (
        TRAIN_FILE_GROUPS
        if DATASET_TYPE == "train"
        else TEST_FILE_GROUPS
    )

    raw_dir = BASE_DIR / "raw" / "data" / "kor_Hang" / DATASET_TYPE
    processed_dir = BASE_DIR / "processed" / DATASET_TYPE
    save_path = BASE_DIR / "processed" / f"{DATASET_TYPE}.txt"

    processed_dir.mkdir(parents=True, exist_ok=True)

    save_path.write_text("", encoding="utf-8")

    for i in range(1):
        for file_group in file_groups:
            file_names = [f"00{i}_0000{j}" for j in file_group]

            tasks = [
                (
                    raw_dir / f"{file_name}.parquet",
                    processed_dir / f"{file_name}.txt",
                )
                for file_name in file_names
            ]

            print(f"[처리 시작] i={i}, files={file_names}")

            with Pool(processes=len(tasks)) as pool:
                pool.map(process_file, tasks)

            output_paths = [output_path for _,output_path in tasks]

            print("[저장] 추출된 데이터셋 저장")
            merge_files(output_paths, save_path)

            for output_path in output_paths:
                output_path.unlink()


if __name__ == "__main__":
    main()