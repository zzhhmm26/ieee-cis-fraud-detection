from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

file_names = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]

for file_name in file_names:
    file_path = DATA_DIR / file_name
    sample = pd.read_csv(file_path, nrows=5)

    print(f"\n文件：{file_name}")
    print("形状：", sample.shape)
    print("字段：", sample.columns.tolist())
    print(sample.head(2))
