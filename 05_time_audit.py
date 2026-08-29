import pandas as pd

data = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=["TransactionDT", "isFraud"]
)

cutoff = data["TransactionDT"].quantile(0.80)

train_part = data[data['TransactionDT'] <= cutoff]
valid_part = data[data['TransactionDT'] > cutoff]

print("最早时间：", data["TransactionDT"].min())
print("最晚时间：", data["TransactionDT"].max())
print("时间切分点：", cutoff)

print("\n训练部分：")
print("样本数：", len(train_part))
print("欺诈率：", train_part["isFraud"].mean())

print("\n验证部分：")
print("样本数：", len(valid_part))
print("欺诈率：", valid_part["isFraud"].mean())

test_time = pd.read_csv(
    "data/raw/test_transaction.csv",
    usecols=["TransactionDT"]
)

print("\n官方测试集时间：")
print("最早时间：", test_time["TransactionDT"].min())
print("最晚时间：", test_time["TransactionDT"].max())

