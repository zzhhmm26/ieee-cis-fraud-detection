import pandas as pd

transaction_ids = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=["TransactionID"]
)

train_identity = pd.read_csv(
    "data/raw/train_identity.csv"
)

matched = transaction_ids["TransactionID"].isin(
    train_identity["TransactionID"]
)

print("身份表形状：", train_identity.shape)
print("重复 TransactionID：",train_identity["TransactionID"].duplicated().sum())
print("能匹配身份信息的交易数：", matched.sum())
print("身份信息覆盖率：", matched.mean())