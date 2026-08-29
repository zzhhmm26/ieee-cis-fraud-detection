import pandas as pd

train_transaction = pd.read_csv(
    "data/raw/train_transaction.csv",
    low_memory=False
)

train_identity = pd.read_csv(
    "data/raw/train_identity.csv",
    low_memory=False
)

train_identity.columns = train_identity.columns.str.replace("-", "_")

merged_train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left",validate="one_to_one",indicator=True)

print("交易表形状：", train_transaction.shape)
print("身份表形状：", train_identity.shape)
print("合并后形状：", merged_train.shape)
print("合并后重复 TransactionID：", merged_train["TransactionID"].duplicated().sum())


print(merged_train.groupby('_merge')['isFraud'].agg(['count', 'sum','mean']))