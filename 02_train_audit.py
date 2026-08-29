import pandas as pd

train_transaction = pd.read_csv("data/raw/train_transaction.csv")

print("训练集形状：", train_transaction.shape)
print("重复 TransactionID：",train_transaction["TransactionID"].duplicated().sum())
print("欺诈统计：",train_transaction["isFraud"].value_counts())
print("欺诈率:",train_transaction["isFraud"].mean())
print(train_transaction.isna().mean().sort_values(ascending=False).head(10))
