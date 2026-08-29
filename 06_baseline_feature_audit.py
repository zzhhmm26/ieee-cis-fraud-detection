import pandas as pd

columns = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "isFraud",
]

data = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=columns,
    low_memory=False
)

print(data.dtypes)
print("\n唯一值数量：")
print(data.nunique().sort_values())

print("\n缺失率：")
print(data.isna().mean().sort_values(ascending=False))
