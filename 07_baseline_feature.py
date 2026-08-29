import numpy as np
import pandas as pd

data = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=[
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "isFraud",
    ],
    low_memory=False
)

data['Transaction_Hour'] = (data['TransactionDT'] // 3600) % 24
data['Transaction_Day'] = data['TransactionDT'] // 86400
data['Transaction_Amt_Log'] = np.log1p(data['TransactionAmt'])

print(data[[
    "TransactionDT",
    "Transaction_Hour",
    "Transaction_Day",
    "TransactionAmt",
    "Transaction_Amt_Log",
    "isFraud",
]].head())

print("\n新字段类型：")
print(data.dtypes)