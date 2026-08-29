import pandas as pd

data = pd.read_csv("data/raw/train_transaction.csv",
                   usecols = lambda column:(
                       column == 'isFraud' or column.startswith(('C', 'M', 'D'))
                   ),low_memory = False)

# 选出
for prefix in ['C','D','M']:
    columns = [column 
               for column in data.columns
               if column.startswith(prefix)
               ]

    print(f'\n{prefix}组字段：',columns)
    print("数据类型：")
    print(data[columns].dtypes.value_counts())
    print("缺失率：")
    print(data[columns].isna().mean().sort_values(ascending = False))
    print("唯一值数量：")
    print(data[columns].nunique().sort_values())