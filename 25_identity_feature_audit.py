import pandas as pd


transaction = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols = [
        'TransactionID',
        'TransactionDT',
        'isFraud',
        ],
        low_memory = False
    )

train_identity = pd.read_csv(
    'data/raw/train_identity.csv',
    low_memory = False
)
train_identity.columns = train_identity.columns.str.replace("-", "_")

merged = pd.merge(
    transaction,
    train_identity,
    on = 'TransactionID',
    how = 'left',
    validate = 'one_to_one', # 一条交易对应一个身份 否则报错（还可以是one_to_many,many_to_one,many_to_many)
    indicator = True # 生成'_merge'列，会有left_only,both等标识左右表信息是否为空
)

merged['has_identity'] = (merged['_merge'] == 'both') #左右表都有即为has_identity

# 切分时间窗口 前80%命名为'early_train',后20%命名为'late_valid'
cutoff = merged['TransactionDT'].quantile(0.8)

merged['time_window'] = 'late_valid'

merged.loc[
    merged['TransactionDT'] <= cutoff,
    'time_window'
] = 'early_train'

#每个时间窗口的 identity 覆盖率
print(
    merged.groupby("time_window")["has_identity"]
    .agg(["count", "mean"])
)

#按“时间窗口、是否有 identity”汇总样本数、欺诈数、欺诈率。
print(
    merged.groupby(["time_window", "has_identity"])["isFraud"]
    .agg(["count", "sum", "mean"])
)

#identity 字段质量表
identity_columns = [col for col in train_identity.columns if col != 'TransactionID']

matched_identity= merged.loc[
    merged['has_identity'],
    identity_columns
]

identity_quality = pd.DataFrame({
    'dtype':matched_identity.dtypes.astype(str),
    'nunique':matched_identity.nunique(),
    'missing_rate_all_transactions':merged[identity_columns].isna().mean(),
    'missing_rate_when_matched':matched_identity.isna().mean(),
}).sort_values(
    by = ["missing_rate_when_matched", "nunique"],
    ascending=[True, True],
    )

print("\nIdentity 字段质量表：")
print(identity_quality)