"""读取 V5 所需交易字段，并按时间进行训练/验证切分。"""

from pathlib import Path

import pandas as pd


TRANSACTION_COLUMNS = [
    "TransactionDT", "TransactionAmt", "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain", "isFraud", "card1", "card2",
    "card5", "addr1", "addr2",
] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]


def load_v5_transaction_data(path: str | Path) -> pd.DataFrame:
    """只读取 V5 使用的原始字段；标签 isFraud 仅用于训练和评价。"""
    return pd.read_csv(path, usecols=TRANSACTION_COLUMNS, low_memory=False)


def chronological_split(
    data: pd.DataFrame, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """按 TransactionDT 前/后顺序切分，返回训练集、验证集和切分点。"""
    cutoff = data["TransactionDT"].quantile(1 - validation_fraction)
    train_part = data.loc[data["TransactionDT"] <= cutoff].copy()
    valid_part = data.loc[data["TransactionDT"] > cutoff].copy()

    if train_part.empty or valid_part.empty:
        raise ValueError("时间切分产生空数据集，请检查 TransactionDT。")
    return train_part, valid_part, cutoff
