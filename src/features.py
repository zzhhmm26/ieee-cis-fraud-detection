"""V5 的确定性特征和仅在训练期拟合的频率编码。"""

import numpy as np
import pandas as pd


HIGH_CARDINALITY_FEATURES = ["card1", "card2", "card5", "addr1", "addr2"]
C_FEATURES = [f"C{i}" for i in range(1, 15)]
D_FEATURES = [f"D{i}" for i in range(1, 16)]
CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
]
BASE_NUMERIC_FEATURES = [
    "TransactionAmt", "Transaction_Amt_Log", "Transaction_Hour", "Transaction_Day",
]


def add_deterministic_features(data: pd.DataFrame) -> pd.DataFrame:
    """生成不依赖样本分布的金额和时间特征。"""
    result = data.copy()
    result["Transaction_Hour"] = (result["TransactionDT"] // 3600) % 24
    result["Transaction_Day"] = result["TransactionDT"] // 86400
    result["Transaction_Amt_Log"] = np.log1p(result["TransactionAmt"])
    return result


def add_train_only_frequency_features(
    train_part: pd.DataFrame, valid_part: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, float]]:
    """用训练期频率映射编码高基数类别；验证期未见类别编码为 0。"""
    train_result = train_part.copy()
    valid_result = valid_part.copy()
    frequency_features: list[str] = []
    unseen_rates: dict[str, float] = {}

    for column in HIGH_CARDINALITY_FEATURES:
        train_values = train_result[column].fillna("missing")
        valid_values = valid_result[column].fillna("missing")
        frequency_map = train_values.value_counts(normalize=True)
        feature_name = f"{column}_frequency"

        train_result[feature_name] = train_values.map(frequency_map)
        valid_result[feature_name] = valid_values.map(frequency_map).fillna(0)
        frequency_features.append(feature_name)
        unseen_rates[column] = float((valid_result[feature_name] == 0).mean())

    return train_result, valid_result, frequency_features, unseen_rates


def add_d_missing_indicators(
    train_part: pd.DataFrame, valid_part: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """保留 D 字段缺失本身可能携带的信息。"""
    train_result = train_part.copy()
    valid_result = valid_part.copy()
    missing_features: list[str] = []

    for column in D_FEATURES:
        feature_name = f"{column}_missing"
        train_result[feature_name] = train_result[column].isna().astype("int8")
        valid_result[feature_name] = valid_result[column].isna().astype("int8")
        missing_features.append(feature_name)

    return train_result, valid_result, missing_features


def build_v5_feature_frames(
    train_part: pd.DataFrame, valid_part: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, float]]:
    """构造 V5 的训练/验证特征矩阵与特征名；不接触标签。"""
    train_part, valid_part, frequency_features, unseen_rates = (
        add_train_only_frequency_features(train_part, valid_part)
    )
    train_part, valid_part, missing_features = add_d_missing_indicators(
        train_part, valid_part
    )
    numeric_features = (
        BASE_NUMERIC_FEATURES + frequency_features + C_FEATURES + D_FEATURES + missing_features
    )
    feature_columns = numeric_features + CATEGORICAL_FEATURES
    return (
        train_part[feature_columns],
        valid_part[feature_columns],
        numeric_features,
        CATEGORICAL_FEATURES,
        unseen_rates,
    )
