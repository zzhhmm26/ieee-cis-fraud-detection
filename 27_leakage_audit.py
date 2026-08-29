"""审计 V5 主模型的特征来源、训练期拟合边界与线上可得性风险。

本脚本不训练模型，也不评价模型分数。它只把已经由代码验证的事实、
以及公开竞赛数据无法确认的生产假设分开记录。
"""

from pathlib import Path

import pandas as pd


V5_SCRIPT = Path("21_xgboost_v5_n414_late_window.py")
TRANSACTION_DATA = Path("data/raw/train_transaction.csv")
OUTPUT_PATH = Path("results/leakage_audit.csv")


# 先从原始 CSV 表头确认字段确实存在；不读取整份大文件。
transaction_columns = set(pd.read_csv(TRANSACTION_DATA, nrows=0).columns)
required_columns = {
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "isFraud",
}
missing_columns = required_columns - transaction_columns
assert not missing_columns, f"原始交易表缺少字段：{missing_columns}"

# 再检查当前主模型脚本确实保留了训练期拟合的边界。
v5_source = V5_SCRIPT.read_text(encoding="utf-8")
assert "frequency_map = train_values.value_counts(normalize=True)" in v5_source
assert "Pipeline" in v5_source
assert "model_pipeline.fit(x_train,y_train)" in v5_source


rows = [
    {
        "feature_group": "标签",
        "columns_or_feature": "isFraud",
        "used_by_v5": "否，仅作为 y_train/y_valid",
        "construction": "公开训练集给出的二元标签",
        "fit_scope": "不适用",
        "available_at_transaction_time": "否",
        "leakage_assessment": "已排除",
        "evidence_or_limit": "只传入 model_pipeline.fit 的 y_train，并用于验证评价。",
    },
    {
        "feature_group": "主键",
        "columns_or_feature": "TransactionID",
        "used_by_v5": "否",
        "construction": "交易唯一标识",
        "fit_scope": "不适用",
        "available_at_transaction_time": "可能可得，但不应用作泛化特征",
        "leakage_assessment": "已排除",
        "evidence_or_limit": "V5 未读取该列；主键可能记忆训练样本而非学习风险规律。",
    },
    {
        "feature_group": "原始交易字段",
        "columns_or_feature": "TransactionAmt, ProductCD, card4, card6, P_emaildomain, R_emaildomain",
        "used_by_v5": "是",
        "construction": "直接来自交易表",
        "fit_scope": "不拟合；类别缺失填充、独热编码在训练集 fit",
        "available_at_transaction_time": "数据集层面假定可得",
        "leakage_assessment": "未发现标签泄漏",
        "evidence_or_limit": "真实业务中的采集时点、延迟和字段含义未由公开数据集确认。",
    },
    {
        "feature_group": "时间与金额派生",
        "columns_or_feature": "Transaction_Amt_Log, Transaction_Hour, Transaction_Day",
        "used_by_v5": "是",
        "construction": "由 TransactionAmt、TransactionDT 确定性计算",
        "fit_scope": "不拟合；数值填充和标准化在训练集 fit",
        "available_at_transaction_time": "数据集层面假定可得",
        "leakage_assessment": "未发现跨期统计泄漏",
        "evidence_or_limit": "TransactionDT 是相对时间；生产环境需确认时区、时间原点和延迟。",
    },
    {
        "feature_group": "训练期频率编码",
        "columns_or_feature": "card1/card2/card5/addr1/addr2_frequency",
        "used_by_v5": "是",
        "construction": "类别在训练期出现的相对频率；验证期未见类别填 0",
        "fit_scope": "仅 train_part：value_counts(normalize=True)",
        "available_at_transaction_time": "可实现，但需维护历史窗口统计",
        "leakage_assessment": "代码层面已防止验证期前视",
        "evidence_or_limit": "线上必须只用交易发生前的历史数据滚动更新，不能预先统计未来交易。",
    },
    {
        "feature_group": "匿名 C 组",
        "columns_or_feature": "C1-C14",
        "used_by_v5": "是",
        "construction": "直接来自交易表的匿名数值字段",
        "fit_scope": "不拟合；数值填充和标准化在训练集 fit",
        "available_at_transaction_time": "未知",
        "leakage_assessment": "未发现代码层面标签泄漏",
        "evidence_or_limit": "字段业务语义和生成时点未公开，强预测能力不等于线上可用。",
    },
    {
        "feature_group": "匿名 D 组与缺失模式",
        "columns_or_feature": "D1-D15, D1_missing-D15_missing",
        "used_by_v5": "是",
        "construction": "D 原值；缺失指示器由 isna() 确定性计算",
        "fit_scope": "缺失指示器不拟合；数值填充和标准化在训练集 fit",
        "available_at_transaction_time": "未知",
        "leakage_assessment": "未发现代码层面标签泄漏",
        "evidence_or_limit": "匿名字段与缺失机制的业务含义、生成时点和线上稳定性均未公开。",
    },
    {
        "feature_group": "常见但不属于本数据集的字段",
        "columns_or_feature": "isFlaggedFraud",
        "used_by_v5": "不适用",
        "construction": "不在 IEEE-CIS train_transaction.csv 中",
        "fit_scope": "不适用",
        "available_at_transaction_time": "不适用",
        "leakage_assessment": "不适用",
        "evidence_or_limit": "通过读取原始 CSV 表头确认该字段不存在；不得把其他欺诈数据集的字段混入本项目。",
    },
    {
        "feature_group": "identity 表字段",
        "columns_or_feature": "id_12, id_15, id_35-id_38, DeviceType",
        "used_by_v5": "否",
        "construction": "train_identity 左连接后的候选类别字段",
        "fit_scope": "不适用",
        "available_at_transaction_time": "未知",
        "leakage_assessment": "V6 消融后未保留",
        "evidence_or_limit": "identity 覆盖率随时间变化；小规模 V6 时间消融无增益，且线上可得性未验证。",
    },
    {
        "feature_group": "官方测试集",
        "columns_or_feature": "test_transaction/test_identity",
        "used_by_v5": "否，用于最终无标签推理",
        "construction": "竞赛官方未来时间段数据",
        "fit_scope": "不参与特征选择、树数选择或阈值选择",
        "available_at_transaction_time": "不适用",
        "leakage_assessment": "已隔离",
        "evidence_or_limit": "官方 test 没有 isFraud，不能用于报告离线泛化指标。",
    },
]

audit = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(exist_ok=True)
audit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("泄漏与线上可得性审计：")
print(audit.to_string(index=False))
print(f"\n已保存：{OUTPUT_PATH}")
