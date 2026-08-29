"""统一计算 V5 的分类、阈值和人工审核容量指标。"""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: pd.Series, probabilities, threshold: float = 0.5
) -> tuple[dict[str, float | list[list[int]]], pd.DataFrame, pd.DataFrame]:
    """返回总指标、阈值表和 Top-K 人工审核表。"""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision_at_0_5": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }

    threshold_rows = []
    for current_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        current_predictions = (probabilities >= current_threshold).astype(int)
        threshold_rows.append({
            "threshold": current_threshold,
            "alerts": int(current_predictions.sum()),
            "alert_rate": float(current_predictions.mean()),
            "precision": float(precision_score(y_true, current_predictions, zero_division=0)),
            "recall": float(recall_score(y_true, current_predictions, zero_division=0)),
            "f1": float(f1_score(y_true, current_predictions, zero_division=0)),
        })

    ranking = pd.DataFrame({"isFraud": y_true.to_numpy(), "probability": probabilities})
    ranking = ranking.sort_values("probability", ascending=False)
    review_rows = []
    total_fraud = ranking["isFraud"].sum()
    for top_rate in [0.01, 0.05, 0.10]:
        alert_count = int(len(ranking) * top_rate)
        top_alerts = ranking.head(alert_count)
        review_rows.append({
            "review_rate": top_rate,
            "alerts": alert_count,
            "captured_fraud": int(top_alerts["isFraud"].sum()),
            "fraud_capture_rate": float(top_alerts["isFraud"].sum() / total_fraud),
            "fraud_rate_in_alerts": float(top_alerts["isFraud"].mean()),
        })
    return metrics, pd.DataFrame(threshold_rows), pd.DataFrame(review_rows)
