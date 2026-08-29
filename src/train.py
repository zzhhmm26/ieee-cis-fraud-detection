"""V5 主模型的可复现训练入口。

从仓库根目录运行：python -m src.train
"""

import json
from pathlib import Path

from src.data_loader import chronological_split, load_v5_transaction_data
from src.evaluate import evaluate_binary_classifier
from src.features import add_deterministic_features, build_v5_feature_frames
from src.preprocessing import build_v5_pipeline


def main() -> None:
    data = load_v5_transaction_data("data/raw/train_transaction.csv")
    data = add_deterministic_features(data)
    train_part, valid_part, cutoff = chronological_split(data)

    x_train, x_valid, numeric_features, categorical_features, unseen_rates = (
        build_v5_feature_frames(train_part, valid_part)
    )
    y_train = train_part["isFraud"]
    y_valid = valid_part["isFraud"]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    pipeline = build_v5_pipeline(
        numeric_features, categorical_features, scale_pos_weight
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_valid)[:, 1]
    metrics, threshold_table, review_table = evaluate_binary_classifier(
        y_valid, probabilities
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    run_record = {
        "experiment": "xgb_v5_C_D_414_late_engineered",
        "validation_scheme": "chronological_80_20",
        "cutoff_transaction_dt": cutoff,
        "n_train": len(train_part),
        "n_valid": len(valid_part),
        "scale_pos_weight": scale_pos_weight,
        "unseen_category_rates": unseen_rates,
        "metrics": metrics,
    }
    (output_dir / "v5_engineered_metrics.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    threshold_table.to_csv(output_dir / "v5_engineered_thresholds.csv", index=False)
    review_table.to_csv(output_dir / "v5_engineered_review_capacity.csv", index=False)

    print("V5 工程化训练完成")
    print(f"时间切分点：{cutoff}")
    print(f"scale_pos_weight：{scale_pos_weight}")
    print("验证集未见类别比例：", unseen_rates)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(review_table.to_string(index=False))


if __name__ == "__main__":
    main()
