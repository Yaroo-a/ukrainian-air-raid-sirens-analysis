"""Аналіз зсуву розподілу ознак (Feature Distribution Shift) між Train, Val, Test та до/після 2025-08-01.

Обчислює середні значення ключових ознак із data/processed/ml_features_hourly.parquet:
1. У розрізі хронологічних сплітів: train, val, test -> reports/feature_shift_summary.csv
2. Усередині test до та після 2025-08-01 00:00:00 UTC -> reports/feature_shift_test_subperiods.csv
"""

from pathlib import Path
import pandas as pd


FEATURES_TO_CHECK = [
    "national_active_count",
    "neighbors_active_count",
    "hours_since_last_alert",
    "adaptive_rate_7d",
    "adaptive_rate_14d",
    "adaptive_rate_30d",
    "onsets_last_24h",
    "burden_last_6h",
    "burden_last_24h",
    "target_onset_next_1h",
]


def analyze_feature_shift(
    parquet_path: str | Path = "data/processed/ml_features_hourly.parquet",
    out_splits_csv: str | Path = "reports/feature_shift_summary.csv",
    out_subperiods_csv: str | Path = "reports/feature_shift_test_subperiods.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(parquet_path)

    # 1. Середні значення за сплітами Train / Val / Test
    split_order = ["train", "val", "test"]
    split_means = (
        df.groupby("split_set")[FEATURES_TO_CHECK]
        .mean()
        .reindex(split_order)
    )
    split_counts = df.groupby("split_set").size().reindex(split_order)
    split_summary = split_means.T.copy()
    split_summary.columns = ["train_mean", "val_mean", "test_mean"]
    split_summary["test_vs_train_ratio"] = split_summary["test_mean"] / split_summary["train_mean"]
    split_summary.index.name = "feature"

    Path(out_splits_csv).parent.mkdir(parents=True, exist_ok=True)
    split_summary.reset_index().to_csv(out_splits_csv, index=False)

    # 2. Усередині Test: до 2025-08-01 та з 2025-08-01
    test_df = df[df["split_set"] == "test"].copy()
    cutoff = pd.Timestamp("2025-08-01 00:00:00", tz="UTC")
    test_df["test_subperiod"] = test_df["timestamp_utc"].apply(
        lambda t: "test_pre_2025_08_01" if t < cutoff else "test_post_2025_08_01"
    )
    sub_order = ["test_pre_2025_08_01", "test_post_2025_08_01"]
    sub_means = (
        test_df.groupby("test_subperiod")[FEATURES_TO_CHECK]
        .mean()
        .reindex(sub_order)
    )
    sub_summary = sub_means.T.copy()
    sub_summary.columns = ["test_pre_2025_08_01_mean", "test_post_2025_08_01_mean"]
    sub_summary["post_vs_pre_ratio"] = (
        sub_summary["test_post_2025_08_01_mean"] / sub_summary["test_pre_2025_08_01_mean"]
    )
    sub_summary.index.name = "feature"
    sub_summary.reset_index().to_csv(out_subperiods_csv, index=False)

    print("=" * 85)
    print("📊 1. СЕРЕДНІ ЗНАЧЕННЯ ОЗНАК ЗА СПЛІТАМИ (Train / Val / Test):")
    print(f"   Кількість рядків: Train={split_counts['train']:,}, Val={split_counts['val']:,}, Test={split_counts['test']:,}")
    print("=" * 85)
    print(split_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    pre_cnt = int((test_df["timestamp_utc"] < cutoff).sum())
    post_cnt = int((test_df["timestamp_utc"] >= cutoff).sum())
    print("\n" + "=" * 85)
    print("📉 2. ЗСУВ УСЕРЕДИНІ TEST ДО І ПІСЛЯ 2025-08-01 (перехід на агрегацію районів):")
    print(f"   Кількість рядків: до 2025-08-01 = {pre_cnt:,}, з 2025-08-01 = {post_cnt:,}")
    print("=" * 85)
    print(sub_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    return split_summary, sub_summary


if __name__ == "__main__":
    analyze_feature_shift()
