"""Data Ingestion & Cleaning Module (Phase 1).

Complies strictly with Quality Gate 1 specifications:
1. Filters level == 'oblast' for longitudinal consistency (2022-2026).
2. Converts timestamps to UTC and derives Europe/Kyiv local times.
3. Flags extreme occupied regions (Luhansk, Crimea, Sevastopol).
4. Filters out micro-alerts (< 60 seconds) as technical glitches.
5. Removes corrupted records (finished_at < started_at).
6. Assigns chronological 3-way split: Train, Validation, Test.
7. Saves output to Apache Parquet with optimized schema.
"""

from pathlib import Path
import pandas as pd
import numpy as np

OCCUPIED_EXTREME_REGIONS = [
    "Луганська область",
    "Автономна Республіка Крим",
    "м. Севастополь",
]


def clean_and_profile_data(
    input_csv_path: str | Path,
    output_parquet_path: str | Path,
    min_duration_seconds: int = 60,
) -> pd.DataFrame:
    """Performs full Phase 1 ingestion, cleaning, feature derivation and verification."""
    input_path = Path(input_csv_path)
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📖 1. Читання сирого CSV-файлу: {input_path}...")
    df = pd.read_csv(input_path)
    raw_count = len(df)
    print(f"   Всього сирих рядків: {raw_count:,}")

    # 1. Фільтрація за рівнем 'oblast'
    df = df[df["level"] == "oblast"].copy()
    oblast_count = len(df)
    print(f"   Рядків рівня 'oblast': {oblast_count:,} (відфільтровано {raw_count - oblast_count:,} районних/громадських)")

    # 2. Парсинг часових міток у UTC
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    # 3. Обробка незавершених тривог
    df["is_finished"] = df["finished_at"].notna()
    unfinished_count = (~df["is_finished"]).sum()
    print(f"   Незавершених тривог (без finished_at): {unfinished_count}")

    # 4. Розрахунок тривалості
    duration_s = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df["duration_seconds"] = duration_s
    df["duration_minutes"] = duration_s / 60.0
    df["duration_hours"] = duration_s / 3600.0

    # 5. Видалення записів із від'ємною тривалістю
    negative_duration_mask = df["is_finished"] & (df["duration_seconds"] < 0)
    corrupted_count = negative_duration_mask.sum()
    if corrupted_count > 0:
        print(f"   ⚠️ Видалено {corrupted_count} записів з від'ємною тривалістю (finished_at < started_at)")
        df = df[~negative_duration_mask].copy()

    # 6. Маркування ТОТ
    df["is_occupied_extreme"] = df["oblast"].isin(OCCUPIED_EXTREME_REGIONS)
    occupied_count = df["is_occupied_extreme"].sum()
    print(f"   Записів ТОТ (Луганщина/Крим/Севастополь): {occupied_count:,}")

    # 7. Фільтрація технічних мікро-глітчів (< 60 сек) для завершених тривог
    glitch_mask = df["is_finished"] & (df["duration_seconds"] < min_duration_seconds)
    glitch_count = glitch_mask.sum()
    print(f"   Видалено мікро-глітчів (< {min_duration_seconds}с): {glitch_count}")
    df = df[~glitch_mask].copy()

    # 8. Локальний час Europe/Kyiv
    kyiv_tz = "Europe/Kyiv"
    df["started_at_kyiv"] = df["started_at"].dt.tz_convert(kyiv_tz)
    df["finished_at_kyiv"] = df["finished_at"].dt.tz_convert(kyiv_tz)

    df["start_year"] = df["started_at_kyiv"].dt.year
    df["start_month"] = df["started_at_kyiv"].dt.month
    df["start_date"] = df["started_at_kyiv"].dt.date
    df["start_hour_kyiv"] = df["started_at_kyiv"].dt.hour
    df["start_day_of_week"] = df["started_at_kyiv"].dt.day_name()
    df["start_day_num"] = df["started_at_kyiv"].dt.dayofweek  # 0=Пн, 6=Нд
    df["is_night"] = df["start_hour_kyiv"].between(0, 5)  # 00:00 - 05:59

    # 9. Хронологічний 3-way Split
    train_end = pd.Timestamp("2024-04-30 23:59:59", tz="UTC")
    val_end = pd.Timestamp("2025-05-31 23:59:59", tz="UTC")

    conditions = [
        df["started_at"] <= train_end,
        (df["started_at"] > train_end) & (df["started_at"] <= val_end),
        df["started_at"] > val_end,
    ]
    choices = ["train", "val", "test"]
    df["split_set"] = np.select(conditions, choices, default="train")

    # Сортування за хронологією
    df = df.sort_values("started_at").reset_index(drop=True)

    # 10. Збереження у Parquet
    print(f"💾 Збереження у {output_path} ({len(df):,} рядків)...")
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print("✅ Очищення успішно завершено!")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "official_data_uk.csv"
    processed_parquet = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"

    cleaned_df = clean_and_profile_data(raw_csv, processed_parquet)

    print("\n" + "=" * 60)
    print("🚦 QUALITY GATE 1: ПЕРЕВІРКА КРИТЕРІЇВ")
    print("=" * 60)

    # Check 1: File created and readable
    assert processed_parquet.exists(), "Gate 1 Fail: Parquet file does not exist!"
    assert len(cleaned_df) > 120000, "Gate 1 Fail: Row count is insufficient!"
    print(f"✔ Критерій 1: Файл Parquet існує і містить {len(cleaned_df):,} рядків.")

    # Check 2: Zero negative duration
    neg_duration = (cleaned_df[cleaned_df["is_finished"]]["duration_seconds"] < 0).sum()
    assert neg_duration == 0, "Gate 1 Fail: Negative duration records found!"
    print(f"✔ Критерій 2: Записів з від'ємною тривалістю: {neg_duration} (ідеально).")

    # Check 3: 100% Kyiv timezone conversion
    assert cleaned_df["started_at_kyiv"].notna().all(), "Gate 1 Fail: Incomplete Kyiv timezone conversion!"
    print("✔ Критерій 3: Часовий пояс Europe/Kyiv розраховано для 100% записів.")

    # Check 4: Strict Chronological Split
    train_dates = cleaned_df[cleaned_df['split_set'] == 'train']['started_at']
    val_dates = cleaned_df[cleaned_df['split_set'] == 'val']['started_at']
    test_dates = cleaned_df[cleaned_df['split_set'] == 'test']['started_at']

    assert train_dates.max() < val_dates.min(), "Gate 1 Fail: Train overlaps with Validation!"
    assert val_dates.max() < test_dates.min(), "Gate 1 Fail: Validation overlaps with Test!"
    print(f"✔ Критерій 4: Хронологічний спліт суворий і монотонний:")
    print(f"     Train:      {len(train_dates):,} рядків ({train_dates.min().date()} — {train_dates.max().date()})")
    print(f"     Validation: {len(val_dates):,} рядків ({val_dates.min().date()} — {val_dates.max().date()})")
    print(f"     Test:       {len(test_dates):,} рядків ({test_dates.min().date()} — {test_dates.max().date()})")

    print("\n🎉 QUALITY GATE 1 УСПІШНО ПРОЙДЕНО!")
