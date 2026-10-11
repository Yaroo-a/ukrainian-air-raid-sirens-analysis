"""Data Ingestion & Cleaning Module (Phase 1).

Performs:
1. Reads raw official_data_uk.csv (299,275 rows) and audits upstream duplication
   (block 0..113,910 repeated in 113,911..227,831 -> 113,845 exact duplicate rows).
2. Deduplicates exact duplicates and asserts 0 duplicates remain.
3. Parses timestamps to UTC, logs and removes negative durations and micro-alerts (< 60s).
4. Handles the post-August 2025 logging format transition:
   - Before 2025-08-01: uses 'oblast'-level records (when all 24 oblasts logged oblast-wide alerts
     directly under level == 'oblast').
   - From 2025-08-01 onwards (when 'oblast' records dropped to 22-328/month as 23 oblasts switched
     to 'raion'/'hromada' logging): reconstructs oblast-level alerts from 'oblast', 'raion', and 'hromada'
     (an oblast is active whenever any alert within it is active, merging overlapping intervals).
5. Merges overlapping or contiguous intervals [started_at, finished_at] per oblast and asserts 0
   duplicate or overlapping intervals remain after cleaning.
6. Derives Europe/Kyiv local time features, flags extreme occupied regions, and assigns
   chronological 3-way split (Train, Validation, Test) in UTC.
7. Saves output to Apache Parquet (data/processed/alerts_clean_oblast.parquet).
"""

from pathlib import Path
import pandas as pd
import numpy as np

OCCUPIED_EXTREME_REGIONS = [
    "Луганська область",
    "Автономна Республіка Крим",
    "м. Севастополь",
]

SUBREGIONAL_TRANSITION_UTC = pd.Timestamp("2025-08-01 00:00:00", tz="UTC")


def merge_oblast_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Merges overlapping or contiguous alert intervals per oblast."""
    df_sorted = df.sort_values(["oblast", "started_at", "finished_at"]).reset_index(drop=True)
    merged_rows = []

    for oblast, group in df_sorted.groupby("oblast", sort=False):
        starts = group["started_at"].tolist()
        ends = group["finished_at"].tolist()
        levels = group["level"].tolist()

        if not starts:
            continue

        cur_start = starts[0]
        cur_end = ends[0]
        has_oblast_lvl = (levels[0] == "oblast")

        for i in range(1, len(starts)):
            s = starts[i]
            e = ends[i]
            lvl = levels[i]

            if pd.isna(cur_end) or s <= cur_end:
                if pd.isna(cur_end) or pd.isna(e):
                    cur_end = pd.NaT
                elif e > cur_end:
                    cur_end = e
                if lvl == "oblast":
                    has_oblast_lvl = True
            else:
                merged_rows.append({
                    "oblast": oblast,
                    "raion": None,
                    "hromada": None,
                    "level": "oblast" if has_oblast_lvl else "raion_merged",
                    "started_at": cur_start,
                    "finished_at": cur_end,
                    "source": "official",
                })
                cur_start = s
                cur_end = e
                has_oblast_lvl = (lvl == "oblast")

        merged_rows.append({
            "oblast": oblast,
            "raion": None,
            "hromada": None,
            "level": "oblast" if has_oblast_lvl else "raion_merged",
            "started_at": cur_start,
            "finished_at": cur_end,
            "source": "official",
        })

    res = pd.DataFrame(merged_rows)
    res["started_at"] = pd.to_datetime(res["started_at"], utc=True)
    res["finished_at"] = pd.to_datetime(res["finished_at"], utc=True)
    return res


def clean_and_profile_data(
    input_csv_path: str | Path,
    output_parquet_path: str | Path,
    min_duration_seconds: int = 60,
) -> pd.DataFrame:
    """Performs full Phase 1 ingestion, deduplication audit, interval merging, and verification."""
    input_path = Path(input_csv_path)
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📖 1. Читання сирого CSV-файлу: {input_path}...")
    df = pd.read_csv(input_path)
    raw_count = len(df)

    # Перевірка якості даних ДО очищення (аудит дублікатів)
    dup_mask = df.duplicated(keep="first")
    raw_dups_count = int(dup_mask.sum())
    dup_indices = df.index[dup_mask]
    min_dup_idx = int(dup_indices.min()) if raw_dups_count > 0 else -1
    max_dup_idx = int(dup_indices.max()) if raw_dups_count > 0 else -1

    print(f"   [Аудит якості ДО очищення]:")
    print(f"   - Всього сирих рядків: {raw_count:,}")
    print(f"   - Виявлено точних дублікатів: {raw_dups_count:,} (діапазон індексів: {min_dup_idx:,}..{max_dup_idx:,})")

    # 1. Видалення точних дублікатів у сирому CSV
    df = df.drop_duplicates(keep="first").copy()
    dedup_count = len(df)
    post_dedup_dups = int(df.duplicated().sum())
    assert post_dedup_dups == 0, f"Deduplication failed: {post_dedup_dups} duplicates remain!"
    print(f"   - Видалено точних дублікатів: {raw_count - dedup_count:,} | Залишилось унікальних рядків: {dedup_count:,} (дублікатів після drop_duplicates: {post_dedup_dups})")

    # 2. Парсинг часових міток у UTC
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    print(f"   - Діапазон сирих даних (UTC): {df['started_at'].min()} — {df['started_at'].max()}")

    # 3. Первинна фільтрація некоректних записів та мікро-глітчів (< 60 сек)
    is_fin_raw = df["finished_at"].notna()
    dur_raw = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    neg_mask = is_fin_raw & (dur_raw < 0)
    neg_removed = int(neg_mask.sum())
    if neg_removed > 0:
        df = df[~neg_mask].copy()
    print(f"   - Видалено записів з від'ємною тривалістю (finished_at < started_at): {neg_removed:,}")

    is_fin_raw = df["finished_at"].notna()
    dur_raw = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    glitch_mask = is_fin_raw & (dur_raw < min_duration_seconds)
    glitch_removed = int(glitch_mask.sum())
    df = df[~glitch_mask].copy()
    after_glitch_count = len(df)
    print(f"   - Видалено технічних мікро-тривог (< {min_duration_seconds}с): {glitch_removed:,} | Залишилось: {after_glitch_count:,}")

    # 4. Відбір за рівнем: до 2025-08-01 — 'oblast', з 2025-08-01 — всі рівні ('oblast', 'raion', 'hromada')
    print(f"🔗 2. Побудова обласних тривог (до {SUBREGIONAL_TRANSITION_UTC.date()} level=='oblast', після — об'єднання з raion/hromada)...")
    mask_keep = (
        ((df["started_at"] < SUBREGIONAL_TRANSITION_UTC) & (df["level"] == "oblast"))
        | (df["started_at"] >= SUBREGIONAL_TRANSITION_UTC)
    )
    subregional_pre_transition_removed = int((~mask_keep).sum())
    df_filtered = df[mask_keep].copy()
    pre_merge_count = len(df_filtered)
    print(f"   - Відфільтровано субрегіональних записів до {SUBREGIONAL_TRANSITION_UTC.date()}: {subregional_pre_transition_removed:,} | Передано на об'єднання інтервалів: {pre_merge_count:,}")

    df = merge_oblast_intervals(df_filtered)
    merged_consolidated = pre_merge_count - len(df)
    print(f"   - Консолідовано перетинних районних/обласних записів при об'єднанні інтервалів: {merged_consolidated:,}")
    print(f"   - Отримано неперервних інтервалів тривог на рівні областей: {len(df):,}")
    print(f"     (з них прямих 'oblast': {(df['level'] == 'oblast').sum():,}, агрегованих з районів/громад: {(df['level'] == 'raion_merged').sum():,})")

    # Явна перевірка відсутності дублікатів ПІСЛЯ очищення та об'єднання інтервалів
    final_exact_dups = int(df.duplicated().sum())
    final_interval_dups = int(df.duplicated(subset=["oblast", "started_at", "finished_at"]).sum())
    final_start_dups = int(df.duplicated(subset=["oblast", "started_at"]).sum())
    print(f"   [Аудит якості ПІСЛЯ очищення]:")
    print(f"   - Точних дублікатів рядків: {final_exact_dups}")
    print(f"   - Дублікатів інтервалів (oblast, started_at, finished_at): {final_interval_dups}")
    print(f"   - Дублікатів стартів (oblast, started_at): {final_start_dups}")
    assert final_exact_dups == 0, f"Found {final_exact_dups} exact duplicates after cleaning!"
    assert final_interval_dups == 0, f"Found {final_interval_dups} interval duplicates after cleaning!"
    assert final_start_dups == 0, f"Found {final_start_dups} start duplicates after cleaning!"

    # 5. Обробка завершеності та розрахунок тривалості об'єднаних інтервалів
    df["is_finished"] = df["finished_at"].notna()
    unfinished_count = int((~df["is_finished"]).sum())
    print(f"   - Незавершених тривог (без finished_at): {unfinished_count}")

    duration_s = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df["duration_seconds"] = duration_s
    df["duration_minutes"] = duration_s / 60.0
    df["duration_hours"] = duration_s / 3600.0

    # 6. Маркування ТОТ
    df["is_occupied_extreme"] = df["oblast"].isin(OCCUPIED_EXTREME_REGIONS)
    occupied_count = int(df["is_occupied_extreme"].sum())
    print(f"   - Записів ТОТ (Луганщина/Крим/Севастополь): {occupied_count:,}")

    # 7. Локальний час Europe/Kyiv
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

    # 8. Хронологічний 3-way Split (UTC)
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

    # 9. Збереження у Parquet
    print(f"💾 Збереження у {output_path} ({len(df):,} рядків)...")
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print("✅ Очищення та агрегацію інтервалів успішно завершено!")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "official_data_uk.csv"
    processed_parquet = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"

    cleaned_df = clean_and_profile_data(raw_csv, processed_parquet)

    print("\n" + "=" * 60)
    print("🚦 QUALITY GATE 1: ПЕРЕВІРКА КРИТЕРІЇВ")
    print("=" * 60)

    assert processed_parquet.exists(), "Gate 1 Fail: Parquet file does not exist!"
    assert len(cleaned_df) > 70000, "Gate 1 Fail: Row count is insufficient!"
    print(f"✔ Критерій 1: Файл Parquet існує і містить {len(cleaned_df):,} об'єднаних обласних тривог.")

    dup_after = int(cleaned_df.duplicated(subset=["oblast", "started_at", "finished_at"]).sum())
    assert dup_after == 0, f"Gate 1 Fail: {dup_after} duplicate records found after cleaning!"
    print(f"✔ Критерій 2: Дублікатів після очищення: {dup_after} (assert == 0 пройдено).")

    neg_duration = int((cleaned_df[cleaned_df["is_finished"]]["duration_seconds"] < 0).sum())
    assert neg_duration == 0, "Gate 1 Fail: Negative duration records found!"
    print(f"✔ Критерій 3: Записів з від'ємною тривалістю: {neg_duration}.")

    assert cleaned_df["started_at_kyiv"].notna().all(), "Gate 1 Fail: Incomplete Kyiv timezone conversion!"
    print("✔ Критерій 4: Часовий пояс Europe/Kyiv розраховано для 100% записів.")

    train_dates = cleaned_df[cleaned_df["split_set"] == "train"]["started_at"]
    val_dates = cleaned_df[cleaned_df["split_set"] == "val"]["started_at"]
    test_dates = cleaned_df[cleaned_df["split_set"] == "test"]["started_at"]

    assert train_dates.max() < val_dates.min(), "Gate 1 Fail: Train overlaps with Validation!"
    assert val_dates.max() < test_dates.min(), "Gate 1 Fail: Validation overlaps with Test!"
    print("✔ Критерій 5: Хронологічний спліт суворий і монотонний:")
    print(f"     Train:      {len(train_dates):,} рядків ({train_dates.min()} — {train_dates.max()})")
    print(f"     Validation: {len(val_dates):,} рядків ({val_dates.min()} — {val_dates.max()})")
    print(f"     Test:       {len(test_dates):,} рядків ({test_dates.min()} — {test_dates.max()})")

    print("\n🎉 QUALITY GATE 1 УСПІШНО ПРОЙДЕНО!")
