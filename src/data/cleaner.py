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
    # Каузальний фільтр мікро-тривог (< 60 с): відкидаємо лише ті короткі записи, чий відбій відбувся
    # до настання чергової годинної точки прийняття рішення T (started_at.dt.ceil('h') == finished_at.dt.ceil('h')).
    # Якщо тривога почалася до T і закінчилася після T (started_at <= T < finished_at), у момент T її майбутній
    # час відбою ще невідомий, тому вона не може ретроспективно видалятися на основі майбутньої тривалості.
    same_hour_window = df["started_at"].dt.ceil("h") == df["finished_at"].dt.ceil("h")
    glitch_mask = is_fin_raw & (dur_raw < min_duration_seconds) & same_hour_window
    glitch_removed = int(glitch_mask.sum())
    df = df[~glitch_mask].copy()
    after_glitch_count = len(df)
    print(f"   - Видалено каузальних технічних мікро-тривог (< {min_duration_seconds}с у межах години): {glitch_removed:,} | Залишилось: {after_glitch_count:,}")

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


def compute_train_median_durations_volunteer(
    raw_df: pd.DataFrame,
    train_start: pd.Timestamp = pd.Timestamp("2022-03-15 00:00:00", tz="UTC"),
    train_end: pd.Timestamp = pd.Timestamp("2024-04-30 23:59:59", tz="UTC"),
    min_duration_seconds: int = 60,
) -> dict[str, float]:
    """
    Обчислює каузальну медіанну тривалість D_oblast (у секундах) для кожної області
    виключно на даних Train (2022-03-15 .. 2024-04-30 23:59:59 UTC) серед тривог із naive == False.
    Тривоги, що перетинають train_end, цензуруються по train_end.
    """
    df = raw_df.copy()
    if "region" in df.columns:
        df = df.rename(columns={"region": "oblast"})
    df["oblast"] = df["oblast"].replace({"Київ": "м. Київ"})
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    train_sub = df[
        (df["started_at"] >= train_start)
        & (df["started_at"] <= train_end)
        & (~df["naive"].astype(bool))
        & df["finished_at"].notna()
    ].copy()

    censored_end = train_sub["finished_at"].clip(upper=train_end)
    dur_sec = (censored_end - train_sub["started_at"]).dt.total_seconds()
    train_sub["dur_sec"] = dur_sec
    train_sub = train_sub[train_sub["dur_sec"] >= min_duration_seconds]

    medians = train_sub.groupby("oblast")["dur_sec"].median().to_dict()
    global_median = float(train_sub["dur_sec"].median())
    for ob in df["oblast"].unique():
        if ob not in medians or pd.isna(medians[ob]):
            medians[ob] = global_median
    return {k: float(v) for k, v in medians.items()}


def merge_volunteer_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Об'єднує перетинні або суміжні інтервали [started_at, finished_at] у межах кожної області для волонтерських даних."""
    df_sorted = df.sort_values(["oblast", "started_at", "finished_at"]).reset_index(drop=True)
    merged_rows = []

    for oblast, group in df_sorted.groupby("oblast", sort=False):
        starts = group["started_at"].tolist()
        ends = group["finished_at"].tolist()
        naives = group["naive"].astype(bool).tolist()

        if not starts:
            continue

        cur_start = starts[0]
        cur_end = ends[0]
        cur_naive = naives[0]

        for i in range(1, len(starts)):
            s = starts[i]
            e = ends[i]
            nv = naives[i]

            if pd.isna(cur_end) or s <= cur_end:
                if pd.isna(cur_end) or pd.isna(e):
                    cur_end = pd.NaT
                elif e > cur_end:
                    cur_end = e
                cur_naive = cur_naive or nv
            else:
                merged_rows.append({
                    "oblast": oblast,
                    "raion": None,
                    "hromada": None,
                    "level": "oblast",
                    "started_at": cur_start,
                    "finished_at": cur_end,
                    "naive": cur_naive,
                    "source": "volunteer",
                })
                cur_start = s
                cur_end = e
                cur_naive = nv

        merged_rows.append({
            "oblast": oblast,
            "raion": None,
            "hromada": None,
            "level": "oblast",
            "started_at": cur_start,
            "finished_at": cur_end,
            "naive": cur_naive,
            "source": "volunteer",
        })

    res = pd.DataFrame(merged_rows)
    res["started_at"] = pd.to_datetime(res["started_at"], utc=True)
    res["finished_at"] = pd.to_datetime(res["finished_at"], utc=True)
    return res


def clean_volunteer_data(
    input_csv_path: str | Path,
    output_parquet_path: str | Path,
    naive_mode: str = "naive_as_is",
    train_start: pd.Timestamp = pd.Timestamp("2022-03-15 00:00:00", tz="UTC"),
    min_duration_seconds: int = 60,
    precomputed_d_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Очищує волонтерський датасет volunteer_data_uk.csv (eTryvoga).
    Підтримує три каузальні режими обробки naive == True:
      - 'naive_as_is': залишає finished_at = started_at + 30 хв як у сирому файлі;
      - 'drop_naive': видаляє записи з naive == True;
      - 'naive_fixed_D': замінює finished_at = started_at + D_oblast для naive == True,
        де D_oblast — медіанна тривалість не-наївних тривог в області строго на Train (<= 2024-04-30).
    """
    if naive_mode not in ("naive_as_is", "drop_naive", "naive_fixed_D"):
        raise ValueError(f"Unknown naive_mode: {naive_mode}")

    input_path = Path(input_csv_path)
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📖 [Volunteer: {naive_mode}] Читання {input_path}...")
    raw_df = pd.read_csv(input_path)
    raw_count = len(raw_df)

    # 1. Дедуплікація сирого CSV
    raw_dups = int(raw_df.duplicated().sum())
    df = raw_df.drop_duplicates(keep="first").copy()
    assert int(df.duplicated().sum()) == 0

    # 2. Адаптер схеми: region -> oblast, Київ -> м. Київ
    df = df.rename(columns={"region": "oblast"})
    df["oblast"] = df["oblast"].replace({"Київ": "м. Київ"})
    df["naive"] = df["naive"].astype(bool)

    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    # 3. Обмеження старту Train від 2022-03-15 00:00:00 UTC (для порівнянності з official)
    before_train_start = int((df["started_at"] < train_start).sum())
    df = df[df["started_at"] >= train_start].copy()

    # 4. Фільтрація технічних збоїв (від'ємна тривалість або каузальні мікро-тривоги < 60с у межах години)
    is_fin = df["finished_at"].notna()
    dur_raw = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    neg_mask = is_fin & (dur_raw < 0)
    if int(neg_mask.sum()) > 0:
        df = df[~neg_mask].copy()
    is_fin = df["finished_at"].notna()
    dur_raw = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    same_hour_window = df["started_at"].dt.ceil("h") == df["finished_at"].dt.ceil("h")
    glitch_mask = is_fin & (dur_raw < min_duration_seconds) & same_hour_window
    glitch_removed = int(glitch_mask.sum())
    df = df[~glitch_mask].copy()

    # 5. Обробка режиму naive
    train_end = pd.Timestamp("2024-04-30 23:59:59", tz="UTC")
    val_end = pd.Timestamp("2025-05-31 23:59:59", tz="UTC")

    if naive_mode == "drop_naive":
        naive_dropped = int(df["naive"].sum())
        df = df[~df["naive"]].copy()
        print(f"   - [drop_naive] Видалено записів з naive=True: {naive_dropped:,}")
    elif naive_mode == "naive_fixed_D":
        d_map = precomputed_d_map or compute_train_median_durations_volunteer(
            raw_df, train_start=train_start, train_end=train_end, min_duration_seconds=min_duration_seconds
        )
        naive_mask = df["naive"].values
        d_seconds = df.loc[naive_mask, "oblast"].map(d_map).values
        df.loc[naive_mask, "finished_at"] = (
            df.loc[naive_mask, "started_at"] + pd.to_timedelta(d_seconds, unit="s")
        )
        print(f"   - [naive_fixed_D] Замінено finished_at = started_at + D_oblast для {int(naive_mask.sum()):,} записів")

    # 6. Об'єднання перетинних/суміжних інтервалів у межах області
    pre_merge_len = len(df)
    df = merge_volunteer_intervals(df)
    merged_diff = pre_merge_len - len(df)
    print(
        f"   - Сирих: {raw_count:,} (дубл: {raw_dups}, до {train_start.date()}: -{before_train_start:,}, "
        f"<60s: -{glitch_removed:,}) -> До злиття: {pre_merge_len:,} -> Після злиття: {len(df):,} (злито: {merged_diff:,})"
    )

    assert int(df.duplicated(subset=["oblast", "started_at", "finished_at"]).sum()) == 0
    assert int(df.duplicated(subset=["oblast", "started_at"]).sum()) == 0

    df["is_finished"] = df["finished_at"].notna()
    duration_s = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df["duration_seconds"] = duration_s
    df["duration_minutes"] = duration_s / 60.0
    df["duration_hours"] = duration_s / 3600.0

    df["is_occupied_extreme"] = df["oblast"].isin(OCCUPIED_EXTREME_REGIONS)

    kyiv_tz = "Europe/Kyiv"
    df["started_at_kyiv"] = df["started_at"].dt.tz_convert(kyiv_tz)
    df["finished_at_kyiv"] = df["finished_at"].dt.tz_convert(kyiv_tz)
    df["start_year"] = df["started_at_kyiv"].dt.year
    df["start_month"] = df["started_at_kyiv"].dt.month
    df["start_date"] = df["started_at_kyiv"].dt.date
    df["start_hour_kyiv"] = df["started_at_kyiv"].dt.hour
    df["start_day_of_week"] = df["started_at_kyiv"].dt.day_name()
    df["start_day_num"] = df["started_at_kyiv"].dt.dayofweek
    df["is_night"] = df["start_hour_kyiv"].between(0, 5)

    conditions = [
        df["started_at"] <= train_end,
        (df["started_at"] > train_end) & (df["started_at"] <= val_end),
        df["started_at"] > val_end,
    ]
    df["split_set"] = np.select(conditions, ["train", "val", "test"], default="train")
    df = df.sort_values("started_at").reset_index(drop=True)

    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"   💾 Збережено у {output_path} ({len(df):,} рядків)")
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

    vol_csv = project_root / "data" / "raw" / "volunteer_data_uk.csv"
    if vol_csv.exists():
        raw_vol_df = pd.read_csv(vol_csv)
        d_map = compute_train_median_durations_volunteer(raw_vol_df)
        d_df = pd.DataFrame([
            {"oblast": k, "train_median_duration_sec": v, "train_median_duration_min": round(v / 60.0, 2)}
            for k, v in sorted(d_map.items())
        ])
        d_out = project_root / "reports" / "volunteer" / "volunteer_train_median_durations.csv"
        d_out.parent.mkdir(parents=True, exist_ok=True)
        d_df.to_csv(d_out, index=False)
        print(f"\n📊 Збережено каузальні медіанні тривалості D_oblast (Train, naive==False) у {d_out}")

        clean_volunteer_data(
            vol_csv,
            project_root / "data" / "processed" / "volunteer_alerts_clean_as_is.parquet",
            naive_mode="naive_as_is",
        )
        clean_volunteer_data(
            vol_csv,
            project_root / "data" / "processed" / "volunteer_alerts_clean_drop_naive.parquet",
            naive_mode="drop_naive",
        )
        clean_volunteer_data(
            vol_csv,
            project_root / "data" / "processed" / "volunteer_alerts_clean_fixed_d.parquet",
            naive_mode="naive_fixed_D",
            precomputed_d_map=d_map,
        )

    print("\n🎉 QUALITY GATE 1 УСПІШНО ПРОЙДЕНО!")

