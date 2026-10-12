"""Тест каузальної інваріантності ознак A/B/C/C'/D, Train-only кластерів, цільової змінної,
каузальної імп'ютації naive_fixed_D та збережених Out-of-Fold прогнозів Walk-Forward
(Causal Truncation, Target & Walk-Forward OOF Invariance Test).

Перевіряє:
1. Відсутність дублікатів у alerts_clean_oblast.parquet, ml_features_hourly.parquet,
   volunteer_alerts_clean_*.parquet, volunteer_ml_features_hourly_*.parquet та
   walk_forward_oof_predictions_*.parquet (assert 0).
2. Каузальність Train-only кластерів (clusters_train_only.csv та volunteer_clusters_train_only.csv):
   - Перевірка, що кластеризація областей (low_activity, medium_activity, high_activity) на повному
     датасеті побітово ідентична кластеризації на датасеті, фізично обрізаному по 2024-04-30 23:59:59 UTC
     (із цензуруванням тривалостей finished_at на межі train_end).
   - Перевірка відповідності one-hot колонок cluster_1_low_activity, cluster_2_medium_activity,
     cluster_3_high_activity у кожному рядку Feature Store значенню Train-only кластера області.
3. Інваріантність усіх 22 ознак набору C (та A/B/C') і 3 похідних відношень набору D при фізичному
   обрізанні майбутнього на момент T (>= 119 моментів T: граничні + 100 випадкових у Train, Val, Test).
4. Залежність таргету виключно від вікна [T, T + 1h).
5. Каузальність варіанта naive_fixed_D на волонтерських даних (незмінність D_oblast при видаленні Val+Test).
6. Каузальність Walk-Forward сплітів (2-годинний embargo для [T, T+1h)) та верифікацію збережених
   Out-of-Fold прогнозів (walk_forward_oof_predictions_official.parquet та
   walk_forward_oof_predictions_volunteer_fixed_d.parquet):
   - SHA-256 збігається з walk_forward_details.json;
   - точний збіг індексу (timestamp_utc, oblast), таргету y та p_adapt з Feature Store (split_set == 'test');
   - точний збіг p_stat_lgb_c із прогнозом замороженої статичної моделі .joblib на FEATURE_SET_C;
   - точне відтворення p_roll_lgb_c на контрольному місяці M при навчанні виключно на фізично обрізаному
     до T_start(M) - 2h датасеті (коли жодного рядка з місяця M чи майбутнього немає в пам'яті).

Обмеження тесту: тест працює поверх очищених інтервалів *_alerts_clean_*.parquet (та сирого CSV для D_oblast),
тому він перевіряє каузальність генератора ознак feature_store.py, train-only кластерів, train-only медіан D_oblast
та Walk-Forward OOF прогнозів, але не бачить етапу первинного відсіювання мікро-тривог < 60 с у cleaner.py.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.data.cleaner import compute_train_median_durations_volunteer
from src.features.clustering import generate_train_only_clusters
from src.features.feature_store import build_neighbor_dict
from src.models.train_models import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_C_PRIME,
)
from src.models.run_walk_forward_experiment import (
    FEATURE_SET_D_POST_HOC,
    add_post_hoc_set_d_features,
    fast_threshold_search_f2,
    fit_fold_lightgbm,
    fit_fold_logreg,
    get_walk_forward_split_masks,
)


def _verify_train_only_clusters_invariance(
    clean_parquet_path: Path,
    saved_clusters_csv: Path,
    label: str,
    train_end_str: str = "2024-04-30 23:59:59",
) -> dict[str, int]:
    """Перевіряє, що Train-only кластери (активність 2022–04.2024) не змінюються при фізичному видаленні Val і Test."""
    assert clean_parquet_path.exists(), f"Missing {clean_parquet_path}"
    assert saved_clusters_csv.exists(), f"Missing {saved_clusters_csv}"

    saved_df = pd.read_csv(saved_clusters_csv).sort_values("oblast").reset_index(drop=True)
    assert len(saved_df) == 24, f"Expected 24 oblasts in {saved_clusters_csv}, got {len(saved_df)}"

    df_full = pd.read_parquet(clean_parquet_path)
    train_end = pd.Timestamp(train_end_str, tz="UTC")

    # Фізично обрізаємо всі рядки після train_end і цензуруємо finished_at на train_end
    df_trunc = df_full[df_full["started_at"] <= train_end].copy()
    df_trunc["finished_at"] = df_trunc["finished_at"].fillna(train_end).clip(upper=train_end)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pq = Path(tmpdir) / "trunc_clean.parquet"
        tmp_csv = Path(tmpdir) / "trunc_clusters.csv"
        df_trunc.to_parquet(tmp_pq, index=False)
        generate_train_only_clusters(tmp_pq, tmp_csv, train_end_utc=train_end_str)
        trunc_cl_df = pd.read_csv(tmp_csv).sort_values("oblast").reset_index(drop=True)

    assert list(saved_df["oblast"]) == list(trunc_cl_df["oblast"])
    assert list(saved_df["cluster"]) == list(trunc_cl_df["cluster"]), (
        f"[{label}] Train-only cluster IDs leaked from post-Train data!"
    )
    assert list(saved_df["cluster_name"]) == list(trunc_cl_df["cluster_name"]), (
        f"[{label}] Train-only cluster names leaked from post-Train data!"
    )
    for col in ["total_alerts", "alerts_per_month", "median_duration_min", "night_ratio", "short_ratio", "long_ratio"]:
        assert np.allclose(saved_df[col].values, trunc_cl_df[col].values, atol=1e-9), (
            f"[{label}] Cluster passport metric {col} differs on truncated Train data!"
        )

    print(
        f"✔ [{label}] Train-Only Clusters Invariance PASSED: "
        f"кластери 24 областей ({saved_clusters_csv.name}) не залежать від даних після {train_end}."
    )
    return dict(zip(saved_df["oblast"], saved_df["cluster"].astype(int)))


def _run_truncation_checks_on_pair(
    parquet_path: Path,
    feature_store_path: Path,
    clusters_csv_path: Path,
    label: str,
    n_random: int = 100,
) -> tuple[int, int]:
    assert parquet_path.exists(), f"Missing {parquet_path}"
    assert feature_store_path.exists(), f"Missing {feature_store_path}"

    expected_cluster_map = _verify_train_only_clusters_invariance(
        parquet_path, clusters_csv_path, label=label
    )

    raw_df = pd.read_parquet(parquet_path)
    store_df = pd.read_parquet(feature_store_path)

    # Перевірка на нуль дублікатів
    clean_dups = int(raw_df.duplicated(subset=["oblast", "started_at", "finished_at"]).sum())
    store_dups = int(store_df.duplicated(subset=["timestamp_utc", "oblast"]).sum())
    assert clean_dups == 0, f"Found {clean_dups} duplicates in {parquet_path}!"
    assert store_dups == 0, f"Found {store_dups} duplicates in {feature_store_path}!"

    # Перевірка узгодженості наборів ознак A (10), B (13), C (22), C' (19)
    assert len(FEATURE_SET_A) == 10 and set(FEATURE_SET_A).issubset(store_df.columns)
    assert len(FEATURE_SET_B) == 13 and set(FEATURE_SET_B).issubset(store_df.columns)
    assert len(FEATURE_SET_C) == 22 and set(FEATURE_SET_C).issubset(store_df.columns)
    assert len(FEATURE_SET_C_PRIME) == 19 and set(FEATURE_SET_C_PRIME).issubset(store_df.columns)
    assert set(FEATURE_SET_C) - set(FEATURE_SET_C_PRIME) == {
        "national_active_count",
        "hours_since_last_alert",
        "onsets_last_24h",
    }

    # Перевірка поточкової каузальності похідних ознак Set D (відсутність глобальних статистик по стовпцю)
    sub_sample = store_df.iloc[:500].copy()
    full_with_d = add_post_hoc_set_d_features(store_df.iloc[:1000].copy())
    sub_with_d = add_post_hoc_set_d_features(sub_sample)
    for dcol in ["rel_activity_24h_vs_14d", "adaptive_ratio_7d_vs_30d", "adaptive_diff_14d_minus_30d"]:
        assert np.array_equal(full_with_d[dcol].iloc[:500].values, sub_with_d[dcol].values), (
            f"[{label}] Set D feature {dcol} is not pointwise causal!"
        )

    raw_clean = raw_df[~raw_df["is_occupied_extreme"]].copy()
    min_global_time = raw_clean["started_at"].min().floor("h")

    all_timestamps = store_df["timestamp_utc"].drop_duplicates().sort_values()
    all_ts_set = set(all_timestamps)

    boundary_candidates = [
        min_global_time + pd.Timedelta(hours=k)
        for k in [0, 1, 2, 3, 6, 23, 24, 25, 167, 168, 335, 336, 719, 720]
    ] + [
        pd.Timestamp("2024-04-30 21:00:00", tz="UTC"),
        pd.Timestamp("2024-05-01 00:00:00", tz="UTC"),
        pd.Timestamp("2025-05-31 21:00:00", tz="UTC"),
        pd.Timestamp("2025-06-01 00:00:00", tz="UTC"),
        all_timestamps.iloc[-1],
    ]
    boundary_timestamps = [t for t in boundary_candidates if t in all_ts_set]

    remaining_pool = all_timestamps[~all_timestamps.isin(boundary_timestamps)]
    random_timestamps = remaining_pool.sample(n_random, random_state=42).tolist()

    sampled_timestamps = sorted(set(boundary_timestamps + random_timestamps))
    assert len(sampled_timestamps) >= 100, f"Expected >= 100 timestamps, got {len(sampled_timestamps)}"

    oblasts = sorted(raw_clean["oblast"].unique())
    neighbor_map = build_neighbor_dict(oblasts)

    checked_rows = 0

    for T in sampled_timestamps:
        past_only = raw_clean[raw_clean["started_at"] <= T].copy()
        far_future = pd.Timestamp("2099-01-01", tz="UTC")
        past_only["finished_known"] = past_only["finished_at"].where(
            past_only["finished_at"].notna() & (past_only["finished_at"] <= T),
            other=pd.NaT,
        )
        past_only["finished_censored"] = past_only["finished_known"].fillna(far_future)

        n_hist = min(720, int((T - min_global_time).total_seconds() // 3600))
        hist_ticks = pd.date_range(end=T, periods=n_hist + 1, freq="1h", tz="UTC")

        active_hist = {}
        onset_hist = {}
        hours_since_last_map = {}

        for o in oblasts:
            o_alerts = past_only[past_only["oblast"] == o]
            s_arr = o_alerts["started_at"].values
            e_arr = o_alerts["finished_censored"].values

            ticks_val = hist_ticks.values[:, None]
            if len(s_arr) > 0:
                act_vec = ((s_arr[None, :] <= ticks_val) & (ticks_val < e_arr[None, :])).any(axis=1).astype(int)
                if len(hist_ticks) > 1:
                    prev_ticks = hist_ticks[:-1].values[:, None]
                    next_ticks = (hist_ticks[:-1] + pd.Timedelta(hours=1)).values[:, None]
                    ons_vec = ((s_arr[None, :] >= prev_ticks) & (s_arr[None, :] < next_ticks)).any(axis=1).astype(int)
                else:
                    ons_vec = np.zeros(0, dtype=int)
            else:
                act_vec = np.zeros(len(hist_ticks), dtype=int)
                ons_vec = np.zeros(max(0, len(hist_ticks) - 1), dtype=int)

            active_hist[o] = act_vec
            onset_hist[o] = ons_vec

            known_ends = o_alerts["finished_known"].dropna()
            if len(known_ends) > 0:
                last_end = known_ends.max()
                elapsed = (T - last_end).total_seconds() / 3600.0
                hours_since_last_map[o] = np.float32(np.clip(elapsed, 0.0, 720.0))
            else:
                hours_since_last_map[o] = np.float32(720.0)

        exp_national_active = sum(int(active_hist[o][-1]) for o in oblasts)

        store_slice = store_df[store_df["timestamp_utc"] == T]
        calm_oblasts_expected = {o for o in oblasts if active_hist[o][-1] == 0}
        calm_oblasts_actual = set(store_slice["oblast"].unique())
        assert calm_oblasts_actual == calm_oblasts_expected, (
            f"[{label}] Calm filter mismatch at T={T}: actual={calm_oblasts_actual ^ calm_oblasts_expected}"
        )

        t_kyiv = T.tz_convert("Europe/Kyiv")
        exp_hour_sin = np.float32(np.sin(2 * np.pi * t_kyiv.hour / 24.0))
        exp_hour_cos = np.float32(np.cos(2 * np.pi * t_kyiv.hour / 24.0))
        exp_dow = int(t_kyiv.dayofweek)
        exp_weekend = int(exp_dow >= 5)
        exp_month = int(t_kyiv.month)

        T_next = T + pd.Timedelta(hours=1)
        future_window_only = raw_clean[
            (raw_clean["started_at"] >= T) & (raw_clean["started_at"] < T_next)
        ]
        oblasts_with_onset = set(future_window_only["oblast"].unique())

        for _, row in store_slice.iterrows():
            o = row["oblast"]
            act_past = active_hist[o][:-1]
            ons_past = onset_hist[o]

            exp_lag_1h = int(ons_past[-1]) if len(ons_past) > 0 else 0
            assert int(row["lagged_onset_1h"]) == exp_lag_1h

            exp_b6 = np.float32(act_past[-6:].mean()) if len(act_past) > 0 else np.float32(0.0)
            exp_b24 = np.float32(act_past[-24:].mean()) if len(act_past) > 0 else np.float32(0.0)
            assert np.isclose(row["burden_last_6h"], exp_b6, atol=1e-6)
            assert np.isclose(row["burden_last_24h"], exp_b24, atol=1e-6)

            exp_adapt_7d = np.float32(ons_past[-168:].mean()) if len(ons_past) >= 24 else np.float32(0.05)
            exp_adapt_14d = np.float32(ons_past[-336:].mean()) if len(ons_past) >= 24 else np.float32(0.05)
            exp_adapt_30d = np.float32(ons_past[-720:].mean()) if len(ons_past) >= 24 else np.float32(0.05)

            assert np.isclose(row["adaptive_rate_7d"], exp_adapt_7d, atol=1e-6)
            assert np.isclose(row["adaptive_rate_14d"], exp_adapt_14d, atol=1e-6)
            assert np.isclose(row["adaptive_rate_30d"], exp_adapt_30d, atol=1e-6)

            exp_onsets_24h = int(ons_past[-24:].sum()) if len(ons_past) > 0 else 0
            exp_hours_since = hours_since_last_map[o]

            assert int(row["national_active_count"]) == exp_national_active
            assert int(row["onsets_last_24h"]) == exp_onsets_24h
            assert np.isclose(row["hours_since_last_alert"], exp_hours_since, atol=1e-5)

            nbrs = [n for n in neighbor_map[o] if n in active_hist]
            exp_nbr_count = sum(int(active_hist[n][-1]) for n in nbrs)
            exp_nbr_any = int(exp_nbr_count > 0)
            exp_nbr_ratio = np.float32(exp_nbr_count / len(nbrs)) if len(nbrs) > 0 else np.float32(0.0)

            exp_nbr_lag1 = sum(int(active_hist[n][-2]) for n in nbrs) if len(hist_ticks) >= 2 else 0
            exp_nbr_lag2 = sum(int(active_hist[n][-3]) for n in nbrs) if len(hist_ticks) >= 3 else 0
            exp_nbr_lag3 = sum(int(active_hist[n][-4]) for n in nbrs) if len(hist_ticks) >= 4 else 0

            assert int(row["neighbors_active_count"]) == exp_nbr_count
            assert int(row["neighbor_any_active"]) == exp_nbr_any
            assert np.isclose(row["neighbor_active_ratio"], exp_nbr_ratio, atol=1e-6)
            assert int(row["neighbors_active_lag_1h"]) == exp_nbr_lag1
            assert int(row["neighbors_active_lag_2h"]) == exp_nbr_lag2
            assert int(row["neighbors_active_lag_3h"]) == exp_nbr_lag3

            assert np.isclose(row["hour_sin"], exp_hour_sin, atol=1e-6)
            assert np.isclose(row["hour_cos"], exp_hour_cos, atol=1e-6)
            assert int(row["day_of_week"]) == exp_dow
            assert int(row["is_weekend"]) == exp_weekend
            assert int(row["month"]) == exp_month

            # Перевірка 3 one-hot колонок Train-only кластерів із FEATURE_SET_C
            exp_cid = expected_cluster_map[o]
            assert int(row["cluster_id"]) == exp_cid
            assert int(row["cluster_1_low_activity"]) == int(exp_cid == 1)
            assert int(row["cluster_2_medium_activity"]) == int(exp_cid == 2)
            assert int(row["cluster_3_high_activity"]) == int(exp_cid == 3)

            exp_target = int(o in oblasts_with_onset)
            assert int(row["target_onset_next_1h"]) == exp_target

            checked_rows += 1

    print(
        f"✔ [{label}] Causal Truncation & Target Invariance Test PASSED: "
        f"0 дублікатів, {len(sampled_timestamps)} моментів T ({len(boundary_timestamps)} граничних + "
        f"{len(random_timestamps)} випадкових, {checked_rows:,} рядків область-година; усі 22 ознаки Set C, Set D та таргет збігаються)."
    )
    return len(sampled_timestamps), checked_rows


def test_causal_truncation_invariance(n_random: int = 100):
    """Перевіряє каузальну інваріантність для основного датасету та для волонтерського датасету (naive_fixed_D)."""
    # 1. Основний офіційний датасет
    _run_truncation_checks_on_pair(
        PROJECT_ROOT / "data" / "processed" / "alerts_clean_oblast.parquet",
        PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet",
        PROJECT_ROOT / "reports" / "official" / "clusters_train_only.csv",
        label="Official Dataset",
        n_random=n_random,
    )

    # 2. Волонтерський датасет та каузальність naive_fixed_D
    vol_raw_csv = PROJECT_ROOT / "data" / "raw" / "volunteer_data_uk.csv"
    vol_as_is_pq = PROJECT_ROOT / "data" / "processed" / "volunteer_alerts_clean_as_is.parquet"
    vol_as_is_clusters_csv = PROJECT_ROOT / "reports" / "volunteer" / "volunteer_clusters_train_only.csv"
    vol_fixed_pq = PROJECT_ROOT / "data" / "processed" / "volunteer_alerts_clean_fixed_d.parquet"
    vol_fixed_fs = PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_fixed_d.parquet"
    vol_clusters_csv = PROJECT_ROOT / "reports" / "volunteer" / "volunteer_clusters_train_only_fixed_d.csv"

    if vol_as_is_pq.exists() and vol_as_is_clusters_csv.exists():
        _verify_train_only_clusters_invariance(
            vol_as_is_pq, vol_as_is_clusters_csv, label="Volunteer Dataset (naive_as_is)"
        )

    if vol_raw_csv.exists() and vol_fixed_pq.exists() and vol_fixed_fs.exists():
        raw_vol = pd.read_csv(vol_raw_csv)
        train_end = pd.Timestamp("2024-04-30 23:59:59", tz="UTC")

        # Перевірка, що D_oblast на повному CSV абсолютно ідентичний D_oblast на CSV, фізично обрізаному по train_end
        d_full = compute_train_median_durations_volunteer(raw_vol, train_end=train_end)

        raw_vol_trunc = raw_vol.copy()
        s_ts = pd.to_datetime(raw_vol_trunc["started_at"], utc=True)
        raw_vol_trunc = raw_vol_trunc[s_ts <= train_end].copy()
        f_ts_trunc = pd.to_datetime(raw_vol_trunc["finished_at"], utc=True).clip(upper=train_end)
        raw_vol_trunc["finished_at"] = f_ts_trunc.astype(str)
        d_trunc = compute_train_median_durations_volunteer(raw_vol_trunc, train_end=train_end)

        assert set(d_full.keys()) == set(d_trunc.keys())
        for k in d_full:
            assert np.isclose(d_full[k], d_trunc[k], atol=1e-9), (
                f"Leakage in D_oblast for {k}: {d_full[k]} vs {d_trunc[k]}"
            )

        fixed_df = pd.read_parquet(vol_fixed_pq)
        naive_rows = fixed_df[fixed_df["naive"]].copy()
        expected_d = naive_rows["oblast"].map(d_full).values
        assert np.allclose(naive_rows["duration_seconds"].values, expected_d, atol=1e-6), (
            "Mismatch between naive_fixed_D duration_seconds and Train D_oblast!"
        )
        print(
            f"✔ [Volunteer naive_fixed_D] Train-only Median Duration Invariance PASSED: "
            f"D_oblast для всіх {len(d_full)} регіонів не залежить від даних після {train_end}."
        )

        _run_truncation_checks_on_pair(
            vol_fixed_pq,
            vol_fixed_fs,
            vol_clusters_csv,
            label="Volunteer Dataset (naive_fixed_D)",
            n_random=n_random,
        )


def test_walk_forward_leakage_and_oof_invariance():
    """
    Перевіряє:
    1. Каузальність Walk-Forward сплітів на кількох тестових місяцях M (2-годинний embargo для інтервалу цілі [T, T+1h)).
    2. Інваріантність навчених моделей (LightGBM Set C, LogReg Set D) та порогів p*_M при псуванні всіх рядків >= T_start(M) - 1h.
    3. Цілісність і каузальність збережених на диск Out-of-Fold прогнозів (`walk_forward_oof_predictions_*.parquet`):
       - SHA-256 збігається з `walk_forward_details.json`;
       - 0 дублікатів за `(timestamp_utc, oblast)`, точний збіг із рядками `split_set == "test"` у Feature Store;
       - `y` точно дорівнює `target_onset_next_1h`, а `p_adapt` точно дорівнює `adaptive_rate_14d`;
       - `p_stat_lgb_c` точно відтворюється замороженою статичною моделлю `.joblib` на `FEATURE_SET_C`;
       - `p_roll_lgb_c` у збереженому OOF parquet для контрольного місяця M точно відтворюється навчанням
         на датасеті, фізично обрізаному до `T_start(M) - 2h` (без жодного рядка з місяця M чи майбутнього).
    """
    datasets = [
        (
            "Official",
            PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet",
            PROJECT_ROOT / "data" / "processed" / "walk_forward_oof_predictions_official.parquet",
            PROJECT_ROOT / "reports" / "official" / "walk_forward_details.json",
            PROJECT_ROOT / "models" / "lightgbm_adaptive.joblib",
            ["2025-06", "2025-12", "2026-08"],
            "2025-12",
        ),
        (
            "Volunteer (naive_fixed_D)",
            PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_fixed_d.parquet",
            PROJECT_ROOT / "data" / "processed" / "walk_forward_oof_predictions_volunteer_fixed_d.parquet",
            PROJECT_ROOT / "reports" / "volunteer" / "volunteer_walk_forward_details.json",
            PROJECT_ROOT / "models" / "volunteer_lightgbm_adaptive_fixed_d.joblib",
            ["2025-06", "2025-12", "2026-09"],
            "2025-12",
        ),
    ]

    for ds_label, pq_path, oof_path, details_path, frozen_lgb_path, check_months, oof_verify_month in datasets:
        assert pq_path.exists(), f"Missing {pq_path}"
        assert oof_path.exists(), f"Missing OOF parquet {oof_path}"
        assert details_path.exists(), f"Missing details JSON {details_path}"
        assert frozen_lgb_path.exists(), f"Missing frozen static model {frozen_lgb_path}"

        # Перевірка SHA-256 збереженого OOF parquet
        actual_sha = hashlib.sha256(oof_path.read_bytes()).hexdigest()
        details = json.loads(details_path.read_text(encoding="utf-8"))
        assert actual_sha == details.get("oof_predictions_sha256"), (
            f"[{ds_label}] SHA-256 mismatch between {oof_path.name} and {details_path.name}!"
        )

        df_orig = add_post_hoc_set_d_features(pd.read_parquet(pq_path))
        oof_df = pd.read_parquet(oof_path)

        # Перевірка відсутності дублікатів та точного збігу тестових рядків
        assert int(oof_df.duplicated(subset=["timestamp_utc", "oblast"]).sum()) == 0, (
            f"[{ds_label}] Duplicate (timestamp_utc, oblast) rows in {oof_path.name}!"
        )
        test_store = df_orig[df_orig["split_set"] == "test"].copy()
        test_store["year_month_utc"] = test_store["timestamp_utc"].dt.strftime("%Y-%m")
        test_store = test_store.sort_values(["year_month_utc", "timestamp_utc", "oblast"]).reset_index(drop=True)
        oof_sorted = oof_df.copy()
        oof_sorted["year_month_utc"] = pd.to_datetime(oof_sorted["timestamp_utc"], utc=True).dt.strftime("%Y-%m")
        oof_sorted = oof_sorted.sort_values(["year_month_utc", "timestamp_utc", "oblast"]).reset_index(drop=True)

        assert len(oof_sorted) == len(test_store), (
            f"[{ds_label}] Row count mismatch between OOF ({len(oof_sorted)}) and Test Store ({len(test_store)})!"
        )
        assert (oof_sorted["timestamp_utc"].values == test_store["timestamp_utc"].values).all()
        assert (oof_sorted["oblast"].values == test_store["oblast"].values).all()
        assert np.array_equal(oof_sorted["y"].values, test_store["target_onset_next_1h"].values)
        assert np.allclose(oof_sorted["p_adapt"].values, test_store["adaptive_rate_14d"].values, atol=1e-7)

        # Перевірка, що p_stat_lgb_c у збереженому OOF parquet відповідає замороженій статичній моделі .joblib
        frozen_lgb = joblib.load(frozen_lgb_path)
        expected_stat_p = frozen_lgb.predict_proba(test_store[FEATURE_SET_C])[:, 1]
        assert np.allclose(oof_sorted["p_stat_lgb_c"].values, expected_stat_p, atol=1e-12), (
            f"[{ds_label}] Saved p_stat_lgb_c does not match frozen {frozen_lgb_path.name} predictions!"
        )

        for m_str in check_months:
            m_start = pd.Timestamp(f"{m_str}-01 00:00:00", tz="UTC")
            m_end = m_start + pd.DateOffset(months=1)

            for win_mode in ["rolling_12m", "expanding"]:
                tr_mask, va_mask, te_mask, bounds = get_walk_forward_split_masks(
                    df_orig["timestamp_utc"], m_start, m_end, window_mode=win_mode, embargo_hours=2
                )
                tr_df = df_orig.loc[tr_mask]
                va_df = df_orig.loc[va_mask]
                te_df = df_orig.loc[te_mask]

                # 1. Суворі нерівності часових міток з 2-годинним embargo для [T, T+1h)
                assert len(tr_df) > 0 and len(va_df) > 0 and len(te_df) > 0
                assert tr_df["timestamp_utc"].max() <= bounds["val_start"] - pd.Timedelta(hours=2), (
                    f"[{ds_label} {m_str} {win_mode}] Train_M leaks into Val_M embargo!"
                )
                assert va_df["timestamp_utc"].min() >= bounds["val_start"]
                assert va_df["timestamp_utc"].max() <= m_start - pd.Timedelta(hours=2), (
                    f"[{ds_label} {m_str} {win_mode}] Val_M leaks into Test month M embargo!"
                )
                assert te_df["timestamp_utc"].min() >= m_start
                assert te_df["timestamp_utc"].max() < m_end

            # 2. Біт-в-біт інваріантність відносно мутації всіх рядків >= m_start - 1h (n_jobs=1, seed=42)
            tr_mask, va_mask, _, _ = get_walk_forward_split_masks(
                df_orig["timestamp_utc"], m_start, m_end, window_mode="rolling_12m", embargo_hours=2
            )
            tr_1 = df_orig.loc[tr_mask]
            va_1 = df_orig.loc[va_mask]
            y_tr_1 = tr_1["target_onset_next_1h"].values
            y_va_1 = va_1["target_onset_next_1h"].values

            th_adapt_1 = fast_threshold_search_f2(y_va_1, va_1["adaptive_rate_14d"].values)["threshold"]
            lr_1, sc_1, th_lr_1, p_va_lr_1 = fit_fold_logreg(tr_1, y_tr_1, va_1, y_va_1, FEATURE_SET_D_POST_HOC, random_state=42)
            lgb_1, th_lgb_1, p_va_lgb_1 = fit_fold_lightgbm(tr_1, y_tr_1, va_1, y_va_1, FEATURE_SET_C, n_jobs=1, random_state=42)

            # Створюємо зіпсовану копію датасету, де все починаючи з (m_start - 1h) інвертовано/зашумлено
            df_mut = df_orig.copy()
            corrupt_mask = df_mut["timestamp_utc"] >= (m_start - pd.Timedelta(hours=1))
            df_mut.loc[corrupt_mask, "target_onset_next_1h"] = 1 - df_mut.loc[corrupt_mask, "target_onset_next_1h"]
            for fcol in FEATURE_SET_D_POST_HOC:
                df_mut.loc[corrupt_mask, fcol] = np.array(-99, dtype=df_mut[fcol].dtype)

            tr_mask_2, va_mask_2, _, _ = get_walk_forward_split_masks(
                df_mut["timestamp_utc"], m_start, m_end, window_mode="rolling_12m", embargo_hours=2
            )
            tr_2 = df_mut.loc[tr_mask_2]
            va_2 = df_mut.loc[va_mask_2]
            y_tr_2 = tr_2["target_onset_next_1h"].values
            y_va_2 = va_2["target_onset_next_1h"].values

            th_adapt_2 = fast_threshold_search_f2(y_va_2, va_2["adaptive_rate_14d"].values)["threshold"]
            lr_2, sc_2, th_lr_2, p_va_lr_2 = fit_fold_logreg(tr_2, y_tr_2, va_2, y_va_2, FEATURE_SET_D_POST_HOC, random_state=42)
            lgb_2, th_lgb_2, p_va_lgb_2 = fit_fold_lightgbm(tr_2, y_tr_2, va_2, y_va_2, FEATURE_SET_C, n_jobs=1, random_state=42)

            assert th_adapt_1 == th_adapt_2, f"Adaptive WF threshold leaked from month {m_str}!"
            assert th_lr_1 == th_lr_2, f"LogReg WF threshold leaked from month {m_str}!"
            assert th_lgb_1 == th_lgb_2, f"LightGBM WF threshold leaked from month {m_str}!"
            assert np.array_equal(p_va_lr_1, p_va_lr_2), f"LogReg Val predictions not bit-for-bit identical on {m_str}!"
            assert np.array_equal(p_va_lgb_1, p_va_lgb_2), f"LightGBM Val predictions not bit-for-bit identical on {m_str}!"
            assert np.array_equal(lr_1.coef_, lr_2.coef_), f"LogReg weights not bit-for-bit identical on {m_str}!"

            # 3. Перевірка, що збережений у OOF parquet p_roll_lgb_c для місяця oof_verify_month
            # точно відтворюється моделлю, навченою на фізично обрізаному до (m_start - 2h) датасеті
            if m_str == oof_verify_month:
                df_past_only = df_orig[df_orig["timestamp_utc"] <= (m_start - pd.Timedelta(hours=2))].copy()
                tr_p_mask, va_p_mask, _, _ = get_walk_forward_split_masks(
                    df_past_only["timestamp_utc"], m_start, m_end, window_mode="rolling_12m", embargo_hours=2
                )
                lgb_past, _, _ = fit_fold_lightgbm(
                    df_past_only.loc[tr_p_mask],
                    df_past_only.loc[tr_p_mask, "target_onset_next_1h"].values,
                    df_past_only.loc[va_p_mask],
                    df_past_only.loc[va_p_mask, "target_onset_next_1h"].values,
                    FEATURE_SET_C,
                    n_jobs=4,
                    random_state=42,
                )
                te_m = test_store[test_store["year_month_utc"] == m_str]
                oof_m = oof_sorted[oof_sorted["year_month_utc"] == m_str]
                pred_m_from_past = lgb_past.predict_proba(te_m[FEATURE_SET_C])[:, 1]
                assert np.allclose(oof_m["p_roll_lgb_c"].values, pred_m_from_past, atol=1e-6), (
                    f"[{ds_label} {m_str}] Saved OOF p_roll_lgb_c differs from model trained on strictly truncated past!"
                )

        print(
            f"✔ [{ds_label}] Walk-Forward & Saved OOF Predictions Invariance PASSED "
            f"(SHA-256 verified, {len(oof_sorted):,} OOF rows, frozen .joblib & truncated {oof_verify_month} fold verified)."
        )


if __name__ == "__main__":
    test_causal_truncation_invariance()
    test_walk_forward_leakage_and_oof_invariance()
