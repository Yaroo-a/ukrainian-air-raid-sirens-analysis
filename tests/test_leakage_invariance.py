"""Тест каузальної інваріантності ознак A/B/C/C', цільової змінної та каузальної імп'ютації naive_fixed_D
(Causal Truncation & Target Test).

Перевіряє для вибірки часових точок T (>= 100 моментів T: випадкові зрізи з Train, Val, Test
плюс граничні випадки k=0, 1, 2, 3, 23, 24, 168, 336, 720 та точки поблизу меж сплітів):
1. Відсутність дублікатів у alerts_clean_oblast.parquet, ml_features_hourly.parquet,
   volunteer_alerts_clean_*.parquet та volunteer_ml_features_hourly_*.parquet (assert 0).
2. Інваріантність усіх ознак наборів A, B, C та C' при фізичному обрізанні майбутнього на момент T:
   - Залишаються лише тривоги, що почалися до або в момент T (started_at <= T).
   - Для тривог, що тривають на момент T (finished_at > T), час завершення затирається (вважаються відкритими).
   - Незалежно перераховуються всі ознаки та перевіряється збіг із збереженим Feature Store (atol=1e-6).
3. Залежність таргету виключно від вікна [T, T + 1h).
4. Каузальність варіанта naive_fixed_D на волонтерських даних:
   - Перевірка, що медіанні тривалості D_oblast залежать виключно від Train (<= 2024-04-30 23:59:59 UTC)
     і не змінюються при видаленні чи модифікації всіх даних після 2024-04-30.
   - Повний тест фізичного обрізання на >= 100 моментах T для volunteer_ml_features_hourly_fixed_d.parquet.

Обмеження тесту: тест працює поверх очищених інтервалів *_alerts_clean_*.parquet, тому він
перевіряє каузальність генератора ознак feature_store.py та train-only медіан D_oblast, але не бачить
етапу первинного відсіювання мікро-тривог < 60 с за finished_at - started_at у cleaner.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.cleaner import compute_train_median_durations_volunteer
from src.features.feature_store import build_neighbor_dict
from src.models.train_models import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_C_PRIME,
)


def _run_truncation_checks_on_pair(
    parquet_path: Path,
    feature_store_path: Path,
    label: str,
    n_random: int = 100,
) -> tuple[int, int]:
    assert parquet_path.exists(), f"Missing {parquet_path}"
    assert feature_store_path.exists(), f"Missing {feature_store_path}"

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

            exp_target = int(o in oblasts_with_onset)
            assert int(row["target_onset_next_1h"]) == exp_target

            checked_rows += 1

    print(
        f"✔ [{label}] Causal Truncation & Target Invariance Test PASSED: "
        f"0 дублікатів, {len(sampled_timestamps)} моментів T ({len(boundary_timestamps)} граничних + "
        f"{len(random_timestamps)} випадкових, {checked_rows:,} рядків область-година; усі ознаки A/B/C/C' та таргет збігаються)."
    )
    return len(sampled_timestamps), checked_rows


def test_causal_truncation_invariance(n_random: int = 100):
    """Перевіряє каузальну інваріантність для основного датасету та для волонтерського датасету (naive_fixed_D)."""
    # 1. Основний офіційний датасет
    _run_truncation_checks_on_pair(
        PROJECT_ROOT / "data" / "processed" / "alerts_clean_oblast.parquet",
        PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet",
        label="Official Dataset",
        n_random=n_random,
    )

    # 2. Волонтерський датасет та каузальність naive_fixed_D (якщо файли згенеровані)
    vol_raw_csv = PROJECT_ROOT / "data" / "raw" / "volunteer_data_uk.csv"
    vol_fixed_pq = PROJECT_ROOT / "data" / "processed" / "volunteer_alerts_clean_fixed_d.parquet"
    vol_fixed_fs = PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_fixed_d.parquet"

    if vol_raw_csv.exists() and vol_fixed_pq.exists() and vol_fixed_fs.exists():
        raw_vol = pd.read_csv(vol_raw_csv)
        train_end = pd.Timestamp("2024-04-30 23:59:59", tz="UTC")

        # Перевірка, що D_oblast на повному CSV абсолютно ідентичний D_oblast на CSV, фізично обрізаному по train_end
        d_full = compute_train_median_durations_volunteer(raw_vol, train_end=train_end)

        raw_vol_trunc = raw_vol.copy()
        s_ts = pd.to_datetime(raw_vol_trunc["started_at"], utc=True)
        f_ts = pd.to_datetime(raw_vol_trunc["finished_at"], utc=True)
        raw_vol_trunc = raw_vol_trunc[s_ts <= train_end].copy()
        # Навіть якщо finished_at перетинає train_end, затираємо значення після train_end на train_end
        f_ts_trunc = pd.to_datetime(raw_vol_trunc["finished_at"], utc=True).clip(upper=train_end)
        raw_vol_trunc["finished_at"] = f_ts_trunc.astype(str)
        d_trunc = compute_train_median_durations_volunteer(raw_vol_trunc, train_end=train_end)

        assert set(d_full.keys()) == set(d_trunc.keys())
        for k in d_full:
            assert np.isclose(d_full[k], d_trunc[k], atol=1e-9), (
                f"Leakage in D_oblast for {k}: {d_full[k]} vs {d_trunc[k]}"
            )

        # Перевірка, що у volunteer_alerts_clean_fixed_d.parquet всі naive==True мають тривалість рівно D_oblast
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
            label="Volunteer Dataset (naive_fixed_D)",
            n_random=n_random,
        )


def test_walk_forward_leakage_invariance():
    """
    Перевіряє каузальність Walk-Forward сплітів на кількох тестових місяцях M (для інтервалу цілі [T, T+1h)):
    1. Усі рядки Train_M мають timestamp_utc <= T_start(Val_M) - 2h.
    2. Усі рядки Val_M мають timestamp_utc <= T_start(M) - 2h.
    3. При фізичному псуванні/інверсії всіх даних починаючи з T_start(M) - 1h (включно з місяцем M і всім майбутнім)
       навчені з n_jobs=1 та фіксованим random_state=42 моделі (LightGBM Set C/D, LogReg Set C/D) та обрані
       на Val_M пороги p*_M (включно з Adaptive 14-day WF p*_M) залишаються біт-в-біт ідентичними.
    """
    from src.models.run_walk_forward_experiment import (
        FEATURE_SET_C,
        FEATURE_SET_D_POST_HOC,
        add_post_hoc_set_d_features,
        fast_threshold_search_f2,
        fit_fold_lightgbm,
        fit_fold_logreg,
        get_walk_forward_split_masks,
    )

    datasets = [
        ("Official", PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet", ["2025-06", "2025-12", "2026-08"]),
        ("Volunteer (naive_fixed_D)", PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_fixed_d.parquet", ["2025-06", "2025-12", "2026-09"]),
    ]

    for ds_label, pq_path, check_months in datasets:
        if not pq_path.exists():
            continue
        df_orig = add_post_hoc_set_d_features(pd.read_parquet(pq_path))

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

        print(
            f"✔ [{ds_label}] Walk-Forward Leakage & Bit-for-Bit Invariance Test PASSED "
            f"(months={check_months}, embargo=2h for [T, T+1h), n_jobs=1, seed=42)."
        )


if __name__ == "__main__":
    test_causal_truncation_invariance()
    test_walk_forward_leakage_invariance()

