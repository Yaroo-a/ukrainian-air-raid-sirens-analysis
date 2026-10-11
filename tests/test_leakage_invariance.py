"""Тест каузальної інваріантності ознак A/B/C та цільової змінної (Causal Truncation & Target Test).

Перевіряє для вибірки часових точок T (>= 100 моментів T: випадкові зрізи з Train, Val, Test
плюс граничні випадки k=0, 1, 2, 3, 23, 24, 168, 336, 720 та точки поблизу меж сплітів):
1. Відсутність дублікатів у alerts_clean_oblast.parquet та ml_features_hourly.parquet (assert 0).
2. Інваріантність усіх ознак наборів A, B та C при фізичному обрізанні майбутнього на момент T:
   - Залишаються лише тривоги, що почалися до або в момент T (started_at <= T).
   - Для тривог, що тривають на момент T (finished_at > T), час завершення затирається (вважаються відкритими).
   - Незалежно перераховуються всі ознаки:
     * фільтр спокою (is_active_T == 0)
     * lagged_onset_1h
     * burden_last_6h, burden_last_24h
     * adaptive_rate_7d, adaptive_rate_14d, adaptive_rate_30d
     * national_active_count
     * onsets_last_24h (Варіант А: сума стартів на годинній сітці за [T-24h, T))
     * hours_since_last_alert (час у годинах від останнього відомого finished_at <= T, clip 720.0)
     * neighbors_active_count, neighbor_any_active, neighbor_active_ratio
     * neighbors_active_lag_1h, neighbors_active_lag_2h, neighbors_active_lag_3h
     * hour_sin, hour_cos, day_of_week, is_weekend, month.
   - Перевіряється збіг із збереженим Feature Store (atol=1e-6).
3. Залежність таргету виключно від вікна [T, T + 1h).

Обмеження тесту: тест працює поверх очищених інтервалів alerts_clean_oblast.parquet, тому він
перевіряє каузальність генератора ознак feature_store.py, але не бачить етапу первинного очищення
в cleaner.py (відсіювання мікро-тривог < 60 с за finished_at - started_at та злиття інтервалів).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.feature_store import build_neighbor_dict
from src.models.train_models import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_C_PRIME,
)


def test_causal_truncation_invariance(n_random: int = 100):
    """Перевіряє нуль дублікатів та всі ознаки A/B/C/C' і таргет на >= 100 часових зрізах T (включно з граничними)."""
    parquet_path = PROJECT_ROOT / "data" / "processed" / "alerts_clean_oblast.parquet"
    feature_store_path = PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet"

    assert parquet_path.exists(), f"Missing {parquet_path}"
    assert feature_store_path.exists(), f"Missing {feature_store_path}"

    raw_df = pd.read_parquet(parquet_path)
    store_df = pd.read_parquet(feature_store_path)

    # Перевірка на нуль дублікатів
    clean_dups = int(raw_df.duplicated(subset=["oblast", "started_at", "finished_at"]).sum())
    store_dups = int(store_df.duplicated(subset=["timestamp_utc", "oblast"]).sum())
    assert clean_dups == 0, f"Found {clean_dups} duplicates in alerts_clean_oblast.parquet!"
    assert store_dups == 0, f"Found {store_dups} duplicates in ml_features_hourly.parquet!"

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

    # Граничні випадки: старт сітки (k=0, 1, 2, 3), межі вікон (k=23, 24, 168, 336, 720) та межі сплітів
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
        # --- ЧАСТИНА 1: ОБРІЗАННЯ МАЙБУТНЬОГО НА МОМЕНТ T ---
        past_only = raw_clean[raw_clean["started_at"] <= T].copy()
        # Тривоги, що завершуються після T, вважаємо відкритими на момент T (finished_at невідомий = NaT)
        far_future = pd.Timestamp("2099-01-01", tz="UTC")
        past_only["finished_known"] = past_only["finished_at"].where(
            past_only["finished_at"].notna() & (past_only["finished_at"] <= T),
            other=pd.NaT,
        )
        past_only["finished_censored"] = past_only["finished_known"].fillna(far_future)

        # Побудова локальної історії за останні 720 годин [T - 720h, T] (достатньо для 30-денного вікна)
        n_hist = min(720, int((T - min_global_time).total_seconds() // 3600))
        hist_ticks = pd.date_range(end=T, periods=n_hist + 1, freq="1h", tz="UTC")
        # hist_ticks[-1] == T, hist_ticks[:-1] це [T - n_hist*1h, ..., T - 1h]

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

            # Час від кінця останньої тривоги (використовує ВИКЛЮЧНО відомі finished_known <= T)
            known_ends = o_alerts["finished_known"].dropna()
            if len(known_ends) > 0:
                last_end = known_ends.max()
                elapsed = (T - last_end).total_seconds() / 3600.0
                hours_since_last_map[o] = np.float32(np.clip(elapsed, 0.0, 720.0))
            else:
                hours_since_last_map[o] = np.float32(720.0)

        # Загальнонаціональна кількість активних областей у момент T
        exp_national_active = sum(int(active_hist[o][-1]) for o in oblasts)

        # Перевірка фільтра спокою: у store_df мають бути присутні рівно ті області, де active_hist[o][-1] == 0
        store_slice = store_df[store_df["timestamp_utc"] == T]
        calm_oblasts_expected = {o for o in oblasts if active_hist[o][-1] == 0}
        calm_oblasts_actual = set(store_slice["oblast"].unique())
        assert calm_oblasts_actual == calm_oblasts_expected, (
            f"Calm filter mismatch at T={T}: actual={calm_oblasts_actual ^ calm_oblasts_expected}"
        )

        # Календарні ознаки на момент T (Europe/Kyiv)
        t_kyiv = T.tz_convert("Europe/Kyiv")
        exp_hour_sin = np.float32(np.sin(2 * np.pi * t_kyiv.hour / 24.0))
        exp_hour_cos = np.float32(np.cos(2 * np.pi * t_kyiv.hour / 24.0))
        exp_dow = int(t_kyiv.dayofweek)
        exp_weekend = int(exp_dow >= 5)
        exp_month = int(t_kyiv.month)

        # --- ЧАСТИНА 2: ПЕРЕВІРКА ТАРГЕТУ ЛИШЕ ЗА ПОДІЯМИ В [T, T + 1h) ---
        T_next = T + pd.Timedelta(hours=1)
        future_window_only = raw_clean[
            (raw_clean["started_at"] >= T) & (raw_clean["started_at"] < T_next)
        ]
        oblasts_with_onset = set(future_window_only["oblast"].unique())

        for _, row in store_slice.iterrows():
            o = row["oblast"]
            act_past = active_hist[o][:-1]  # strictly before T: [T - n_hist, ..., T - 1h]
            ons_past = onset_hist[o]        # strictly before T: windows [tau, tau + 1h) up to [T - 1h, T)

            # 1. lagged_onset_1h: чи був старт у [T - 1h, T)
            exp_lag_1h = int(ons_past[-1]) if len(ons_past) > 0 else 0
            assert int(row["lagged_onset_1h"]) == exp_lag_1h, (
                f"lagged_onset_1h mismatch at T={T}, {o}: {row['lagged_onset_1h']} vs {exp_lag_1h}"
            )

            # 2. burden_last_6h та burden_last_24h
            exp_b6 = np.float32(act_past[-6:].mean()) if len(act_past) > 0 else np.float32(0.0)
            exp_b24 = np.float32(act_past[-24:].mean()) if len(act_past) > 0 else np.float32(0.0)
            assert np.isclose(row["burden_last_6h"], exp_b6, atol=1e-6), (
                f"burden_last_6h mismatch at T={T}, {o}: {row['burden_last_6h']} vs {exp_b6}"
            )
            assert np.isclose(row["burden_last_24h"], exp_b24, atol=1e-6), (
                f"burden_last_24h mismatch at T={T}, {o}: {row['burden_last_24h']} vs {exp_b24}"
            )

            # 3. Ковзні частоти стартів adaptive_rate_7d (168h), 14d (336h), 30d (720h)
            exp_adapt_7d = np.float32(ons_past[-168:].mean()) if len(ons_past) >= 24 else np.float32(0.05)
            exp_adapt_14d = np.float32(ons_past[-336:].mean()) if len(ons_past) >= 24 else np.float32(0.05)
            exp_adapt_30d = np.float32(ons_past[-720:].mean()) if len(ons_past) >= 24 else np.float32(0.05)

            assert np.isclose(row["adaptive_rate_7d"], exp_adapt_7d, atol=1e-6), (
                f"adaptive_rate_7d mismatch at T={T}, {o}: {row['adaptive_rate_7d']} vs {exp_adapt_7d}"
            )
            assert np.isclose(row["adaptive_rate_14d"], exp_adapt_14d, atol=1e-6), (
                f"adaptive_rate_14d mismatch at T={T}, {o}: {row['adaptive_rate_14d']} vs {exp_adapt_14d}"
            )
            assert np.isclose(row["adaptive_rate_30d"], exp_adapt_30d, atol=1e-6), (
                f"adaptive_rate_30d mismatch at T={T}, {o}: {row['adaptive_rate_30d']} vs {exp_adapt_30d}"
            )

            # 4. Нові локальні та національні ознаки Set C: national_active_count, onsets_last_24h, hours_since_last_alert
            exp_onsets_24h = int(ons_past[-24:].sum()) if len(ons_past) > 0 else 0
            exp_hours_since = hours_since_last_map[o]

            assert int(row["national_active_count"]) == exp_national_active, (
                f"national_active_count mismatch at T={T}, {o}: {row['national_active_count']} vs {exp_national_active}"
            )
            assert int(row["onsets_last_24h"]) == exp_onsets_24h, (
                f"onsets_last_24h mismatch at T={T}, {o}: {row['onsets_last_24h']} vs {exp_onsets_24h}"
            )
            assert np.isclose(row["hours_since_last_alert"], exp_hours_since, atol=1e-5), (
                f"hours_since_last_alert mismatch at T={T}, {o}: {row['hours_since_last_alert']} vs {exp_hours_since}"
            )

            # 5. Просторові ознаки сусідів на момент T та лаги T-1h, T-2h, T-3h
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
            assert int(row["neighbors_active_lag_1h"]) == exp_nbr_lag1, (
                f"neighbors_active_lag_1h mismatch at T={T}, {o}: {row['neighbors_active_lag_1h']} vs {exp_nbr_lag1}"
            )
            assert int(row["neighbors_active_lag_2h"]) == exp_nbr_lag2, (
                f"neighbors_active_lag_2h mismatch at T={T}, {o}: {row['neighbors_active_lag_2h']} vs {exp_nbr_lag2}"
            )
            assert int(row["neighbors_active_lag_3h"]) == exp_nbr_lag3, (
                f"neighbors_active_lag_3h mismatch at T={T}, {o}: {row['neighbors_active_lag_3h']} vs {exp_nbr_lag3}"
            )

            # 6. Календарні ознаки
            assert np.isclose(row["hour_sin"], exp_hour_sin, atol=1e-6)
            assert np.isclose(row["hour_cos"], exp_hour_cos, atol=1e-6)
            assert int(row["day_of_week"]) == exp_dow
            assert int(row["is_weekend"]) == exp_weekend
            assert int(row["month"]) == exp_month

            # 7. Таргет: залежить виключно від стартів у [T, T + 1h)
            exp_target = int(o in oblasts_with_onset)
            assert int(row["target_onset_next_1h"]) == exp_target, (
                f"target_onset_next_1h mismatch at T={T}, {o}: {row['target_onset_next_1h']} vs {exp_target}"
            )

            checked_rows += 1

    print(
        f"✔ Causal Truncation & Target Invariance Test PASSED: "
        f"перевірено 0 дублікатів та {len(sampled_timestamps)} моментів T ({len(boundary_timestamps)} граничних + "
        f"{len(random_timestamps)} випадкових, {checked_rows:,} рядків область-година; усі ознаки A/B/C та таргет збігаються)."
    )


if __name__ == "__main__":
    test_causal_truncation_invariance()
