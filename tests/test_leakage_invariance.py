"""Тест каузальної інваріантності ознак та цільової змінної (Causal Truncation & Target Test).

Перевіряє для випадкової вибірки часових точок T (N=25 з Train, Val, Test):
1. Інваріантність ознак при обрізанні майбутнього:
   - Залишаються лише тривоги, що почалися до або в момент T (started_at <= T).
   - Для тривог, що тривають на момент T (finished_at > T), час завершення затирається (вважаються відкритими).
   - Незалежно перераховуються всі часові, авторегресійні, адаптивні та просторові ознаки:
     is_active_T (фільтр спокою), lagged_onset_1h, burden_last_6h, burden_last_24h,
     adaptive_rate_14d, neighbors_active_count, neighbor_any_active, neighbor_active_ratio,
     hour_sin, hour_cos, day_of_week, is_weekend, month.
   - Перевіряється збіг із збереженим Feature Store.
2. Залежність таргету виключно від вікна [T, T + 1h):
   - Перевіряється, що target_onset_next_1h визначається виключно стартами тривог
     у піввідкритому інтервалі [T, T + 1h) і не залежить від подій до T або після T + 1h.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.feature_store import build_neighbor_dict


def test_causal_truncation_invariance(n_samples: int = 25):
    """Перевіряє всі ознаки та таргет на випадкових часових зрізах T."""
    parquet_path = PROJECT_ROOT / "data" / "processed" / "alerts_clean_oblast.parquet"
    feature_store_path = PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet"

    assert parquet_path.exists(), f"Missing {parquet_path}"
    assert feature_store_path.exists(), f"Missing {feature_store_path}"

    raw_df = pd.read_parquet(parquet_path)
    raw_clean = raw_df[~raw_df["is_occupied_extreme"]].copy()
    store_df = pd.read_parquet(feature_store_path)

    min_global_time = raw_clean["started_at"].min().floor("h")

    # Вибираємо N випадкових моментів T (після перших 15 діб, щоб вікно 336 год було повним, плюс кілька ранніх)
    all_timestamps = store_df["timestamp_utc"].drop_duplicates().sort_values()
    valid_timestamps = all_timestamps[all_timestamps >= (min_global_time + pd.Timedelta(hours=25))]
    sampled_timestamps = valid_timestamps.sample(n_samples, random_state=42).tolist()

    oblasts = sorted(raw_clean["oblast"].unique())
    neighbor_map = build_neighbor_dict(oblasts)

    checked_rows = 0

    for T in sampled_timestamps:
        # --- ЧАСТИНА 1: ОБРІЗАННЯ МАЙБУТНЬОГО НА МОМЕНТ T ---
        past_only = raw_clean[raw_clean["started_at"] <= T].copy()
        # Тривоги, що завершуються після T, вважаємо відкритими на момент T (finished_at невідомий)
        far_future = pd.Timestamp("2099-01-01", tz="UTC")
        past_only["finished_censored"] = past_only["finished_at"].where(
            past_only["finished_at"].notna() & (past_only["finished_at"] <= T),
            other=far_future,
        )

        # Побудова локальної історії за останні 336 годин [T - 336h, T]
        n_hist = min(336, int((T - min_global_time).total_seconds() // 3600))
        hist_ticks = pd.date_range(end=T, periods=n_hist + 1, freq="1h", tz="UTC")
        # hist_ticks[-1] == T, hist_ticks[:-1] це [T - n_hist*1h, ..., T - 1h]

        active_hist = {}
        onset_hist = {}

        for o in oblasts:
            o_alerts = past_only[past_only["oblast"] == o]
            s_arr = o_alerts["started_at"].values
            e_arr = o_alerts["finished_censored"].values

            # Активність у кожній годині tau з hist_ticks: started_at <= tau < finished_censored
            ticks_val = hist_ticks.values[:, None]
            if len(s_arr) > 0:
                act_vec = ((s_arr[None, :] <= ticks_val) & (ticks_val < e_arr[None, :])).any(axis=1).astype(int)
                # Старти у вікнах [tau, tau + 1h) для tau < T (тобто hist_ticks[:-1])
                prev_ticks = hist_ticks[:-1].values[:, None]
                next_ticks = (hist_ticks[:-1] + pd.Timedelta(hours=1)).values[:, None]
                # Зауважимо: для останнього інтервалу [T-1h, T) тривога точно має started_at < T <= T
                ons_vec = ((s_arr[None, :] >= prev_ticks) & (s_arr[None, :] < next_ticks)).any(axis=1).astype(int)
            else:
                act_vec = np.zeros(len(hist_ticks), dtype=int)
                ons_vec = np.zeros(len(hist_ticks) - 1, dtype=int)

            active_hist[o] = act_vec
            onset_hist[o] = ons_vec

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

            # 3. adaptive_rate_14d (вікно 336 годин до T)
            exp_adapt = np.float32(ons_past[-336:].mean()) if len(ons_past) >= 24 else np.float32(0.05)
            assert np.isclose(row["adaptive_rate_14d"], exp_adapt, atol=1e-6), (
                f"adaptive_rate_14d mismatch at T={T}, {o}: {row['adaptive_rate_14d']} vs {exp_adapt}"
            )

            # 4. Просторові ознаки сусідів на момент T
            nbrs = [n for n in neighbor_map[o] if n in active_hist]
            exp_nbr_count = sum(int(active_hist[n][-1]) for n in nbrs)
            exp_nbr_any = int(exp_nbr_count > 0)
            exp_nbr_ratio = np.float32(exp_nbr_count / len(nbrs)) if len(nbrs) > 0 else np.float32(0.0)

            assert int(row["neighbors_active_count"]) == exp_nbr_count, (
                f"neighbors_active_count mismatch at T={T}, {o}"
            )
            assert int(row["neighbor_any_active"]) == exp_nbr_any, (
                f"neighbor_any_active mismatch at T={T}, {o}"
            )
            assert np.isclose(row["neighbor_active_ratio"], exp_nbr_ratio, atol=1e-6), (
                f"neighbor_active_ratio mismatch at T={T}, {o}"
            )

            # 5. Календарні ознаки
            assert np.isclose(row["hour_sin"], exp_hour_sin, atol=1e-6)
            assert np.isclose(row["hour_cos"], exp_hour_cos, atol=1e-6)
            assert int(row["day_of_week"]) == exp_dow
            assert int(row["is_weekend"]) == exp_weekend
            assert int(row["month"]) == exp_month

            # 6. Таргет: залежить виключно від стартів у [T, T + 1h)
            exp_target = int(o in oblasts_with_onset)
            assert int(row["target_onset_next_1h"]) == exp_target, (
                f"target_onset_next_1h mismatch at T={T}, {o}: {row['target_onset_next_1h']} vs {exp_target}"
            )

            checked_rows += 1

    print(
        f"✔ Causal Truncation & Target Invariance Test PASSED: "
        f"перевірено {n_samples} випадкових моментів T ({checked_rows} рядків, усі ознаки та таргет збігаються)."
    )


if __name__ == "__main__":
    test_causal_truncation_invariance()
