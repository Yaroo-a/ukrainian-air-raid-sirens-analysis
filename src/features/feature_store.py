"""Feature Store Generator for Machine Learning Forecasting (Phase 2.1).

Transforms historical alert events into a regular hourly panel dataset (X, y)
for 24 Ukrainian regions with strict Zero Data Leakage:
1. Precise decision timestamp T (e.g. 14:00:00 UTC).
2. Conditioning on calm periods: filtered strictly to rows where alert is NOT active at T (is_active_T == 0).
3. Target variable y_T: Onset of a new alert in the next hour interval (T, T + 60 min].
4. Cyclical temporal transforms: hour_sin, hour_cos, day_of_week, is_weekend, month at moment T.
5. Strictly backward-looking autoregressive features: burden_last_6h, burden_last_24h before moment T.
6. Spatial neighbor features at exact moment T: neighbors_active_count, neighbor_any_active, neighbor_active_ratio.
7. Regional clusters: read strictly from reports/clusters_train_only.csv (fitted on Train only).
8. Purging & Embargo: 2-hour buffer at Train/Val and Val/Test boundaries.
9. Output: data/processed/ml_features_hourly.parquet.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Land border adjacency mapping for neighbors
from src.features.propagation import ADJACENCY_PAIRS


def build_neighbor_dict(regions: list[str]) -> dict[str, list[str]]:
    """Builds a bidirectional adjacency lookup for each region."""
    neighbor_map = {r: [] for r in regions}
    for ra, rb in ADJACENCY_PAIRS:
        if ra in neighbor_map and rb in neighbor_map:
            if rb not in neighbor_map[ra]:
                neighbor_map[ra].append(rb)
            if ra not in neighbor_map[rb]:
                neighbor_map[rb].append(ra)
    return neighbor_map


def generate_feature_store(
    parquet_path: str | Path = "data/processed/alerts_clean_oblast.parquet",
    clusters_csv_path: str | Path = "reports/clusters_train_only.csv",
    output_parquet_path: str | Path = "data/processed/ml_features_hourly.parquet",
) -> pd.DataFrame:
    """Constructs the calm-period hourly ML feature matrix with strict causal lags."""
    print("🚀 [Фаза 2.1] Початок генерації Feature Store (строго для періодів тиші)...")
    input_file = Path(parquet_path)
    clusters_file = Path(clusters_csv_path)
    output_file = Path(output_parquet_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load alerts (excluding extreme occupied territories only)
    print(f"📖 Завантаження даних з {input_file}...")
    df = pd.read_parquet(input_file)
    clean_df = df[~df["is_occupied_extreme"]].copy()

    # Load cluster labels strictly fitted on Train
    print(f"📍 Читання кластерів строго з Train: {clusters_file}...")
    clusters_df = pd.read_csv(clusters_file, index_col="oblast")
    cluster_map = clusters_df["cluster"].to_dict()

    oblasts = sorted(clean_df["oblast"].unique())
    neighbor_map = build_neighbor_dict(oblasts)
    print(f"   Регіонів у вибірці: {len(oblasts)}")

    # 2. Hourly time grid: decision moments T
    min_time = clean_df["started_at"].min().floor("h")
    max_time = clean_df["started_at"].max().ceil("h")
    # Time grid T_k with 1 hour step
    hourly_index = pd.date_range(start=min_time, end=max_time - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    n_hours = len(hourly_index)
    print(f"⏱️ Побудова годинної сітки моментів прогнозу T: {n_hours:,} точок ({min_time.date()} — {max_time.date()})...")

    # 3. Create high-speed lookup matrices for each oblast
    # active_at_T[k, o]: is alert active at exact timestamp T_k?
    # onset_next_1h[k, o]: did a new alert START in (T_k, T_k + 1h]?
    active_at_T = np.zeros((n_hours, len(oblasts)), dtype=np.int8)
    onset_next_1h = np.zeros((n_hours, len(oblasts)), dtype=np.int8)
    o_idx_map = {o: i for i, o in enumerate(oblasts)}

    far_future = pd.Timestamp("2099-01-01", tz="UTC")
    clean_df["finished_at_safe"] = clean_df["finished_at"].fillna(far_future)

    print("⚡ Векторизоване заповнення матриць активності на момент T та стартів у вікні (T, T+1h]...")
    for _, row in clean_df.iterrows():
        o_idx = o_idx_map[row["oblast"]]
        s_t = row["started_at"]
        e_t = row["finished_at_safe"]

        s_sec = (s_t - min_time).total_seconds()
        e_sec = (e_t - min_time).total_seconds()

        # Alert is active at timestamp T_k if T_k >= s_t and T_k < e_t
        k_start = max(0, int(np.ceil(s_sec / 3600.0)))
        k_end = min(n_hours - 1, int(np.ceil(e_sec / 3600.0) - 1))
        if k_end >= k_start:
            active_at_T[k_start:k_end + 1, o_idx] = 1

        # Alert starts in window (T_k, T_k + 1h]
        k_onset = int(s_sec // 3600)
        if 0 <= k_onset < n_hours:
            onset_next_1h[k_onset, o_idx] = 1

    # 4. Generate records for each region
    print("🛠️ Формування панельних ознак (час, сусіди, історія)...")
    records = []

    # Precalculate temporal features for decision moments T
    kyiv_times = hourly_index.tz_convert("Europe/Kyiv")
    hours = kyiv_times.hour.values
    day_of_weeks = kyiv_times.dayofweek.values
    months = kyiv_times.month.values
    is_weekends = (day_of_weeks >= 5).astype(np.int8)

    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    # Precalculate train/val/test splits with 2-hour Embargo buffer
    train_end = pd.Timestamp("2024-04-30 23:59:59", tz="UTC")
    val_end = pd.Timestamp("2025-05-31 23:59:59", tz="UTC")

    embargo_buffer = pd.Timedelta(hours=2)
    split_labels = np.full(n_hours, "train", dtype=object)

    is_train = hourly_index <= (train_end - embargo_buffer)
    is_train_embargo = (hourly_index > (train_end - embargo_buffer)) & (hourly_index <= train_end)
    is_val = (hourly_index > train_end) & (hourly_index <= (val_end - embargo_buffer))
    is_val_embargo = (hourly_index > (val_end - embargo_buffer)) & (hourly_index <= val_end)
    is_test = hourly_index > val_end

    split_labels[is_train] = "train"
    split_labels[is_train_embargo] = "embargo_train"
    split_labels[is_val] = "val"
    split_labels[is_val_embargo] = "embargo_val"
    split_labels[is_test] = "test"

    for o_name, o_idx in o_idx_map.items():
        # Own active status at moment T
        own_active_at_T = active_at_T[:, o_idx]

        # Target: whether new alert starts in (T, T+1h]
        target = onset_next_1h[:, o_idx]

        # Autoregressive historical burden strictly BEFORE moment T (closed="left")
        s_series = pd.Series(own_active_at_T)
        burden_6h = s_series.rolling(window=6, min_periods=1, closed="left").mean().fillna(0).values
        burden_24h = s_series.rolling(window=24, min_periods=1, closed="left").mean().fillna(0).values

        # Spatial neighbors status at moment T
        nbr_names = neighbor_map[o_name]
        nbr_indices = [o_idx_map[n] for n in nbr_names if n in o_idx_map]

        if len(nbr_indices) > 0:
            nbr_active_matrix = active_at_T[:, nbr_indices]
            nbr_active_count = nbr_active_matrix.sum(axis=1)
            nbr_any_active = (nbr_active_count > 0).astype(np.int8)
            nbr_active_ratio = nbr_active_count / len(nbr_indices)
        else:
            nbr_active_count = np.zeros(n_hours, dtype=np.int8)
            nbr_any_active = np.zeros(n_hours, dtype=np.int8)
            nbr_active_ratio = np.zeros(n_hours, dtype=np.float32)

        c_id = cluster_map.get(o_name, 1)

        region_df = pd.DataFrame({
            "timestamp_utc": hourly_index,
            "oblast": o_name,
            "cluster_id": c_id,
            "split_set": split_labels,
            "is_active_T": own_active_at_T,
            # Features
            "hour_sin": hour_sin.astype(np.float32),
            "hour_cos": hour_cos.astype(np.float32),
            "day_of_week": day_of_weeks.astype(np.int8),
            "is_weekend": is_weekends,
            "month": months.astype(np.int8),
            "burden_last_6h": burden_6h.astype(np.float32),
            "burden_last_24h": burden_24h.astype(np.float32),
            "neighbors_active_count": nbr_active_count.astype(np.int8),
            "neighbor_any_active": nbr_any_active,
            "neighbor_active_ratio": nbr_active_ratio.astype(np.float32),
            # Target: onset in (T, T+1h]
            "target_onset_next_1h": target,
        })
        records.append(region_df)

    panel_df = pd.concat(records, ignore_index=True)

    # 5. Filter out embargo buffer
    panel_df = panel_df[~panel_df["split_set"].isin(["embargo_train", "embargo_val"])].copy()

    # 6. CRUCIAL RULE: Forecast strictly when alert is NOT active at moment T
    print(f"🔍 Всього спостережень до фільтрації спокою: {len(panel_df):,}")
    panel_df = panel_df[panel_df["is_active_T"] == 0].copy()
    panel_df.drop(columns=["is_active_T"], inplace=True)
    print(f"✅ Залишено періодів спокою (is_active_T == 0): {len(panel_df):,}")

    # One-Hot Encoding for cluster_id (Categorical, strictly from Train)
    panel_df["cluster_1_rear"] = (panel_df["cluster_id"] == 1).astype(np.int8)
    panel_df["cluster_2_center"] = (panel_df["cluster_id"] == 2).astype(np.int8)
    panel_df["cluster_3_front"] = (panel_df["cluster_id"] == 3).astype(np.int8)

    print(f"💾 Збереження {len(panel_df):,} рядків матриці ознак у {output_file}...")
    panel_df.to_parquet(output_file, engine="pyarrow", index=False)
    print("✅ Feature Store успішно згенеровано!")

    return panel_df


if __name__ == "__main__":
    generate_feature_store()
