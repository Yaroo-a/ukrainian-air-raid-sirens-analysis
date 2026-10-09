"""Spatial Propagation & Lead-Lag Route Analysis Module (Phase 4 - RQ3).

Performs:
1. Builds Land Adjacency Matrix for 49 pairs of physical neighbors.
2. Resamples alert intervals to 10-minute discrete time grid (2022-2026).
3. Detects and isolates synchronous nationwide alerts (>= 15 oblasts simultaneously).
4. Computes cross-correlation with lags tau in [-180, +180] min for all adjacent pairs.
5. Accurately profiles both synchronous co-activations (< 10 min) and directional lead-times.
6. Generates comprehensive 2-panel visualization:
   - Panel 1: Top-15 Neighbor Pairs by Correlation Strength (Co-activation network).
   - Panel 2: Directional Wave Delays and Lead-Times between Neighbors.
7. Exports reports/figures/06_spatial_cross_correlation_routes.png & spatial_lead_lag_summary.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Land border adjacency graph (49 direct physical border neighbors)
ADJACENCY_PAIRS = [
    # Захід
    ("Закарпатська область", "Львівська область"),
    ("Закарпатська область", "Івано-Франківська область"),
    ("Львівська область", "Волинська область"),
    ("Львівська область", "Рівненська область"),
    ("Львівська область", "Тернопільська область"),
    ("Львівська область", "Івано-Франківська область"),
    ("Волинська область", "Рівненська область"),
    ("Рівненська область", "Житомирська область"),
    ("Рівненська область", "Хмельницька область"),
    ("Рівненська область", "Тернопільська область"),
    ("Тернопільська область", "Івано-Франківська область"),
    ("Тернопільська область", "Чернівецька область"),
    ("Тернопільська область", "Хмельницька область"),
    ("Івано-Франківська область", "Чернівецька область"),
    ("Чернівецька область", "Хмельницька область"),
    ("Чернівецька область", "Вінницька область"),
    ("Хмельницька область", "Житомирська область"),
    ("Хмельницька область", "Вінницька область"),
    ("Житомирська область", "Вінницька область"),
    ("Житомирська область", "Київська область"),
    ("Вінницька область", "Київська область"),
    ("Вінницька область", "Черкаська область"),
    ("Вінницька область", "Кіровоградська область"),
    ("Вінницька область", "Одеська область"),
    # Центр та Північ
    ("Київська область", "м. Київ"),
    ("Київська область", "Чернігівська область"),
    ("Київська область", "Полтавська область"),
    ("Київська область", "Черкаська область"),
    ("Чернігівська область", "Полтавська область"),
    ("Чернігівська область", "Сумська область"),
    ("Сумська область", "Полтавська область"),
    ("Сумська область", "Харківська область"),
    ("Полтавська область", "Черкаська область"),
    ("Полтавська область", "Кіровоградська область"),
    ("Полтавська область", "Дніпропетровська область"),
    ("Полтавська область", "Харківська область"),
    ("Черкаська область", "Кіровоградська область"),
    ("Кіровоградська область", "Одеська область"),
    ("Кіровоградська область", "Миколаївська область"),
    ("Кіровоградська область", "Дніпропетровська область"),
    # Схід та Південь
    ("Харківська область", "Дніпропетровська область"),
    ("Харківська область", "Донецька область"),
    ("Дніпропетровська область", "Донецька область"),
    ("Дніпропетровська область", "Запорізька область"),
    ("Дніпропетровська область", "Миколаївська область"),
    ("Донецька область", "Запорізька область"),
    ("Запорізька область", "Херсонська область"),
    ("Херсонська область", "Миколаївська область"),
    ("Миколаївська область", "Одеська область"),
]


def analyze_spatial_propagation(
    parquet_path: str | Path,
    output_dir: str | Path,
    time_step_min: int = 10,
    max_lag_min: int = 180,
    nationwide_sync_threshold: int = 15,
) -> pd.DataFrame:
    """Computes spatial cross-correlation and directional lead-times."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 1. Завантаження очищеного датасету з {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    clean_df = df[~df["is_occupied_extreme"] & df["is_finished"]].copy()

    # --- 2. Discrete Grid ---
    min_time = clean_df["started_at"].min().floor("10min")
    max_time = clean_df["finished_at"].max().ceil("10min")
    time_index = pd.date_range(start=min_time, end=max_time, freq=f"{time_step_min}min", tz="UTC")
    n_steps = len(time_index)
    oblasts = sorted(clean_df["oblast"].unique())
    oblast_to_idx = {o: i for i, o in enumerate(oblasts)}

    grid_matrix = np.zeros((n_steps, len(oblasts)), dtype=np.int8)
    for _, row in clean_df.iterrows():
        o_idx = oblast_to_idx[row["oblast"]]
        start_step = max(0, int((row["started_at"] - min_time).total_seconds() // (time_step_min * 60)))
        end_step = min(n_steps - 1, int((row["finished_at"] - min_time).total_seconds() // (time_step_min * 60)))
        if end_step >= start_step:
            grid_matrix[start_step:end_step + 1, o_idx] = 1

    # --- 3. Filter Synchronous Nationwide Events ---
    simultaneous_active = grid_matrix.sum(axis=1)
    sync_mask = simultaneous_active >= nationwide_sync_threshold
    local_grid = grid_matrix.copy()
    local_grid[sync_mask, :] = 0

    # --- 4. Cross-Correlation for Adjacent Pairs ---
    max_lags = max_lag_min // time_step_min
    lag_steps = np.arange(-max_lags, max_lags + 1)
    lag_minutes = lag_steps * time_step_min

    results = []
    for region_a, region_b in ADJACENCY_PAIRS:
        if region_a not in oblast_to_idx or region_b not in oblast_to_idx:
            continue
        idx_a = oblast_to_idx[region_a]
        idx_b = oblast_to_idx[region_b]

        series_a = local_grid[:, idx_a]
        series_b = local_grid[:, idx_b]

        std_a = np.std(series_a)
        std_b = np.std(series_b)
        if std_a == 0 or std_b == 0:
            continue

        corrs = []
        for lag in lag_steps:
            if lag < 0:
                c = np.corrcoef(series_a[:lag], series_b[-lag:])[0, 1]
            elif lag > 0:
                c = np.corrcoef(series_a[lag:], series_b[:-lag])[0, 1]
            else:
                c = np.corrcoef(series_a, series_b)[0, 1]
            corrs.append(c)

        corrs = np.array(corrs)
        best_idx = np.argmax(corrs)
        best_lag_min = int(lag_minutes[best_idx])
        max_corr = float(corrs[best_idx])
        corr_lag0 = float(corrs[len(lag_steps) // 2])

        # Also calculate average lead-lag over positive and negative correlation shoulders
        pos_shoulder = corrs[lag_minutes > 0].mean()
        neg_shoulder = corrs[lag_minutes < 0].mean()
        shoulder_diff = pos_shoulder - neg_shoulder

        if best_lag_min > 0 or shoulder_diff > 0.01:
            dominant_direction = f"{region_a} ➔ {region_b}"
            direction_type = "Спрямоване випередження (A ➔ B)"
            lead_time = best_lag_min if best_lag_min > 0 else 10
        elif best_lag_min < 0 or shoulder_diff < -0.01:
            dominant_direction = f"{region_b} ➔ {region_a}"
            direction_type = "Спрямоване випередження (B ➔ A)"
            lead_time = abs(best_lag_min) if best_lag_min < 0 else 10
        else:
            dominant_direction = f"{region_a} ⟷ {region_b}"
            direction_type = "Синхронне включення (< 10 хв)"
            lead_time = 0

        # Number of joint co-activations
        co_activations = int((series_a & series_b).sum())

        results.append({
            "region_a": region_a,
            "region_b": region_b,
            "dominant_pair": dominant_direction,
            "direction_type": direction_type,
            "lead_time_min": lead_time,
            "max_correlation": round(max_corr, 4),
            "correlation_lag0": round(corr_lag0, 4),
            "joint_active_intervals": co_activations,
        })

    results_df = pd.DataFrame(results).sort_values("max_correlation", ascending=False).reset_index(drop=True)

    # Export Full Clean Summary CSV
    csv_path = out_dir / "spatial_lead_lag_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"💾 Оновлено підсумок просторових затримок для всіх {len(results_df)} пар: {csv_path}")

    # --- 5. Generate Multi-Panel Figure ---
    print("🎨 Побудова розширеної візуалізації взаємних затримок та кореляцій...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Panel 1: Top-15 Strongest Adjacent Pairs (Correlation Strength)
    top_15_corr = results_df.head(15).copy()
    pair_labels_1 = [f"{r['region_a']} — {r['region_b']}" for _, r in top_15_corr.iterrows()]
    colors_1 = sns.color_palette("mako", n_colors=15)

    bars1 = ax1.barh(pair_labels_1, top_15_corr["max_correlation"], color=colors_1, alpha=0.85)
    ax1.invert_yaxis()
    ax1.set_xlabel("Коефіцієнт крос-кореляції (r)", fontsize=12)
    ax1.set_title("Топ-15 суміжних пар за силою зв'язку (Координація сусідів)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, (_, row) in zip(bars1, top_15_corr.iterrows()):
        ax1.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"r = {row['max_correlation']:.3f} | {row['direction_type'][:12]}...",
            va="center",
            fontsize=9.5,
        )
    ax1.set_xlim(0, 1.0)

    # Panel 2: Distribution of Lead Times & Directional Flow
    type_counts = results_df["direction_type"].value_counts()
    wedges, texts, autotexts = ax2.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2b83ba", "#fdae61", "#d7191c"],
        explode=[0.05] * len(type_counts),
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax2.set_title("Структура часових зв'язків між сусідами (49 пар)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig_path = figures_dir / "06_spatial_cross_correlation_routes.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"   Оновлено графік: {fig_path}")

    return results_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parquet_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    reports_dir = project_root / "reports"

    df_res = analyze_spatial_propagation(parquet_file, reports_dir)
    print("\nТоп-5 пар за силою кореляції:")
    print(df_res[["dominant_pair", "max_correlation", "lead_time_min", "direction_type"]].head(5).to_string())
