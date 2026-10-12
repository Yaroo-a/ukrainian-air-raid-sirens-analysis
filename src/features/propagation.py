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
    time_step_min: int = 2,
    onset_window_min: int = 10,
    max_lag_min: int = 180,
    nationwide_sync_threshold: int = 15,
) -> pd.DataFrame:
    """Computes spatial onset cross-correlation and directional wave lead-times on a 2-minute grid."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 1. Завантаження очищеного датасету з {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    clean_df = df[~df["is_occupied_extreme"] & df["is_finished"]].copy()

    # --- 2. Discrete 2-Minute Grid & 10-Minute Alert Onset Impulse Windows ---
    min_time = clean_df["started_at"].min().floor("10min")
    max_time = clean_df["finished_at"].max().ceil("10min")
    n_steps = int((max_time - min_time).total_seconds() // (time_step_min * 60)) + 1
    oblasts = sorted(clean_df["oblast"].unique())
    oblast_to_idx = {o: i for i, o in enumerate(oblasts)}

    active_grid = np.zeros((n_steps, len(oblasts)), dtype=np.int8)
    onset_grid = np.zeros((n_steps, len(oblasts)), dtype=np.int8)
    window_steps = max(1, onset_window_min // time_step_min)

    start_steps = np.clip(
        ((clean_df["started_at"] - min_time).dt.total_seconds() // (time_step_min * 60)).astype(int).values,
        0,
        n_steps - 1,
    )
    end_steps = np.clip(
        ((clean_df["finished_at"] - min_time).dt.total_seconds() // (time_step_min * 60)).astype(int).values,
        0,
        n_steps - 1,
    )
    o_indices = clean_df["oblast"].map(oblast_to_idx).values

    for s_step, e_step, o_idx in zip(start_steps, end_steps, o_indices):
        if e_step >= s_step:
            active_grid[s_step:e_step + 1, o_idx] = 1
        onset_grid[s_step:min(n_steps, s_step + window_steps), o_idx] = 1

    # --- 3. Filter Synchronous Nationwide Events (>= 15 oblasts active simultaneously) ---
    simultaneous_active = active_grid.sum(axis=1)
    sync_mask = simultaneous_active >= nationwide_sync_threshold
    local_active_grid = active_grid.copy()
    local_active_grid[sync_mask, :] = 0
    local_onset_grid = onset_grid.copy()
    local_onset_grid[sync_mask, :] = 0

    # --- 4. Cross-Correlation of Onset Impulses for Adjacent Pairs ---
    max_lags = max_lag_min // time_step_min
    lag_steps = np.arange(-max_lags, max_lags + 1)
    lag_minutes = lag_steps * time_step_min
    pos_mask = lag_minutes > 0
    neg_mask = lag_minutes < 0
    pos_lags_min = lag_minutes[pos_mask]

    results = []
    for region_a, region_b in ADJACENCY_PAIRS:
        if region_a not in oblast_to_idx or region_b not in oblast_to_idx:
            continue
        idx_a = oblast_to_idx[region_a]
        idx_b = oblast_to_idx[region_b]

        series_a = local_onset_grid[:, idx_a].astype(np.float64)
        series_b = local_onset_grid[:, idx_b].astype(np.float64)

        sa_centered = series_a - series_a.mean()
        sb_centered = series_b - series_b.mean()
        denom = float(np.sqrt(np.dot(sa_centered, sa_centered) * np.dot(sb_centered, sb_centered)))
        if denom == 0.0:
            continue

        # Convention:
        # lag > 0 means A at time t precedes B at time t + lag (A ➔ B): correlate series_a[:-lag] with series_b[lag:]
        # lag < 0 means B precedes A by abs(lag) (B ➔ A): correlate series_a[-lag:] with series_b[:lag]
        corrs = np.empty(len(lag_steps), dtype=np.float64)
        for i_lag, lag in enumerate(lag_steps):
            if lag < 0:
                corrs[i_lag] = np.dot(sa_centered[-lag:], sb_centered[:lag]) / denom
            elif lag > 0:
                corrs[i_lag] = np.dot(sa_centered[:-lag], sb_centered[lag:]) / denom
            else:
                corrs[i_lag] = np.dot(sa_centered, sb_centered) / denom

        best_idx = int(np.argmax(corrs))
        best_lag_min = int(lag_minutes[best_idx])
        max_corr = float(corrs[best_idx])
        corr_lag0 = float(corrs[len(lag_steps) // 2])

        # Positive shoulder (tau in +2..+180 min, A ➔ B) vs Negative shoulder (tau in -180..-2 min, B ➔ A)
        pos_corrs = corrs[pos_mask]
        neg_corrs = corrs[neg_mask][::-1]  # aligned by +tau = 2, 4, ..., 180 min
        pos_shoulder = float(pos_corrs.mean())
        neg_shoulder = float(neg_corrs.mean())
        shoulder_diff = pos_shoulder - neg_shoulder

        if shoulder_diff > 0.001:
            dominant_direction = f"{region_a} ➔ {region_b}"
            direction_type = "Спрямоване випередження (A ➔ B)"
            if best_lag_min > 0:
                peak_nonzero_lag_min = int(best_lag_min)
            else:
                peak_nonzero_lag_min = int(pos_lags_min[int(np.argmax(pos_corrs - neg_corrs))])
            lead_time = int(peak_nonzero_lag_min)
        elif shoulder_diff < -0.001:
            dominant_direction = f"{region_b} ➔ {region_a}"
            direction_type = "Спрямоване випередження (B ➔ A)"
            if best_lag_min < 0:
                peak_nonzero_lag_min = int(best_lag_min)
            else:
                peak_nonzero_lag_min = -int(pos_lags_min[int(np.argmax(neg_corrs - pos_corrs))])
            lead_time = int(abs(peak_nonzero_lag_min))
        else:
            dominant_direction = f"{region_a} ⟷ {region_b}"
            direction_type = "Синхронне включення (< 2 хв)"
            peak_nonzero_lag_min = 0
            lead_time = 0

        # Number of joint active 2-min steps & joint onset impulse steps
        co_activations = int((local_active_grid[:, idx_a] & local_active_grid[:, idx_b]).sum())
        co_onset_steps = int((local_onset_grid[:, idx_a] & local_onset_grid[:, idx_b]).sum())

        results.append({
            "region_a": region_a,
            "region_b": region_b,
            "dominant_pair": dominant_direction,
            "direction_type": direction_type,
            "lead_time_min": lead_time,
            "best_lag_min": best_lag_min,
            "peak_nonzero_lag_min": peak_nonzero_lag_min,
            "shoulder_diff": round(shoulder_diff, 6),
            "max_correlation": round(max_corr, 4),
            "correlation_lag0": round(corr_lag0, 4),
            "joint_active_intervals": co_activations,
            "joint_onset_steps": co_onset_steps,
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
    ax1.set_xlabel("Коефіцієнт крос-кореляції імпульсів старту (r)", fontsize=12)
    ax1.set_title("Топ-15 суміжних пар за силою зв'язку стартів тривог (сітка 2 хв)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, (_, row) in zip(bars1, top_15_corr.iterrows()):
        ax1.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"r = {row['max_correlation']:.3f} | Δτ = {row['lead_time_min']} хв | {row['direction_type'][:12]}...",
            va="center",
            fontsize=9.5,
        )
    ax1.set_xlim(0, 1.0)

    # Panel 2: Distribution of Lead Times & Directional Flow
    type_counts = results_df["direction_type"].value_counts()
    palette_pie = ["#2b83ba", "#fdae61", "#d7191c"][:len(type_counts)]
    wedges, texts, autotexts = ax2.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette_pie,
        explode=[0.05] * len(type_counts),
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax2.set_title("Структура хвильових зв'язків між сусідами (49 пар, сітка 2 хв)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig_path = figures_dir / "06_spatial_cross_correlation_routes.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"   Оновлено графік: {fig_path}")

    return results_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parquet_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    reports_dir = project_root / "reports" / "official"

    df_res = analyze_spatial_propagation(parquet_file, reports_dir)
    print("\nТоп-5 пар за силою кореляції:")
    print(df_res[["dominant_pair", "max_correlation", "best_lag_min", "peak_nonzero_lag_min", "lead_time_min", "direction_type"]].head(5).to_string())
