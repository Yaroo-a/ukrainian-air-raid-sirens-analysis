"""Temporal Cyclicity Analysis Module (Phase 2 - RQ1).

Performs:
1. Diurnal (24h) and Day-of-Week (7 days) frequency profiling.
2. Kruskal-Wallis non-parametric test on weekday distributions.
3. Post-hoc Mann-Whitney U tests with Bonferroni correction (21 pairs).
4. Duration distribution modeling (quantiles, skewness, median vs mean).
5. Longitudinal yearly evolution analysis (2022 to 2026).
6. Quality Gate 2 verification and figure export.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_temporal_cyclicity(parquet_path: str | Path, output_dir: str | Path) -> dict:
    """Runs complete Phase 2 analysis, saves figures, and returns statistics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 1. Завантаження очищеного датасету з {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Exclude extreme occupied regions (Luhansk, Crimea) for national time series analysis
    clean_df = df[~df["is_occupied_extreme"]].copy()
    print(f"   Аналізується {len(clean_df):,} тривог для 24 регулярних регіонів України.")

    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    days_ua = [
        "Понеділок",
        "Вівторок",
        "Середа",
        "Четвер",
        "П'ятниця",
        "Субота",
        "Неділя",
    ]
    day_map = dict(zip(days_order, days_ua))

    # --- 2. Kruskal-Wallis Test & Post-Hoc with Bonferroni ---
    print("\n🔬 2. Статистичний тест: Чи однакова активність за днями тижня?")
    daily_counts = (
        clean_df.groupby(["start_date", "start_day_of_week"])
        .size()
        .reset_index(name="alert_count")
    )
    daily_by_day = [
        daily_counts[daily_counts["start_day_of_week"] == d]["alert_count"].values
        for d in days_order
    ]

    kw_stat, kw_pvalue = stats.kruskal(*daily_by_day)
    print(f"   Kruskal-Wallis H-статистика: {kw_stat:.4f}")
    print(f"   p-value: {kw_pvalue:.4e}")

    # Post-hoc pairwise Mann-Whitney U test with Bonferroni correction
    pairwise_results = []
    bonferroni_pairs = 21  # 7 * 6 / 2
    alpha_bonf = 0.05 / bonferroni_pairs
    print(f"   Поправка Бонферроні для 21 пари: alpha_adj = {alpha_bonf:.5f}")

    for i in range(len(days_order)):
        for j in range(i + 1, len(days_order)):
            d1, d2 = days_order[i], days_order[j]
            v1, v2 = daily_by_day[i], daily_by_day[j]
            u_stat, u_p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            is_sig = u_p < alpha_bonf
            pairwise_results.append({
                "day1": day_map[d1],
                "day2": day_map[d2],
                "u_stat": u_stat,
                "p_value": u_p,
                "is_significant": is_sig,
            })

    sig_pairs = [p for p in pairwise_results if p["is_significant"]]
    print(f"   Знайдено {len(sig_pairs)} статистично значущих пар днів з поправкою Бонферроні.")

    # Day of week ranking
    day_summary = (
        daily_counts.groupby("start_day_of_week")["alert_count"]
        .agg(["median", "mean", "std"])
        .reindex(days_order)
    )
    day_summary.index = [day_map[d] for d in day_summary.index]
    print("\n   Середня та медіанна кількість тривог на добу за днями:")
    print(day_summary.to_string())

    # --- 3. Figure 1: Heatmap 24h x 7 Days ---
    print("\n🎨 3. Побудова теплової карти (Година доби x День тижня)...")
    heatmap_data = (
        clean_df.groupby(["start_day_of_week", "start_hour_kyiv"])
        .size()
        .unstack(fill_value=0)
    )
    heatmap_data = heatmap_data.reindex(days_order)
    heatmap_data.index = [day_map[d] for d in heatmap_data.index]

    plt.figure(figsize=(15, 6))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Кількість оголошених тривог"},
        linewidths=0.5,
    )
    plt.title(
        "Теплова карта оголошення тривог: Година доби (Europe/Kyiv) × День тижня (2022–2026)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Година доби (Europe/Kyiv)", fontsize=12)
    plt.ylabel("День тижня", fontsize=12)
    plt.tight_layout()
    fig1_path = out_dir / "01_heatmap_24x7.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"   Збережено: {fig1_path}")

    # --- 4. Figure 2: Yearly Hourly Evolution (2022 - 2026) ---
    print("\n📈 4. Побудова графіка еволюції добових профілів за роками...")
    hourly_by_year = (
        clean_df.groupby(["start_year", "start_hour_kyiv"])
        .size()
        .unstack(level=0, fill_value=0)
    )
    hourly_pct = (hourly_by_year / hourly_by_year.sum()) * 100

    plt.figure(figsize=(13, 6))
    palette = sns.color_palette("tab10", n_colors=clean_df["start_year"].nunique())
    for idx, year in enumerate(sorted(clean_df["start_year"].unique())):
        plt.plot(
            hourly_pct.index,
            hourly_pct[year],
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=palette[idx],
            label=f"{year} рік (всього {hourly_by_year[year].sum():,} тривог)",
        )

    plt.title(
        "Еволюція добового профілю тривог за роками (частка % від загальної кількості за рік)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Година початку (Europe/Kyiv)", fontsize=12)
    plt.ylabel("Частка тривог у цю годину (%)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    fig2_path = out_dir / "02_hourly_evolution_by_year.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"   Збережено: {fig2_path}")

    # Night ratio trend
    night_stats = (
        clean_df.groupby("start_year")["is_night"]
        .agg(["mean", "count"])
        .reset_index()
    )
    night_stats["night_pct"] = night_stats["mean"] * 100

    # --- 5. Figure 3: Duration Distribution ---
    print("\n⏱️ 5. Побудова емпіричного розподілу тривалості тривог...")
    finished_durations = clean_df[clean_df["is_finished"]]["duration_minutes"]

    plt.figure(figsize=(11, 5.5))
    durations_clipped = finished_durations[finished_durations <= 240]  # First 4 hours
    sns.histplot(durations_clipped, bins=48, kde=True, color="#d95f02", alpha=0.5)

    q25 = finished_durations.quantile(0.25)
    q50 = finished_durations.median()
    q75 = finished_durations.quantile(0.75)
    q95 = finished_durations.quantile(0.95)
    mean_val = finished_durations.mean()

    plt.axvline(q50, color="blue", linestyle="--", linewidth=2, label=f"Медіана (Q50) = {q50:.1f} хв")
    plt.axvline(mean_val, color="red", linestyle="-", linewidth=2, label=f"Середнє = {mean_val:.1f} хв")
    plt.axvline(q75, color="green", linestyle=":", linewidth=1.5, label=f"Q75 = {q75:.1f} хв")
    plt.axvline(q95, color="purple", linestyle=":", linewidth=1.5, label=f"Q95 = {q95:.1f} хв")

    plt.title(
        "Розподіл тривалості повітряних тривог в Україні (до 240 хвилин)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Тривалість тривоги (хвилини)", fontsize=12)
    plt.ylabel("Кількість спостережень", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig3_path = out_dir / "03_duration_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"   Збережено: {fig3_path}")

    return {
        "kw_stat": kw_stat,
        "kw_pvalue": kw_pvalue,
        "pairwise_bonferroni": pairwise_results,
        "q25": q25,
        "median": q50,
        "q75": q75,
        "q95": q95,
        "mean": mean_val,
        "night_stats": night_stats,
        "day_summary": day_summary,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parquet_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    figures_dir = project_root / "reports" / "figures"

    results = analyze_temporal_cyclicity(parquet_file, figures_dir)

    print("\n" + "=" * 60)
    print("🚦 QUALITY GATE 2: ПЕРЕВІРКА КРИТЕРІЇВ")
    print("=" * 60)
    assert not np.isnan(results["kw_pvalue"]), "Gate 2 Fail: p-value is NaN!"
    assert (figures_dir / "01_heatmap_24x7.png").exists(), "Gate 2 Fail: Figure 1 missing!"
    assert (figures_dir / "02_hourly_evolution_by_year.png").exists(), "Gate 2 Fail: Figure 2 missing!"
    assert (figures_dir / "03_duration_distribution.png").exists(), "Gate 2 Fail: Figure 3 missing!"
    print("✔ Критерій 1: p-value Kruskal-Wallis отримано з коректною математичною оцінкою.")
    print("✔ Критерій 2: Попарний тест Бонферроні на 21 пару розраховано.")
    print("✔ Критерій 3: Усі 3 графіки високої роздільної здатності згенеровано в reports/figures/.")
    print("🎉 QUALITY GATE 2 УСПІШНО ПРОЙДЕНО!")
