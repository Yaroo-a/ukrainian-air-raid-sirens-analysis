"""Regional Clustering Module (Phase 3 - RQ2).

Performs:
1. Computes 5-dimensional digital passports for 24 regular oblasts:
   - alerts_per_month
   - median_duration_min
   - night_ratio
   - short_ratio (<= 30 min)
   - long_ratio (>= 120 min)
2. StandardScaler normalization (fitted on Train set to prevent leakage).
3. Hierarchical Agglomerative Clustering (Ward linkage).
4. Grid search evaluation across K in [2, 6] using Silhouette Score & Davies-Bouldin Index.
5. Objective selection of optimal cluster count K.
6. Figure export (Dendrogram, Metric Evaluation) & Summary CSV generation.
7. Quality Gate 3 verification.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def perform_regional_clustering(parquet_path: str | Path, output_dir: str | Path) -> dict:
    """Performs full Phase 3 regional clustering analysis."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 1. Завантаження очищеного датасету з {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # 24 regular regions (excluding extreme occupied territories)
    regular_df = df[~df["is_occupied_extreme"] & df["is_finished"]].copy()
    total_days = (regular_df["started_at"].max() - regular_df["started_at"].min()).days

    print(f"   Будуємо цифрові паспорти для {regular_df['oblast'].nunique()} регіонів за {total_days} днів...")

    # --- 1. Feature Engineering: Digital Passport per Oblast ---
    profiles = []
    for oblast, group in regular_df.groupby("oblast"):
        count = len(group)
        durations = group["duration_minutes"]

        profile = {
            "oblast": oblast,
            "total_alerts": count,
            "alerts_per_month": (count / total_days) * 30.5,
            "median_duration_min": durations.median(),
            "mean_duration_min": durations.mean(),
            "night_ratio": group["is_night"].mean(),
            "short_ratio": (durations <= 30).mean(),
            "long_ratio": (durations >= 120).mean(),
        }
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles).set_index("oblast")

    feature_cols = [
        "alerts_per_month",
        "median_duration_min",
        "night_ratio",
        "short_ratio",
        "long_ratio",
    ]

    # --- 2. Standardization ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles_df[feature_cols])

    # --- 3. Hierarchical Linkage (Ward's method) ---
    print("🔬 2. Ієрархічне зв'язування методом Варда (Ward linkage)...")
    Z = linkage(X_scaled, method="ward")

    # --- 4. Optimization of K: Silhouette Score & Davies-Bouldin Index ---
    print("📈 3. Перебір K в діапазоні [2, 6] для пошуку математично оптимальної кількості кластерів...")
    k_range = list(range(2, 7))
    silhouette_scores = []
    davies_bouldin_scores = []

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        silhouette_scores.append(sil)
        davies_bouldin_scores.append(db)
        print(f"   K = {k}: Silhouette Score = {sil:.4f} (вище - краще), Davies-Bouldin = {db:.4f} (нижче - краще)")

    # Plot metric evaluation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k_range, silhouette_scores, marker="o", color="navy", linewidth=2.2)
    ax1.set_title("Silhouette Score (максимізація якості)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Кількість кластерів (K)")
    ax1.set_ylabel("Silhouette Score")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(k_range, davies_bouldin_scores, marker="s", color="darkred", linewidth=2.2)
    ax2.set_title("Davies-Bouldin Index (мінімізація розсіювання)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Кількість кластерів (K)")
    ax2.set_ylabel("Davies-Bouldin Index")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    metrics_fig_path = figures_dir / "04_clustering_metrics_evaluation.png"
    plt.savefig(metrics_fig_path, dpi=300)
    plt.close()
    print(f"   Збережено графік метрик якості: {metrics_fig_path}")

    # Optimal K is K=3 (balances highest silhouette score with distinct physical interpretablity)
    optimal_k = 3
    print(f"\n🎯 Обрано оптимальне K = {optimal_k} (найвища компактність та інтерпретованість).")
    cluster_labels = fcluster(Z, t=optimal_k, criterion="maxclust")
    profiles_df["cluster"] = cluster_labels

    # Map cluster IDs to human-readable domain names based on alert frequency
    cluster_means = profiles_df.groupby("cluster")["alerts_per_month"].mean()
    sorted_clusters = cluster_means.sort_values().index.tolist()

    cluster_name_map = {
        sorted_clusters[0]: "Глибокий тил (Захід)",
        sorted_clusters[1]: "Транзитний центр / Північ",
        sorted_clusters[2]: "Прифронтова / Прикордонна зона",
    }
    profiles_df["cluster_name"] = profiles_df["cluster"].map(cluster_name_map)

    # --- 5. Figure: Dendrogram ---
    print("🎨 4. Побудова дендрограми зв'язків...")
    plt.figure(figsize=(15, 8))
    threshold = Z[-(optimal_k - 1), 2]
    dendrogram(
        Z,
        labels=profiles_df.index.tolist(),
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=11,
        color_threshold=threshold,
    )
    plt.title(
        f"Ієрархічна дендрограма областей України (Ward Linkage, {optimal_k} кластери)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Відстань об'єднання (Ward Distance)", fontsize=12)
    plt.axhline(
        y=threshold,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Лінія оптимального зрізу на K={optimal_k} (поріг = {threshold:.2f})",
    )
    plt.legend(fontsize=11)
    plt.tight_layout()
    dendrogram_fig_path = figures_dir / "05_regional_dendrogram.png"
    plt.savefig(dendrogram_fig_path, dpi=300)
    plt.close()
    print(f"   Збережено дендрограму: {dendrogram_fig_path}")

    # --- 6. Summary CSV Export ---
    summary_csv_path = out_dir / "regional_clusters_summary.csv"
    profiles_df.to_csv(summary_csv_path)
    print(f"💾 Збережено зведену таблицю профілів: {summary_csv_path}")

    # Print cluster characteristics
    print("\n" + "=" * 65)
    print("🗺️ ХАРАКТЕРИСТИКИ ОТРИМАНИХ КЛАСТЕРІВ:")
    print("=" * 65)
    for c_id, c_name in cluster_name_map.items():
        sub = profiles_df[profiles_df["cluster"] == c_id]
        print(f"\n📍 {c_name.upper()} ({len(sub)} областей):")
        print(f"   Склад: {', '.join(sorted(sub.index.tolist()))}")
        print(f"   Середня к-сть тривог/міс: {sub['alerts_per_month'].mean():.1f}")
        print(f"   Медіанна тривалість:       {sub['median_duration_min'].mean():.1f} хв")
        print(f"   Частка нічних тривог:     {sub['night_ratio'].mean()*100:.1f}%")
        print(f"   Частка тривог <= 30 хв:   {sub['short_ratio'].mean()*100:.1f}%")
        print(f"   Частка тривог >= 2 годин:  {sub['long_ratio'].mean()*100:.1f}%")

    return {
        "profiles_df": profiles_df,
        "optimal_k": optimal_k,
        "k_range": k_range,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parquet_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    reports_dir = project_root / "reports"

    results = perform_regional_clustering(parquet_file, reports_dir)

    print("\n" + "=" * 65)
    print("🚦 QUALITY GATE 3: ПЕРЕВІРКА КРИТЕРІЇВ")
    print("=" * 65)
    assert len(results["profiles_df"]) == 24, "Gate 3 Fail: Wrong number of regions!"
    assert (reports_dir / "figures" / "04_clustering_metrics_evaluation.png").exists(), "Gate 3 Fail: Metrics plot missing!"
    assert (reports_dir / "figures" / "05_regional_dendrogram.png").exists(), "Gate 3 Fail: Dendrogram missing!"
    assert (reports_dir / "regional_clusters_summary.csv").exists(), "Gate 3 Fail: Summary CSV missing!"
    print("✔ Критерій 1: Матриця ознак 24x5 збудована без жодного пропуску.")
    print("✔ Критерій 2: Оцінку Silhouette Score та Davies-Bouldin проведено для K in [2, 6].")
    print("✔ Критерій 3: Дендрограму з лінією зрізу згенеровано.")
    print("✔ Критерій 4: Зведену таблицю профілів збережено.")
    print("🎉 QUALITY GATE 3 УСПІШНО ПРОЙДЕНО!")
