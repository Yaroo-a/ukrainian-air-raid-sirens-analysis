"""Baseline Forecasting Models Module (Phase 2.2).

Implements Level 0 benchmarks:
1. Naive Persistence Baseline:
   Predicts that an onset in the previous hour (t-1 -> t) persists into the next hour (t -> t+1).
2. Historical Lookup Table Baseline:
   Computes empirical probability P(Onset | day_of_week, hour, cluster_id) strictly on Train set.
   Applies this lookup table out-of-time to Validation set.
3. Full Metric Evaluation:
   PR-AUC, Recall, Precision, F2-Score, Brier Score.
4. Threshold Optimization:
   Searches for optimal classification threshold p* maximizing F2 on Validation set.
5. Regional Breakdowns:
   Evaluates performance for Zakarpattia (Rear), Kyiv (Center), and Donetsk (Front).
6. Visualization & CSV Export:
   Generates reports/figures/07_baselines_performance.png & reports/baselines_benchmark.csv.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss, fbeta_score, precision_score, recall_score


def evaluate_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Calculates classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    # PR-AUC
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)

    return {
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f2_score": f2,
        "pr_auc": pr_auc,
        "brier_score": brier,
    }


def run_baselines_evaluation(
    ml_features_parquet: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Runs full Phase 2.2 baseline benchmarking."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 1. Завантаження Feature Store з {ml_features_parquet}...")
    df = pd.read_parquet(ml_features_parquet)

    train_df = df[df["split_set"] == "train"].copy()
    val_df = df[df["split_set"] == "val"].copy()

    print(f"   Train вибірка: {len(train_df):,} рядків")
    print(f"   Validation вибірка: {len(val_df):,} рядків")

    # --- 1. Baseline 1: Naive Persistence ---
    print("\n🔬 2. Оцінка Baseline 1: Naive Persistence...")
    # Target in previous step was whether an alert started between t-1 and t.
    # We derive previous onset from lagged target within each oblast
    val_df = val_df.sort_values(["oblast", "timestamp_utc"]).reset_index(drop=True)
    val_df["lagged_onset"] = val_df.groupby("oblast")["target_onset_next_1h"].shift(1).fillna(0).astype(int)

    y_val = val_df["target_onset_next_1h"].values
    y_pred_naive = val_df["lagged_onset"].values

    prec_naive = precision_score(y_val, y_pred_naive, zero_division=0)
    rec_naive = recall_score(y_val, y_pred_naive, zero_division=0)
    f2_naive = fbeta_score(y_val, y_pred_naive, beta=2, zero_division=0)
    brier_naive = brier_score_loss(y_val, y_pred_naive)

    print(f"   Naive Persistence: Precision = {prec_naive:.4f}, Recall = {rec_naive:.4f}, F2 = {f2_naive:.4f}")

    # --- 2. Baseline 2: Historical Lookup Table ---
    print("\n📈 3. Навчання Baseline 2: Historical Lookup Table (суворо на Train)...")
    # Derive hour from hour_sin/hour_cos or day_of_week
    # Extract hour directly from timestamp_utc in Kyiv time
    train_df["hour_kyiv"] = train_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    val_df["hour_kyiv"] = val_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour

    lookup_table = (
        train_df.groupby(["day_of_week", "hour_kyiv", "cluster_id"])["target_onset_next_1h"]
        .mean()
        .reset_index(name="prob_historical")
    )
    print(f"   Таблицю шансів побудовано: {len(lookup_table)} комбінацій [день, година, кластер].")

    # Merge lookup table onto Validation set
    val_merged = val_df.merge(
        lookup_table,
        on=["day_of_week", "hour_kyiv", "cluster_id"],
        how="left",
    )
    # Fill missing combinations with global train mean
    global_train_mean = train_df["target_onset_next_1h"].mean()
    val_merged["prob_historical"] = val_merged["prob_historical"].fillna(global_train_mean)

    y_prob_lookup = val_merged["prob_historical"].values

    # Grid search for optimal threshold p* maximizing F2 on Validation
    thresholds = np.linspace(0.01, 0.40, 80)
    threshold_results = []
    for th in thresholds:
        metrics = evaluate_metrics_at_threshold(y_val, y_prob_lookup, th)
        threshold_results.append(metrics)

    th_df = pd.DataFrame(threshold_results)
    best_row = th_df.loc[th_df["f2_score"].idxmax()]
    best_th = best_row["threshold"]

    print(f"   Historical Lookup: PR-AUC = {best_row['pr_auc']:.4f}, Brier Score = {best_row['brier_score']:.4f}")
    print(f"   Оптимальний поріг на Val: p* = {best_th:.3f}")
    print(f"   Метрики при p*: Precision = {best_row['precision']:.4f}, Recall = {best_row['recall']:.4f}, F2 = {best_row['f2_score']:.4f}")

    # --- 3. Regional Breakdowns for 3 Key Oblasts ---
    print("\n🗺️ 4. Регіональний зріз бейзлайну (Закарпаття vs Київ vs Донеччина)...")
    key_oblasts = ["Закарпатська область", "м. Київ", "Донецька область"]
    regional_benchmark = []

    for obl in key_oblasts:
        obl_mask = val_merged["oblast"] == obl
        y_obl = y_val[obl_mask]
        p_obl = y_prob_lookup[obl_mask]

        obl_metrics = evaluate_metrics_at_threshold(y_obl, p_obl, best_th)
        obl_metrics["oblast"] = obl
        obl_metrics["baseline_name"] = "Historical Lookup"
        regional_benchmark.append(obl_metrics)

        print(f"   {obl:<25}: Recall = {obl_metrics['recall']*100:.1f}%, Prec = {obl_metrics['precision']*100:.1f}%, F2 = {obl_metrics['f2_score']:.4f}, PR-AUC = {obl_metrics['pr_auc']:.4f}")

    # Summary benchmark table
    benchmark_rows = [
        {
            "model": "Baseline 1: Naive Persistence",
            "threshold": 0.5,
            "precision": prec_naive,
            "recall": rec_naive,
            "f2_score": f2_naive,
            "pr_auc": np.nan,
            "brier_score": brier_naive,
        },
        {
            "model": "Baseline 2: Historical Lookup (Загальний)",
            "threshold": best_th,
            "precision": best_row["precision"],
            "recall": best_row["recall"],
            "f2_score": best_row["f2_score"],
            "pr_auc": best_row["pr_auc"],
            "brier_score": best_row["brier_score"],
        },
    ]
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_csv_path = out_dir / "baselines_benchmark.csv"
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"💾 Збережено бенчмарк бейзлайнів: {benchmark_csv_path}")

    # --- 4. Multi-Panel Visualization ---
    print("\n🎨 5. Побудова графіків результатів бейзлайнів...")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Subplot 1: Precision-Recall Curve for Historical Lookup
    p_vals, r_vals, _ = precision_recall_curve(y_val, y_prob_lookup)
    axes[0].plot(r_vals, p_vals, color="#2b83ba", linewidth=2.5, label=f"Historical Lookup (AUC = {best_row['pr_auc']:.3f})")
    axes[0].scatter([rec_naive], [prec_naive], color="red", s=80, zorder=5, label=f"Naive Persistence (F2={f2_naive:.3f})")
    axes[0].axhline(y_val.mean(), color="gray", linestyle="--", label=f"Random Chance ({y_val.mean()*100:.1f}%)")
    axes[0].set_title("Precision-Recall Крива (Validation Set)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Recall (Повнота виявлених тривог)", fontsize=11)
    axes[0].set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(0, 0.40)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9.5)

    # Subplot 2: Threshold Tuning Curve for F2-Score
    axes[1].plot(th_df["threshold"], th_df["f2_score"], color="crimson", linewidth=2.2, label="F2-Score")
    axes[1].plot(th_df["threshold"], th_df["recall"], color="green", linestyle=":", linewidth=1.8, label="Recall")
    axes[1].plot(th_df["threshold"], th_df["precision"], color="navy", linestyle="--", linewidth=1.8, label="Precision")
    axes[1].axvline(best_th, color="black", linestyle="-.", label=f"Оптимум p* = {best_th:.3f}")
    axes[1].set_title("Підбір робочого порогу p* (Threshold Moving)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Поріг імовірності (Threshold)", fontsize=11)
    axes[1].set_ylabel("Значення метрик", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9.5)

    # Subplot 3: Regional Contrast (Recall vs Precision)
    reg_df = pd.DataFrame(regional_benchmark)
    x = np.arange(len(reg_df))
    width = 0.35

    axes[2].bar(x - width/2, reg_df["recall"] * 100, width, label="Recall (%)", color="#31a354", alpha=0.85)
    axes[2].bar(x + width/2, reg_df["precision"] * 100, width, label="Precision (%)", color="#756bb1", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Закарпаття\n(Тил)", "м. Київ\n(Центр)", "Донеччина\n(Фронт)"], fontsize=10.5)
    axes[2].set_title("Регіональний контраст якості бейзлайну", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Відсоток (%)", fontsize=11)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    fig_path = figures_dir / "07_baselines_performance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"   Збережено візуалізацію бейзлайнів: {fig_path}")

    return benchmark_df, {
        "best_threshold": best_th,
        "best_f2": best_row["f2_score"],
        "pr_auc": best_row["pr_auc"],
        "naive_f2": f2_naive,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    features_file = project_root / "data" / "processed" / "ml_features_hourly.parquet"
    reports_dir = project_root / "reports"

    bench_df, metrics = run_baselines_evaluation(features_file, reports_dir)

    print("\n" + "=" * 65)
    print("🚦 QUALITY GATE 2.2: ПЕРЕВІРКА КРИТЕРІЇВ")
    print("=" * 65)
    assert (reports_dir / "baselines_benchmark.csv").exists(), "Gate 2.2 Fail: CSV missing!"
    assert (reports_dir / "figures" / "07_baselines_performance.png").exists(), "Gate 2.2 Fail: Figure missing!"
    assert not np.isnan(metrics["best_f2"]), "Gate 2.2 Fail: F2 is NaN!"
    print("✔ Критерій 1: Naive Persistence та Historical Lookup повністю реалізовано.")
    print("✔ Критерій 2: Таблицю шансів розраховано виключно на Train.")
    print("✔ Критерій 3: Оптимальний поріг p* знайдено на Validation.")
    print("✔ Критерій 4: Трьохпанельний графік результатів збережено в reports/figures/.")
    print("🎉 QUALITY GATE 2.2 УСПІШНО ПРОЙДЕНО!")
