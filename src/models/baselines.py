"""Baseline Forecasting Models Module (Phase 2.2 - Refactored).

Implements Level 0 benchmarks:
1. True Naive Persistence Baseline:
   True temporal lag of onset in the immediately preceding hour [T-1h, T).
2. Historical Lookup Table Baseline:
   Computes empirical probability P(Onset | day_of_week, hour_kyiv, cluster_id) strictly on Train.
3. Adaptive Rolling Baseline (14-day local onset frequency):
   Uses rolling empirical frequency over past 14 days before T.
4. Threshold Optimization:
   Searches for optimal classification threshold p* maximizing F2 on Validation set.
5. Regional Breakdowns & Visualizations:
   Reports metrics and generates reports/figures/07_baselines_performance.png.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score
)


def evaluate_threshold_search(y_true, y_prob, thresholds=None):
    """Searches for optimal threshold maximizing F2 score."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 150)
    best_th = 0.5
    best_f2 = -1.0
    best_p, best_r = 0, 0
    records = []

    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
        records.append({"threshold": th, "precision": p, "recall": r, "f2_score": f2})
        if f2 > best_f2:
            best_f2 = f2
            best_th = th
            best_p = p
            best_r = r

    df_curve = pd.DataFrame(records)
    best_stats = {
        "threshold": best_th,
        "precision": best_p,
        "recall": best_r,
        "f2_score": best_f2,
        "pr_auc": pr_auc,
        "brier_score": brier,
    }
    return best_stats, df_curve


def run_baselines_evaluation(
    ml_features_parquet: str | Path = "data/processed/ml_features_hourly.parquet",
    output_dir: str | Path = "reports/official",
):
    """Runs full baseline evaluation on Train/Val."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 1. Завантаження Feature Store з {ml_features_parquet}...")
    df = pd.read_parquet(ml_features_parquet)

    train_df = df[df["split_set"] == "train"].copy()
    val_df = df[df["split_set"] == "val"].copy()

    y_val = val_df["target_onset_next_1h"].values

    # 1. Baseline 1: True Naive Persistence
    print("🔬 2. Оцінка Baseline 1: True Naive Persistence (T-1 onset)...")
    naive_preds = val_df["lagged_onset_1h"].values
    prec_naive = precision_score(y_val, naive_preds, zero_division=0)
    rec_naive = recall_score(y_val, naive_preds, zero_division=0)
    f2_naive = fbeta_score(y_val, naive_preds, beta=2, zero_division=0)
    brier_naive = brier_score_loss(y_val, naive_preds)

    # 2. Baseline 2: Historical Lookup Table
    print("📈 3. Навчання Baseline 2: Historical Lookup Table (Київський час, Train-кластери)...")
    train_df["hour_kyiv"] = train_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    val_df["hour_kyiv"] = val_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour

    lookup_table = (
        train_df.groupby(["day_of_week", "hour_kyiv", "cluster_id"])["target_onset_next_1h"]
        .mean()
        .reset_index()
    )
    lookup_table.rename(columns={"target_onset_next_1h": "lookup_prob"}, inplace=True)
    global_mean = train_df["target_onset_next_1h"].mean()

    val_merged = val_df.merge(lookup_table, on=["day_of_week", "hour_kyiv", "cluster_id"], how="left")
    val_lookup_probs = val_merged["lookup_prob"].fillna(global_mean).values

    best_lookup, curve_lookup = evaluate_threshold_search(y_val, val_lookup_probs)

    # 3. Baseline 3: Adaptive 14-day Rolling Baseline
    print("🔄 4. Оцінка Baseline 3: Adaptive 14-day Rolling Frequency...")
    val_adapt_probs = val_df["adaptive_rate_14d"].values
    best_adapt, curve_adapt = evaluate_threshold_search(y_val, val_adapt_probs)

    # Summary DataFrame
    summary_rows = [
        {
            "model": "Baseline 1: True Naive Persistence",
            "threshold": 0.500,
            "precision": prec_naive,
            "recall": rec_naive,
            "f2_score": f2_naive,
            "pr_auc": np.nan,
            "brier_score": brier_naive,
        },
        {
            "model": "Baseline 2: Historical Lookup Table",
            "threshold": best_lookup["threshold"],
            "precision": best_lookup["precision"],
            "recall": best_lookup["recall"],
            "f2_score": best_lookup["f2_score"],
            "pr_auc": best_lookup["pr_auc"],
            "brier_score": best_lookup["brier_score"],
        },
        {
            "model": "Baseline 3: Adaptive 14-day Rolling",
            "threshold": best_adapt["threshold"],
            "precision": best_adapt["precision"],
            "recall": best_adapt["recall"],
            "f2_score": best_adapt["f2_score"],
            "pr_auc": best_adapt["pr_auc"],
            "brier_score": best_adapt["brier_score"],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "baselines_benchmark.csv", index=False)
    print("\n" + "=" * 80)
    print("БЕНЧМАРК БЕЙЗЛАЙНІВ НА ВАЛІДАЦІЇ:")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # PR curve
    p_lk, r_lk, _ = precision_recall_curve(y_val, val_lookup_probs)
    p_ad, r_ad, _ = precision_recall_curve(y_val, val_adapt_probs)
    base_rate = np.mean(y_val)

    ax1.plot(r_lk, p_lk, label=f"Historical Lookup (PR-AUC={best_lookup['pr_auc']:.4f})", color="steelblue", lw=2)
    ax1.plot(r_ad, p_ad, label=f"Adaptive 14-day (PR-AUC={best_adapt['pr_auc']:.4f})", color="forestgreen", lw=2)
    ax1.axhline(base_rate, color="gray", linestyle="--", label=f"Base Rate ({base_rate:.3f})")
    ax1.set_title("Precision-Recall криві бейзлайнів (Validation)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)

    # F2 vs Threshold
    ax2.plot(curve_lookup["threshold"], curve_lookup["f2_score"], label=f"Historical Lookup (p*={best_lookup['threshold']:.3f})", color="steelblue", lw=2)
    ax2.plot(curve_adapt["threshold"], curve_adapt["f2_score"], label=f"Adaptive 14-day (p*={best_adapt['threshold']:.3f})", color="forestgreen", lw=2)
    ax2.set_title("Оптимізація порогу класифікації $p^*$ для $F_2$", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Поріг p", fontsize=11)
    ax2.set_ylabel("F2-Score", fontsize=11)
    ax2.legend(loc="lower center", frameon=True)

    plt.tight_layout()
    fig.savefig(figures_dir / "07_baselines_performance.png")
    plt.close(fig)
    print(f"Збережено графіки: {figures_dir / '07_baselines_performance.png'}")


if __name__ == "__main__":
    run_baselines_evaluation()
