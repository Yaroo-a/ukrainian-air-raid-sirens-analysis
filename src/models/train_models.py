"""Модуль навчання моделей машинного навчання та тюнінгу порогів (Фаза 2.3 - Набори A, B, C, C').

Реалізує:
1. Logistic Regression (L2, class_weight='balanced', StandardScaler виключно на Train)
2. LightGBM (Gradient Boosted Decision Trees, early stopping виключно на Validation)
3. Чотирьохрівневий Ablation Study:
   - Підмножина A (10 ознак): Часові цикли, burden (6h/24h) та Train-кластери (No Spatial)
   - Підмножина B (13 ознак): A + просторовий стан сусідів у момент T (With Spatial)
   - Підмножина C (22 ознаки): B + адаптивні частоти (7d/14d/30d), нац. активність, старти за 24h,
     час від останньої тривоги та лаги сусідів за 1, 2, 3 год (Spatial + Adaptive Full)
   - Підмножина C' (19 ознак): B + адаптивні частоти (7d/14d/30d) + лаги сусідів за 1, 2, 3 год
     (без national_active_count, hours_since_last_alert та onsets_last_24h)
4. Пошук оптимального порогу p* = argmax F2(p) виключно на Validation
5. Збереження навчених моделей та артефактів для Phase 2.4 (Test evaluation)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
)
import lightgbm as lgb

FEATURE_SET_A = [
    "hour_sin", "hour_cos", "day_of_week", "is_weekend", "month",
    "burden_last_6h", "burden_last_24h",
    "cluster_1_low_activity", "cluster_2_medium_activity", "cluster_3_high_activity",
]

FEATURE_SET_B = FEATURE_SET_A + [
    "neighbors_active_count", "neighbor_any_active", "neighbor_active_ratio",
]

FEATURE_SET_C = FEATURE_SET_B + [
    "adaptive_rate_7d", "adaptive_rate_14d", "adaptive_rate_30d",
    "national_active_count", "onsets_last_24h", "hours_since_last_alert",
    "neighbors_active_lag_1h", "neighbors_active_lag_2h", "neighbors_active_lag_3h",
]

FEATURE_SET_C_PRIME = FEATURE_SET_B + [
    "adaptive_rate_7d", "adaptive_rate_14d", "adaptive_rate_30d",
    "neighbors_active_lag_1h", "neighbors_active_lag_2h", "neighbors_active_lag_3h",
]


def load_datasets(parquet_path: str = "data/processed/ml_features_hourly.parquet"):
    """Завантажує набори Train та Validation з паркету."""
    print(f"Завантаження даних із {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    train_df = df[df["split_set"] == "train"].copy()
    val_df = df[df["split_set"] == "val"].copy()

    print(f"Train розмір: {train_df.shape}, Val розмір: {val_df.shape}")
    print(f"Train позитивні: {train_df['target_onset_next_1h'].sum()} ({train_df['target_onset_next_1h'].mean():.4%})")
    print(f"Val позитивні: {val_df['target_onset_next_1h'].sum()} ({val_df['target_onset_next_1h'].mean():.4%})")
    return train_df, val_df


def evaluate_threshold_grid(y_true, y_probas, thresholds=None):
    """Сканує сітку порогів та знаходить оптимальний поріг p* для максимізації F2 виключно на Validation."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.70, 175)

    records = []
    best_f2 = -1.0
    best_record = {}

    pr_auc = average_precision_score(y_true, y_probas)
    brier = brier_score_loss(y_true, y_probas)

    for th in thresholds:
        preds = (y_probas >= th).astype(int)
        r = recall_score(y_true, preds, zero_division=0)
        p = precision_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)

        row = {
            "threshold": float(th),
            "precision": float(p),
            "recall": float(r),
            "f2_score": float(f2),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
        }
        records.append(row)

        if f2 > best_f2:
            best_f2 = f2
            best_record = row

    df_curve = pd.DataFrame(records)
    return best_record, df_curve


def train_logistic_regression(X_train, y_train, feature_names):
    """Навчає масштабовану логістичну регресію (StandardScaler строго на Train)."""
    print(f"Навчання Logistic Regression ({len(feature_names)} ознак)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_names])

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_lightgbm_model(X_train, y_train, X_val=None, y_val=None, feature_names=None, es_days: int = 60, embargo_hours: int = 2):
    """Навчає LightGBM з ранньою зупинкою на виділеному внутрішньому хронологічному спліті Train_ES
    (останні es_days днів Train із 2-годинним embargo після Train_Fit), залишаючи Validation вибірку
    100% недоторканою для калібрування порогу p* та вибору моделі за Val PR-AUC."""
    print(f"Навчання LightGBM ({len(feature_names)} ознак, внутрішній Train_ES={es_days}д з embargo={embargo_hours}г)...")
    t_max = X_train["timestamp_utc"].max()
    es_start = t_max - pd.Timedelta(days=es_days)
    fit_end = es_start - pd.Timedelta(hours=embargo_hours)

    fit_mask = (X_train["timestamp_utc"] < fit_end).values
    es_mask = (X_train["timestamp_utc"] >= es_start).values

    y_train_arr = np.asarray(y_train)
    X_fit = X_train.loc[fit_mask, feature_names]
    y_fit = y_train_arr[fit_mask]
    X_es = X_train.loc[es_mask, feature_names]
    y_es = y_train_arr[es_mask]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_es, y_es)],
        eval_names=["train_es"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def run_pipeline():
    """Повний цикл навчання, валідації та оцінки для фази 2.3 (A, B, C, C')."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/official/figures", exist_ok=True)

    train_df, val_df = load_datasets()

    y_train = train_df["target_onset_next_1h"].values
    y_val = val_df["target_onset_next_1h"].values

    baselines_df = pd.read_csv("reports/official/baselines_benchmark.csv")

    # 1. Logistic Regression: Set A, Set B, Set C, Set C'
    lr_a, scaler_a = train_logistic_regression(train_df, y_train, FEATURE_SET_A)
    p_val_lr_a = lr_a.predict_proba(scaler_a.transform(val_df[FEATURE_SET_A]))[:, 1]
    best_lr_a, curve_lr_a = evaluate_threshold_grid(y_val, p_val_lr_a)
    joblib.dump(lr_a, "models/logreg_nospatial.joblib")
    joblib.dump(scaler_a, "models/scaler_nospatial.joblib")

    lr_b, scaler_b = train_logistic_regression(train_df, y_train, FEATURE_SET_B)
    p_val_lr_b = lr_b.predict_proba(scaler_b.transform(val_df[FEATURE_SET_B]))[:, 1]
    best_lr_b, curve_lr_b = evaluate_threshold_grid(y_val, p_val_lr_b)
    joblib.dump(lr_b, "models/logreg_spatial.joblib")
    joblib.dump(scaler_b, "models/scaler_spatial.joblib")

    lr_c, scaler_c = train_logistic_regression(train_df, y_train, FEATURE_SET_C)
    p_val_lr_c = lr_c.predict_proba(scaler_c.transform(val_df[FEATURE_SET_C]))[:, 1]
    best_lr_c, curve_lr_c = evaluate_threshold_grid(y_val, p_val_lr_c)
    joblib.dump(lr_c, "models/logreg_adaptive.joblib")
    joblib.dump(scaler_c, "models/scaler_adaptive.joblib")

    lr_cp, scaler_cp = train_logistic_regression(train_df, y_train, FEATURE_SET_C_PRIME)
    p_val_lr_cp = lr_cp.predict_proba(scaler_cp.transform(val_df[FEATURE_SET_C_PRIME]))[:, 1]
    best_lr_cp, curve_lr_cp = evaluate_threshold_grid(y_val, p_val_lr_cp)
    joblib.dump(lr_cp, "models/logreg_c_prime.joblib")
    joblib.dump(scaler_cp, "models/scaler_c_prime.joblib")

    # 2. LightGBM: Set A, Set B, Set C, Set C'
    lgb_a = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_A)
    p_val_lgb_a = lgb_a.predict_proba(val_df[FEATURE_SET_A])[:, 1]
    best_lgb_a, curve_lgb_a = evaluate_threshold_grid(y_val, p_val_lgb_a)
    joblib.dump(lgb_a, "models/lightgbm_nospatial.joblib")

    lgb_b = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_B)
    p_val_lgb_b = lgb_b.predict_proba(val_df[FEATURE_SET_B])[:, 1]
    best_lgb_b, curve_lgb_b = evaluate_threshold_grid(y_val, p_val_lgb_b)
    joblib.dump(lgb_b, "models/lightgbm_spatial.joblib")

    lgb_c = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_C)
    p_val_lgb_c = lgb_c.predict_proba(val_df[FEATURE_SET_C])[:, 1]
    best_lgb_c, curve_lgb_c = evaluate_threshold_grid(y_val, p_val_lgb_c)
    joblib.dump(lgb_c, "models/lightgbm_adaptive.joblib")

    lgb_cp = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_C_PRIME)
    p_val_lgb_cp = lgb_cp.predict_proba(val_df[FEATURE_SET_C_PRIME])[:, 1]
    best_lgb_cp, curve_lgb_cp = evaluate_threshold_grid(y_val, p_val_lgb_cp)
    joblib.dump(lgb_cp, "models/lightgbm_c_prime.joblib")

    # Збереження всіх порогів у JSON
    optimal_thresholds = {
        "Baseline 1: True Naive Persistence": 0.500,
        "Baseline 2: Historical Lookup Table": float(
            baselines_df.loc[baselines_df["model"].str.contains("Lookup"), "threshold"].iloc[0]
        ),
        "Baseline 3: Adaptive 14-day Rolling": float(
            baselines_df.loc[baselines_df["model"].str.contains("Adaptive"), "threshold"].iloc[0]
        ),
        "LogReg (Set A: No Spatial)": float(best_lr_a["threshold"]),
        "LogReg (Set B: With Spatial)": float(best_lr_b["threshold"]),
        "LogReg (Set C: Spatial + Adaptive)": float(best_lr_c["threshold"]),
        "LogReg (Set C': Adaptive + Lags)": float(best_lr_cp["threshold"]),
        "LightGBM (Set A: No Spatial)": float(best_lgb_a["threshold"]),
        "LightGBM (Set B: With Spatial)": float(best_lgb_b["threshold"]),
        "LightGBM (Set C: Spatial + Adaptive)": float(best_lgb_c["threshold"]),
        "LightGBM (Set C': Adaptive + Lags)": float(best_lgb_cp["threshold"]),
    }
    with open("models/optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)

    benchmark_rows = []
    for _, r in baselines_df.iterrows():
        benchmark_rows.append({
            "model": r["model"],
            "feature_set": "None",
            "threshold": r["threshold"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f2_score": r["f2_score"],
            "pr_auc": r["pr_auc"],
            "brier_score": r["brier_score"],
        })

    models_meta = [
        ("LogReg (Set A: No Spatial)", "A (Time, Burden & Cluster)", best_lr_a),
        ("LogReg (Set B: With Spatial)", "B (A + Spatial at T)", best_lr_b),
        ("LogReg (Set C: Spatial + Adaptive)", "C (B + Adaptive, Nat & Lags)", best_lr_c),
        ("LogReg (Set C': Adaptive + Lags)", "C' (B + Adaptive & Lags only)", best_lr_cp),
        ("LightGBM (Set A: No Spatial)", "A (Time, Burden & Cluster)", best_lgb_a),
        ("LightGBM (Set B: With Spatial)", "B (A + Spatial at T)", best_lgb_b),
        ("LightGBM (Set C: Spatial + Adaptive)", "C (B + Adaptive, Nat & Lags)", best_lgb_c),
        ("LightGBM (Set C': Adaptive + Lags)", "C' (B + Adaptive & Lags only)", best_lgb_cp),
    ]

    for name, fset, res in models_meta:
        benchmark_rows.append({
            "model": name,
            "feature_set": fset,
            "threshold": res["threshold"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f2_score": res["f2_score"],
            "pr_auc": res["pr_auc"],
            "brier_score": res["brier_score"],
        })

    res_df = pd.DataFrame(benchmark_rows)
    res_df.to_csv("reports/official/ml_validation_benchmark.csv", index=False)
    print("\n" + "=" * 95)
    print("ПІДСУМКОВА ТАБЛИЦЯ ВАЛІДАЦІЇ (Validation Set: A vs B vs C vs C'):")
    print("=" * 95)
    print(res_df.to_string(index=False))

    plot_curves(
        y_val,
        probas_dict={
            "LogReg (Set A: No Spatial)": p_val_lr_a,
            "LogReg (Set B: With Spatial)": p_val_lr_b,
            "LogReg (Set C: Spatial + Adaptive)": p_val_lr_c,
            "LogReg (Set C': Adaptive + Lags)": p_val_lr_cp,
            "LightGBM (Set A: No Spatial)": p_val_lgb_a,
            "LightGBM (Set B: With Spatial)": p_val_lgb_b,
            "LightGBM (Set C: Spatial + Adaptive)": p_val_lgb_c,
            "LightGBM (Set C': Adaptive + Lags)": p_val_lgb_cp,
        },
        curves_dict={
            "LogReg (Set A: No Spatial)": curve_lr_a,
            "LogReg (Set B: With Spatial)": curve_lr_b,
            "LogReg (Set C: Spatial + Adaptive)": curve_lr_c,
            "LogReg (Set C': Adaptive + Lags)": curve_lr_cp,
            "LightGBM (Set A: No Spatial)": curve_lgb_a,
            "LightGBM (Set B: With Spatial)": curve_lgb_b,
            "LightGBM (Set C: Spatial + Adaptive)": curve_lgb_c,
            "LightGBM (Set C': Adaptive + Lags)": curve_lgb_cp,
        },
        optimal_thresholds=optimal_thresholds,
        figures_dir="reports/official/figures",
        dataset_tag="",
    )
    print("Графіки валідації збережено успішно.")


def plot_curves(
    y_val,
    probas_dict,
    curves_dict,
    optimal_thresholds,
    figures_dir: str = "reports/official/figures",
    dataset_tag: str = "",
):
    """Будує Precision-Recall криві та графіки F2 від порогу на Validation (A, B, C, C')."""
    os.makedirs(figures_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "LogReg (Set A: No Spatial)": "#9ecae1",
        "LogReg (Set B: With Spatial)": "#4292c6",
        "LogReg (Set C: Spatial + Adaptive)": "#08519c",
        "LogReg (Set C': Adaptive + Lags)": "#6a51a3",
        "LightGBM (Set A: No Spatial)": "#fdae6b",
        "LightGBM (Set B: With Spatial)": "#f16913",
        "LightGBM (Set C: Spatial + Adaptive)": "#d62728",
        "LightGBM (Set C': Adaptive + Lags)": "#8c2d04",
    }

    title_suffix = f" [{dataset_tag}]" if dataset_tag else ""

    # 1. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=300)
    base_rate = np.mean(y_val)
    ax.axhline(base_rate, color="gray", linestyle="--", label=f"Випадковий вибір (Base rate = {base_rate:.3f})")

    for name, probas in probas_dict.items():
        precision, recall, _ = precision_recall_curve(y_val, probas)
        auc_score = average_precision_score(y_val, probas)
        ax.plot(
            recall, precision,
            label=f"{name} (PR-AUC = {auc_score:.4f})",
            color=colors.get(name, "black"),
            linewidth=2.3 if "Set C" in name else 1.7,
        )

    ax.set_title(f"Precision-Recall криві на Validation (Ablation Study: Набори A, B, C, C'){title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Повнота виявлення початку тривог)", fontsize=11)
    ax.set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="upper right", frameon=True, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "08_ml_pr_curves_validation.png"))
    plt.close(fig)

    # 2. F2 Score vs Classification Threshold
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=300)
    for name, curve in curves_dict.items():
        opt_th = optimal_thresholds[name]
        best_f2 = curve.loc[(curve["threshold"] - opt_th).abs().idxmin(), "f2_score"]
        ax.plot(
            curve["threshold"], curve["f2_score"],
            label=f"{name} (max F2 = {best_f2:.4f} @ p*={opt_th:.3f})",
            color=colors.get(name, "black"),
            linewidth=2.1 if "Set C" in name else 1.6,
        )
        ax.scatter([opt_th], [best_f2], color=colors.get(name, "black"), s=60, zorder=5)

    ax.set_title(f"Підбір оптимального порогу $p^*$ на Validation для максимізації $F_2$-Score (A, B, C, C'){title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Поріг класифікації ($p$)", fontsize=11)
    ax.set_ylabel("$F_2$-Score (пріоритет Recall)", fontsize=11)
    ax.set_xlim([0.01, 0.70])
    ax.legend(loc="lower center", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "09_threshold_tuning_f2.png"))
    plt.close(fig)


if __name__ == "__main__":
    run_pipeline()
