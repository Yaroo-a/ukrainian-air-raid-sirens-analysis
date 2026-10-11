"""Модуль фінальної оцінки на вибірці Test (червень 2025 - вересень 2026) та останніх 60 днях (Фаза 2.4 - A, B, C).

Реалізує:
1. Заздалегідь зафіксоване головне порівняння: LightGBM (Set C: Spatial + Adaptive) проти Adaptive 14-day Rolling Baseline.
2. Повний Ablation Study наборів ознак A vs B vs C для LightGBM та Logistic Regression із зафіксованими на Validation порогами p*.
3. Узгодження часових поясів (Київський час dt.tz_convert("Europe/Kyiv").dt.hour ідентично до Val).
4. Оцінку True Naive Persistence [T-1h, T), Historical Lookup та Adaptive 14-day Rolling Baseline.
5. Чесний Block Bootstrap за календарними добами (n_bootstraps=500) для 95% CI PR-AUC, F2 та парних різниць (B-A, C-B, C-A, C-Adaptive).
6. Порівняння моделей при фіксованому Recall та фіксованому Precision (на рівні Adaptive 14-day Baseline).
7. Повну оцінку на останніх 60 днях із збереженням у reports/ml_recent_60d_benchmark.csv та побудовою PR-кривих.
8. Reliability Diagram (калібрувальна крива) та квантилі прогнозованих імовірностей LightGBM (Set C).
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from src.models.train_models import FEATURE_SET_A, FEATURE_SET_B, FEATURE_SET_C


def load_test_data_and_models(parquet_path: str = "data/processed/ml_features_hourly.parquet"):
    """Завантажує тестовий набір даних, таблицю бейзлайну та натреновані моделі (A, B, C)."""
    print("Завантаження даних та моделей...")
    df = pd.read_parquet(parquet_path)
    train_df = df[df["split_set"] == "train"].copy()
    test_df = df[df["split_set"] == "test"].copy()

    with open("models/optimal_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Таблиця шансів Historical Lookup з Train (Київський час!)
    train_df["hour_kyiv"] = train_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    test_df["hour_kyiv"] = test_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    test_df["date_kyiv"] = test_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.date

    lookup = (
        train_df.groupby(["day_of_week", "hour_kyiv", "cluster_id"])["target_onset_next_1h"]
        .mean()
        .reset_index()
    )
    lookup.rename(columns={"target_onset_next_1h": "lookup_prob"}, inplace=True)
    global_mean = train_df["target_onset_next_1h"].mean()

    test_merged = test_df.merge(lookup, on=["day_of_week", "hour_kyiv", "cluster_id"], how="left")
    test_merged["lookup_prob"] = test_merged["lookup_prob"].fillna(global_mean)

    models = {
        "scaler_a": joblib.load("models/scaler_nospatial.joblib"),
        "scaler_b": joblib.load("models/scaler_spatial.joblib"),
        "scaler_c": joblib.load("models/scaler_adaptive.joblib"),
        "lr_a": joblib.load("models/logreg_nospatial.joblib"),
        "lr_b": joblib.load("models/logreg_spatial.joblib"),
        "lr_c": joblib.load("models/logreg_adaptive.joblib"),
        "lgb_a": joblib.load("models/lightgbm_nospatial.joblib"),
        "lgb_b": joblib.load("models/lightgbm_spatial.joblib"),
        "lgb_c": joblib.load("models/lightgbm_adaptive.joblib"),
    }

    return test_merged, thresholds, models


def _summarize_diff(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def day_block_bootstrap_all_models(
    dates_array: np.ndarray,
    y_true: np.ndarray,
    probas_dict: dict[str, np.ndarray],
    thresholds_dict: dict[str, float],
    n_bootstraps: int = 500,
    random_state: int = 42,
) -> tuple[dict, dict]:
    """Виконує парний Block Bootstrap за календарними добами (n_bootstraps=500)."""
    rng = np.random.RandomState(random_state)
    unique_days, day_inverse = np.unique(dates_array, return_inverse=True)
    n_days = len(unique_days)

    day_indices = [np.where(day_inverse == d)[0] for d in range(n_days)]
    y_by_day = [y_true[idx] for idx in day_indices]

    sampled_days = rng.randint(0, n_days, size=(n_bootstraps, n_days))

    # 1. Швидкий точний розрахунок F2 через добові суми TP, FP, FN
    f2_samples = {}
    for name, p_vals in probas_dict.items():
        th = thresholds_dict[name]
        preds = (p_vals >= th).astype(int)
        tp_day = np.array([int(((preds[idx] == 1) & (y_true[idx] == 1)).sum()) for idx in day_indices])
        fp_day = np.array([int(((preds[idx] == 1) & (y_true[idx] == 0)).sum()) for idx in day_indices])
        fn_day = np.array([int(((preds[idx] == 0) & (y_true[idx] == 1)).sum()) for idx in day_indices])

        tp_b = tp_day[sampled_days].sum(axis=1)
        fp_b = fp_day[sampled_days].sum(axis=1)
        fn_b = fn_day[sampled_days].sum(axis=1)
        denom = 5 * tp_b + 4 * fn_b + fp_b
        f2_samples[name] = np.where(denom > 0, (5 * tp_b) / denom, 0.0)

    # 2. Розрахунок PR-AUC на тих самих блоках діб
    prauc_samples = {name: np.zeros(n_bootstraps) for name in probas_dict if "Naive" not in name}
    p_by_day = {
        name: [p_vals[idx] for idx in day_indices]
        for name, p_vals in probas_dict.items()
        if "Naive" not in name
    }

    for b in range(n_bootstraps):
        d_sel = sampled_days[b]
        y_b = np.concatenate([y_by_day[d] for d in d_sel])
        if y_b.sum() == 0:
            continue
        for name, p_day_list in p_by_day.items():
            p_b = np.concatenate([p_day_list[d] for d in d_sel])
            prauc_samples[name][b] = average_precision_score(y_b, p_b)

    ci_results = {}
    for name in probas_dict:
        f2_arr = f2_samples[name]
        if "Naive" in name:
            ci_results[name] = {
                "f2_ci_low": float(np.percentile(f2_arr, 2.5)),
                "f2_ci_high": float(np.percentile(f2_arr, 97.5)),
                "prauc_ci_low": np.nan,
                "prauc_ci_high": np.nan,
            }
        else:
            pr_arr = prauc_samples[name]
            ci_results[name] = {
                "f2_ci_low": float(np.percentile(f2_arr, 2.5)),
                "f2_ci_high": float(np.percentile(f2_arr, 97.5)),
                "prauc_ci_low": float(np.percentile(pr_arr, 2.5)),
                "prauc_ci_high": float(np.percentile(pr_arr, 97.5)),
            }

    adapt = "Baseline 3: Adaptive 14-day Rolling"
    lgb_a, lgb_b, lgb_c = (
        "LightGBM (Set A: No Spatial)",
        "LightGBM (Set B: With Spatial)",
        "LightGBM (Set C: Spatial + Adaptive)",
    )
    lr_a, lr_b, lr_c = (
        "LogReg (Set A: No Spatial)",
        "LogReg (Set B: With Spatial)",
        "LogReg (Set C: Spatial + Adaptive)",
    )

    paired_diffs = {
        # Головне заздалегідь зафіксоване порівняння: LightGBM (Set C) vs Adaptive 14-day
        "PRIMARY_lgb_C_minus_adaptive_prauc": _summarize_diff(prauc_samples[lgb_c] - prauc_samples[adapt]),
        "PRIMARY_lgb_C_minus_adaptive_f2": _summarize_diff(f2_samples[lgb_c] - f2_samples[adapt]),
        # LogReg (Set C) vs Adaptive 14-day
        "lr_C_minus_adaptive_prauc": _summarize_diff(prauc_samples[lr_c] - prauc_samples[adapt]),
        "lr_C_minus_adaptive_f2": _summarize_diff(f2_samples[lr_c] - f2_samples[adapt]),
        # LightGBM Ablation: B-A, C-B, C-A
        "lgb_B_minus_A_prauc": _summarize_diff(prauc_samples[lgb_b] - prauc_samples[lgb_a]),
        "lgb_B_minus_A_f2": _summarize_diff(f2_samples[lgb_b] - f2_samples[lgb_a]),
        "lgb_C_minus_B_prauc": _summarize_diff(prauc_samples[lgb_c] - prauc_samples[lgb_b]),
        "lgb_C_minus_B_f2": _summarize_diff(f2_samples[lgb_c] - f2_samples[lgb_b]),
        "lgb_C_minus_A_prauc": _summarize_diff(prauc_samples[lgb_c] - prauc_samples[lgb_a]),
        "lgb_C_minus_A_f2": _summarize_diff(f2_samples[lgb_c] - f2_samples[lgb_a]),
        # LogReg Ablation: B-A, C-B, C-A
        "lr_B_minus_A_prauc": _summarize_diff(prauc_samples[lr_b] - prauc_samples[lr_a]),
        "lr_B_minus_A_f2": _summarize_diff(f2_samples[lr_b] - f2_samples[lr_a]),
        "lr_C_minus_B_prauc": _summarize_diff(prauc_samples[lr_c] - prauc_samples[lr_b]),
        "lr_C_minus_B_f2": _summarize_diff(f2_samples[lr_c] - f2_samples[lr_b]),
        "lr_C_minus_A_prauc": _summarize_diff(prauc_samples[lr_c] - prauc_samples[lr_a]),
        "lr_C_minus_A_f2": _summarize_diff(f2_samples[lr_c] - f2_samples[lr_a]),
    }

    return ci_results, paired_diffs


def compute_equal_operating_points(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float, target_precision: float):
    """Обчислює Precision при фіксованому Recall та Recall при фіксованому Precision."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    idx_rec = np.argmin(np.abs(rec - target_recall))
    prec_at_target_rec = float(prec[idx_rec])

    valid_p = np.where(prec >= target_precision)[0]
    rec_at_target_prec = float(rec[valid_p].max()) if len(valid_p) > 0 else 0.0
    return prec_at_target_rec, rec_at_target_prec


def evaluate_slice(
    dates_array: np.ndarray,
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    configs: list[tuple[str, str, float]],
    n_bootstraps: int = 500,
) -> tuple[pd.DataFrame, dict, str]:
    """Оцінює всі моделі на заданому часовому зрізі (Test або останні 60 днів)."""
    th_map = {name: th for name, _, th in configs}
    ci_map, paired_diffs = day_block_bootstrap_all_models(
        dates_array, y_true, probas, th_map, n_bootstraps=n_bootstraps
    )

    adapt_name = "Baseline 3: Adaptive 14-day Rolling"
    adapt_preds = (probas[adapt_name] >= th_map[adapt_name]).astype(int)
    ref_recall = recall_score(y_true, adapt_preds, zero_division=0)
    ref_precision = precision_score(y_true, adapt_preds, zero_division=0)

    rows = []
    for name, fset, th in configs:
        p_vals = probas[name]
        preds = (p_vals >= th).astype(int)

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
        pr_auc = average_precision_score(y_true, p_vals) if "Naive" not in name else np.nan
        brier = brier_score_loss(y_true, p_vals)
        ci = ci_map[name]

        if "Naive" not in name:
            p_at_ref_r, r_at_ref_p = compute_equal_operating_points(y_true, p_vals, ref_recall, ref_precision)
        else:
            p_at_ref_r, r_at_ref_p = np.nan, np.nan

        rows.append({
            "model": name,
            "feature_set": fset,
            "frozen_threshold": th,
            "precision": prec,
            "recall": rec,
            "f2_score": f2,
            "f2_ci_95": f"[{ci['f2_ci_low']:.3f}, {ci['f2_ci_high']:.3f}]",
            "pr_auc": pr_auc,
            "pr_auc_ci_95": f"[{ci['prauc_ci_low']:.3f}, {ci['prauc_ci_high']:.3f}]" if not np.isnan(ci["prauc_ci_low"]) else "-",
            "brier_score": brier,
            "prec_at_adapt_recall": p_at_ref_r,
            "rec_at_adapt_prec": r_at_ref_p,
        })

    df_res = pd.DataFrame(rows)
    best_base_f2 = df_res[df_res["model"].str.contains("Baseline")]["f2_score"].max()
    best_base_name = df_res.loc[df_res["f2_score"] == best_base_f2, "model"].iloc[0]
    best_base_prauc = df_res[df_res["model"].str.contains("Baseline")]["pr_auc"].max()

    df_res["lift_f2_over_best_baseline_pct"] = ((df_res["f2_score"] - best_base_f2) / best_base_f2) * 100
    df_res["lift_prauc_over_best_baseline_pct"] = ((df_res["pr_auc"] - best_base_prauc) / best_base_prauc) * 100
    return df_res, paired_diffs, best_base_name


def run_test_evaluation():
    """Проводить тестування всіх моделей (A, B, C) на Test і останніх 60 днях та зберігає звіти."""
    os.makedirs("reports/figures", exist_ok=True)
    test_df, thresholds, models = load_test_data_and_models()

    y_test = test_df["target_onset_next_1h"].values
    dates_test = test_df["date_kyiv"].values
    n_unique_days = len(np.unique(dates_test))
    print(f"Розмір Test: {len(test_df):,} годин ({n_unique_days} діб), позитивних: {y_test.sum():,} ({y_test.mean():.4%})")

    probas = {
        "Baseline 1: True Naive Persistence": test_df["lagged_onset_1h"].values.astype(float),
        "Baseline 2: Historical Lookup Table": test_df["lookup_prob"].values,
        "Baseline 3: Adaptive 14-day Rolling": test_df["adaptive_rate_14d"].values,
        "LogReg (Set A: No Spatial)": models["lr_a"].predict_proba(models["scaler_a"].transform(test_df[FEATURE_SET_A]))[:, 1],
        "LogReg (Set B: With Spatial)": models["lr_b"].predict_proba(models["scaler_b"].transform(test_df[FEATURE_SET_B]))[:, 1],
        "LogReg (Set C: Spatial + Adaptive)": models["lr_c"].predict_proba(models["scaler_c"].transform(test_df[FEATURE_SET_C]))[:, 1],
        "LightGBM (Set A: No Spatial)": models["lgb_a"].predict_proba(test_df[FEATURE_SET_A])[:, 1],
        "LightGBM (Set B: With Spatial)": models["lgb_b"].predict_proba(test_df[FEATURE_SET_B])[:, 1],
        "LightGBM (Set C: Spatial + Adaptive)": models["lgb_c"].predict_proba(test_df[FEATURE_SET_C])[:, 1],
    }

    configs = [
        ("Baseline 1: True Naive Persistence", "None", 0.500),
        ("Baseline 2: Historical Lookup Table", "None", thresholds["Baseline 2: Historical Lookup Table"]),
        ("Baseline 3: Adaptive 14-day Rolling", "None", thresholds["Baseline 3: Adaptive 14-day Rolling"]),
        ("LogReg (Set A: No Spatial)", "A (Time, Burden & Cluster)", thresholds["LogReg (Set A: No Spatial)"]),
        ("LogReg (Set B: With Spatial)", "B (A + Spatial at T)", thresholds["LogReg (Set B: With Spatial)"]),
        ("LogReg (Set C: Spatial + Adaptive)", "C (B + Adaptive & Lags)", thresholds["LogReg (Set C: Spatial + Adaptive)"]),
        ("LightGBM (Set A: No Spatial)", "A (Time, Burden & Cluster)", thresholds["LightGBM (Set A: No Spatial)"]),
        ("LightGBM (Set B: With Spatial)", "B (A + Spatial at T)", thresholds["LightGBM (Set B: With Spatial)"]),
        ("LightGBM (Set C: Spatial + Adaptive)", "C (B + Adaptive & Lags)", thresholds["LightGBM (Set C: Spatial + Adaptive)"]),
    ]

    print("🔄 Запуск парного Day-Block Bootstrap (500 ітерацій за добами) на повному Test...")
    df_results, paired_diffs, best_base_name = evaluate_slice(
        dates_test, y_test, probas, configs, n_bootstraps=500
    )
    df_results.to_csv("reports/ml_models_benchmark.csv", index=False)

    print("\n" + "=" * 110)
    print("ФІНАЛЬНИЙ БЕНЧМАРК НА ТЕСТОВІЙ ВИБІРЦІ (TEST: 2025-06 — 2026-09, Day-Block Bootstrap N=500):")
    print("=" * 110)
    cols_show = [
        "model", "frozen_threshold", "precision", "recall",
        "f2_score", "f2_ci_95", "pr_auc", "pr_auc_ci_95",
        "brier_score", "prec_at_adapt_recall",
    ]
    print(df_results[cols_show].to_string(index=False))

    print("\n📊 Заздалегідь зафіксоване головне порівняння та Ablation (Day-Block Bootstrap N=500 на Test):")
    for k, v in paired_diffs.items():
        print(f"   {k}: mean={v['mean']:+.4f}, 95% CI=[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")

    # Distribution of predicted probabilities for LightGBM (Set C)
    lgb_p_c = probas["LightGBM (Set C: Spatial + Adaptive)"]
    quantiles_c = np.quantile(lgb_p_c, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print("\n📈 Квантилі прогнозованих імовірностей LightGBM (Set C: Spatial + Adaptive) на Test:")
    print(
        f"   Q10={quantiles_c[0]:.4f}, Q25={quantiles_c[1]:.4f}, Median={quantiles_c[2]:.4f}, "
        f"Q75={quantiles_c[3]:.4f}, Q90={quantiles_c[4]:.4f}, Q95={quantiles_c[5]:.4f}, Q99={quantiles_c[6]:.4f}"
    )

    # Оцінка на останніх 60 днях
    test_max_time = test_df["timestamp_utc"].max()
    recent_cutoff = test_max_time - pd.Timedelta(days=60)
    recent_mask = (test_df["timestamp_utc"] >= recent_cutoff).values
    y_recent = y_test[recent_mask]
    dates_recent = dates_test[recent_mask]
    probas_recent = {k: v[recent_mask] for k, v in probas.items()}

    print(
        f"\n⏱️ Оцінка на останніх 60 днях ({recent_cutoff.date()} — {test_max_time.date()}, "
        f"рядків: {recent_mask.sum():,}, стартів: {y_recent.sum():,}, базова частка: {y_recent.mean():.4%}):"
    )
    df_recent, recent_diffs, _ = evaluate_slice(
        dates_recent, y_recent, probas_recent, configs, n_bootstraps=500
    )
    df_recent.to_csv("reports/ml_recent_60d_benchmark.csv", index=False)
    print(df_recent[cols_show].to_string(index=False))

    print("\n📊 Парні різниці на останніх 60 днях (Day-Block Bootstrap N=500):")
    for k, v in recent_diffs.items():
        print(f"   {k}: mean={v['mean']:+.4f}, 95% CI=[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")

    stats_summary = {
        "test_total_rows": int(len(test_df)),
        "test_total_days": int(n_unique_days),
        "test_positives": int(y_test.sum()),
        "test_base_rate": float(y_test.mean()),
        "recent_60d_rows": int(recent_mask.sum()),
        "recent_60d_positives": int(y_recent.sum()),
        "recent_60d_base_rate": float(y_recent.mean()),
        "paired_day_bootstrap_500_test": paired_diffs,
        "paired_day_bootstrap_500_recent_60d": recent_diffs,
        "lgb_set_c_prob_quantiles": {
            "q10": float(quantiles_c[0]),
            "q25": float(quantiles_c[1]),
            "q50": float(quantiles_c[2]),
            "q75": float(quantiles_c[3]),
            "q90": float(quantiles_c[4]),
            "q95": float(quantiles_c[5]),
            "q99": float(quantiles_c[6]),
        },
    }
    with open("reports/test_evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    # Feature Importance для головної моделі LightGBM (Set C)
    lgb_c_model = models["lgb_c"]
    importance_gain = lgb_c_model.booster_.feature_importance(importance_type="gain")
    importance_split = lgb_c_model.booster_.feature_importance(importance_type="split")

    feat_names_ua = {
        "adaptive_rate_7d": "Адаптивна частота за 7 діб",
        "adaptive_rate_14d": "Адаптивна частота за 14 діб",
        "adaptive_rate_30d": "Адаптивна частота за 30 діб",
        "national_active_count": "Активних областей по країні о T",
        "onsets_last_24h": "Кількість стартів за 24 год",
        "hours_since_last_alert": "Годин від кінця останньої тривоги",
        "neighbors_active_count": "Кількість активних сусідів о T",
        "neighbor_any_active": "Чи активний бодай один сусід о T",
        "neighbor_active_ratio": "Частка активних сусідів о T",
        "neighbors_active_lag_1h": "Активних сусідів о T-1 год",
        "neighbors_active_lag_2h": "Активних сусідів о T-2 год",
        "neighbors_active_lag_3h": "Активних сусідів о T-3 год",
        "burden_last_24h": "Навантаження за 24 год (частка активних)",
        "burden_last_6h": "Навантаження за 6 год (частка активних)",
        "cluster_1_low_activity": "Кластер 1 (low_activity, Train)",
        "cluster_2_medium_activity": "Кластер 2 (medium_activity, Train)",
        "cluster_3_high_activity": "Кластер 3 (high_activity, Train)",
        "month": "Місяць (сезонність)",
        "hour_sin": "Година доби (синус)",
        "hour_cos": "Година доби (косинус)",
        "day_of_week": "День тижня",
        "is_weekend": "Вихідний день",
    }

    df_imp = pd.DataFrame({
        "feature": FEATURE_SET_C,
        "feature_ua": [feat_names_ua.get(f, f) for f in FEATURE_SET_C],
        "gain": importance_gain,
        "split": importance_split,
    }).sort_values("gain", ascending=False)

    df_imp.to_csv("reports/feature_importance_lightgbm.csv", index=False)

    plot_test_figures(y_test, probas, y_recent, probas_recent, df_imp, df_results, best_base_name)
    print("Всі графіки та звіти збережено успішно.")


def plot_test_figures(y_test, probas, y_recent, probas_recent, df_imp, df_results, best_base_name):
    """Створює графіки PR-кривих (повний Test + останні 60 днів), Feature Importance, Lift та Reliability Diagram."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Baseline 2: Historical Lookup Table": "#7f7f7f",
        "Baseline 3: Adaptive 14-day Rolling": "#2ca02c",
        "LogReg (Set A: No Spatial)": "#9ecae1",
        "LogReg (Set B: With Spatial)": "#3182bd",
        "LogReg (Set C: Spatial + Adaptive)": "#08519c",
        "LightGBM (Set A: No Spatial)": "#fdae6b",
        "LightGBM (Set B: With Spatial)": "#e6550d",
        "LightGBM (Set C: Spatial + Adaptive)": "#d62728",
    }
    model_order = [
        "Baseline 2: Historical Lookup Table",
        "Baseline 3: Adaptive 14-day Rolling",
        "LogReg (Set A: No Spatial)",
        "LogReg (Set B: With Spatial)",
        "LogReg (Set C: Spatial + Adaptive)",
        "LightGBM (Set A: No Spatial)",
        "LightGBM (Set B: With Spatial)",
        "LightGBM (Set C: Spatial + Adaptive)",
    ]

    # Графік 1: Двопанельні Precision-Recall Curves (Повний Test та Останні 60 днів)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=300)

    base_rate_test = np.mean(y_test)
    ax1.axhline(base_rate_test, color="darkgray", linestyle="--", label=f"Base rate = {base_rate_test:.2%}")
    for name in model_order:
        p_vals = probas[name]
        prec, rec, _ = precision_recall_curve(y_test, p_vals)
        auc_score = average_precision_score(y_test, p_vals)
        ax1.plot(
            rec, prec,
            label=f"{name} (PR-AUC = {auc_score:.4f})",
            color=colors.get(name, "black"),
            linewidth=2.5 if ("Set C" in name or "Adaptive" in name) else 1.6,
        )
    ax1.set_title("А. Повний Test (червень 2025 – вересень 2026)", fontsize=12.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Recall (Повнота виявлення)", fontsize=11)
    ax1.set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 0.70])
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    base_rate_rec = np.mean(y_recent)
    ax2.axhline(base_rate_rec, color="darkgray", linestyle="--", label=f"Base rate = {base_rate_rec:.2%}")
    for name in model_order:
        p_vals = probas_recent[name]
        prec, rec, _ = precision_recall_curve(y_recent, p_vals)
        auc_score = average_precision_score(y_recent, p_vals)
        ax2.plot(
            rec, prec,
            label=f"{name} (PR-AUC = {auc_score:.4f})",
            color=colors.get(name, "black"),
            linewidth=2.5 if ("Set C" in name or "Adaptive" in name) else 1.6,
        )
    ax2.set_title("Б. Останні 60 днів (липень – вересень 2026)", fontsize=12.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Recall (Повнота виявлення)", fontsize=11)
    ax2.set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 0.70])
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle("Precision-Recall криві (Ablation A vs B vs C): Повний Test та останні 60 днів", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("reports/figures/10_test_pr_curves_comparison.png")
    plt.close(fig)

    # Графік 2: Feature Importance (Gain) для LightGBM (Set C)
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)
    top_imp = df_imp.copy()
    palette = []
    for f in top_imp["feature"]:
        if "neighbor" in f or "national" in f:
            palette.append("#d62728")  # Просторові та загальнонаціональні
        elif "adaptive" in f or "hours_since" in f or "onsets_last" in f or "burden" in f:
            palette.append("#1f77b4")  # Адаптивна пам'ять та локальна динаміка
        elif "cluster" in f:
            palette.append("#2ca02c")  # Статичні Train-кластери
        else:
            palette.append("#ff7f0e")  # Календарні цикли

    ax.barh(top_imp["feature_ua"][::-1], (top_imp["gain"] / 1000)[::-1], color=palette[::-1], edgecolor="black", alpha=0.85)
    ax.set_title("Важливість ознак у LightGBM Set C (Gain × 1,000)\nПросторові сигнали, адаптивна пам'ять та календар", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Gain (× 1,000)", fontsize=11)
    fig.tight_layout()
    fig.savefig("reports/figures/11_feature_importance_lightgbm.png")
    plt.close(fig)

    # Графік 3: Lift над НАЙКРАЩИМ бейзлайном (Adaptive 14-day)
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
    ml_models = df_results[df_results["model"].str.contains("LogReg|LightGBM")].copy()
    x = np.arange(len(ml_models))
    width = 0.38

    rects1 = ax.bar(x - width / 2, ml_models["lift_f2_over_best_baseline_pct"], width, label=f"Lift F2 (%) над {best_base_name}", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width / 2, ml_models["lift_prauc_over_best_baseline_pct"], width, label=f"Lift PR-AUC (%) над {best_base_name}", color="#2ca02c", edgecolor="black")

    ax.set_title(f"Відносна різниця метрик порівняно з найкращим бейзлайном ({best_base_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Різниця (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ml_models["model"], rotation=18, ha="right", fontsize=9.5)
    ax.axhline(0, color="gray", linestyle="--")
    ax.legend(frameon=True, fontsize=10)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:+.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:+.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig("reports/figures/12_lift_over_baselines.png")
    plt.close(fig)

    # Графік 4: Reliability Diagram (Калібрувальна крива для LightGBM Set B та Set C)
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=300)
    for m_label, color, marker in [
        ("LightGBM (Set B: With Spatial)", "#e6550d", "s"),
        ("LightGBM (Set C: Spatial + Adaptive)", "#d62728", "o"),
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probas[m_label], n_bins=10)
        ax.plot(prob_pred, prob_true, marker=marker, linewidth=2, color=color, label=m_label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ідеальне калібрування")
    ax.set_title("Reliability Diagram (Калібрування ймовірностей LightGBM на Test)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Прогнозована ймовірність", fontsize=11)
    ax.set_ylabel("Фактична частка подій", fontsize=11)
    ax.set_xlim([0.0, 0.70])
    ax.set_ylim([0.0, 0.70])
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig("reports/figures/13_calibration_reliability_diagram.png")
    plt.close(fig)


if __name__ == "__main__":
    run_test_evaluation()
