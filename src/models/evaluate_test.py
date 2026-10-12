"""Модуль фінальної оцінки на вибірці Test (повний Test, до/після 2025-08-01 та останні 60 днів) для A, B, C, C'.

Реалізує:
1. Заздалегідь зафіксоване головне порівняння: LightGBM (Set C) проти Adaptive 14-day Rolling Baseline.
2. Додатковий експеримент Set C' (B + adaptive_rate_7d/14d/30d + neighbors_active_lag_1h/2h/3h, без national_active_count,
   hours_since_last_alert та onsets_last_24h) для перевірки гіпотези про зсув розподілу ознак.
3. Оцінку на 4 часових зрізах Test із замороженими порогами p* з Validation:
   - Повний Test (2025-06-01 — 2026-09-07) -> reports/ml_models_benchmark.csv
   - Test до 2025-08-01 (2025-06-01 — 2025-07-31) -> reports/ml_test_pre_aug2025_benchmark.csv
   - Test з 2025-08-01 (2025-08-01 — 2026-09-07) -> reports/ml_test_post_aug2025_benchmark.csv
   - Останні 60 днів (2026-07-09 — 2026-09-07) -> reports/ml_recent_60d_benchmark.csv
4. Парний Block Bootstrap за календарними добами (n_bootstraps=500) для 95% CI PR-AUC, F2 та парних різниць.
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

from src.models.train_models import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_C_PRIME,
)


def load_test_data_and_models(parquet_path: str = "data/processed/ml_features_hourly.parquet"):
    """Завантажує валідаційний та тестовий набори даних, таблицю бейзлайну та натреновані моделі (A, B, C, C')."""
    print("Завантаження даних та моделей...")
    df = pd.read_parquet(parquet_path)
    train_df = df[df["split_set"] == "train"].copy()
    val_df = df[df["split_set"] == "val"].copy()
    test_df = df[df["split_set"] == "test"].copy()

    with open("models/optimal_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Таблиця шансів Historical Lookup з Train (Київський час!)
    train_df["hour_kyiv"] = train_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    val_df["hour_kyiv"] = val_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    val_df["date_kyiv"] = val_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.date
    test_df["hour_kyiv"] = test_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    test_df["date_kyiv"] = test_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.date

    lookup = (
        train_df.groupby(["day_of_week", "hour_kyiv", "cluster_id"])["target_onset_next_1h"]
        .mean()
        .reset_index()
    )
    lookup.rename(columns={"target_onset_next_1h": "lookup_prob"}, inplace=True)
    global_mean = train_df["target_onset_next_1h"].mean()

    val_merged = val_df.merge(lookup, on=["day_of_week", "hour_kyiv", "cluster_id"], how="left")
    val_merged["lookup_prob"] = val_merged["lookup_prob"].fillna(global_mean)

    test_merged = test_df.merge(lookup, on=["day_of_week", "hour_kyiv", "cluster_id"], how="left")
    test_merged["lookup_prob"] = test_merged["lookup_prob"].fillna(global_mean)

    models = {
        "scaler_a": joblib.load("models/scaler_nospatial.joblib"),
        "scaler_b": joblib.load("models/scaler_spatial.joblib"),
        "scaler_c": joblib.load("models/scaler_adaptive.joblib"),
        "scaler_cp": joblib.load("models/scaler_c_prime.joblib"),
        "lr_a": joblib.load("models/logreg_nospatial.joblib"),
        "lr_b": joblib.load("models/logreg_spatial.joblib"),
        "lr_c": joblib.load("models/logreg_adaptive.joblib"),
        "lr_cp": joblib.load("models/logreg_c_prime.joblib"),
        "lgb_a": joblib.load("models/lightgbm_nospatial.joblib"),
        "lgb_b": joblib.load("models/lightgbm_spatial.joblib"),
        "lgb_c": joblib.load("models/lightgbm_adaptive.joblib"),
        "lgb_cp": joblib.load("models/lightgbm_c_prime.joblib"),
    }

    return val_merged, test_merged, thresholds, models


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

    # 2. Точний швидкий розрахунок PR-AUC на тих самих блоках діб (попереднє сортування 1 раз на модель)
    day_weights = np.array(
        [np.bincount(sampled_days[b], minlength=n_days) for b in range(n_bootstraps)],
        dtype=np.float64,
    )
    non_naive_names = [name for name in probas_dict if "Naive" not in name]
    y_true_f = y_true.astype(np.float64)

    def _compute_model_prauc_boot(p_vals: np.ndarray) -> np.ndarray:
        order = np.argsort(p_vals, kind="mergesort")[::-1]
        p_s = p_vals[order]
        y_s = y_true_f[order]
        d_s = day_inverse[order]
        thresh_idx = np.where(np.r_[np.diff(p_s) != 0, True])[0]

        out = np.zeros(n_bootstraps, dtype=np.float64)
        for b in range(n_bootstraps):
            w_s = day_weights[b, d_s]
            tp_cum = np.cumsum(y_s * w_s)[thresh_idx]
            if tp_cum[-1] <= 0:
                continue
            tot_cum = np.cumsum(w_s)[thresh_idx]
            keep = np.r_[tot_cum[0] > 0, np.diff(tot_cum) > 0]
            tps = tp_cum[keep]
            tots = tot_cum[keep]
            prec = tps / tots
            rec = tps / tps[-1]
            d_rec = np.r_[rec[0], np.diff(rec)]
            out[b] = float(np.sum(d_rec * prec))
        return out

    prauc_list = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(_compute_model_prauc_boot)(probas_dict[name]) for name in non_naive_names
    )
    prauc_samples = dict(zip(non_naive_names, prauc_list))

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
    lgb_a, lgb_b, lgb_c, lgb_cp = (
        "LightGBM (Set A: No Spatial)",
        "LightGBM (Set B: With Spatial)",
        "LightGBM (Set C: Spatial + Adaptive)",
        "LightGBM (Set C': Adaptive + Lags)",
    )
    lr_a, lr_b, lr_c, lr_cp = (
        "LogReg (Set A: No Spatial)",
        "LogReg (Set B: With Spatial)",
        "LogReg (Set C: Spatial + Adaptive)",
        "LogReg (Set C': Adaptive + Lags)",
    )

    paired_diffs = {
        # Головне заздалегідь зафіксоване порівняння: LightGBM (Set C) vs Adaptive 14-day
        "PRIMARY_lgb_C_minus_adaptive_prauc": _summarize_diff(prauc_samples[lgb_c] - prauc_samples[adapt]),
        "PRIMARY_lgb_C_minus_adaptive_f2": _summarize_diff(f2_samples[lgb_c] - f2_samples[adapt]),
        # LightGBM (Set C') vs Adaptive 14-day та vs Set B і Set C
        "lgb_Cprime_minus_adaptive_prauc": _summarize_diff(prauc_samples[lgb_cp] - prauc_samples[adapt]),
        "lgb_Cprime_minus_adaptive_f2": _summarize_diff(f2_samples[lgb_cp] - f2_samples[adapt]),
        "lgb_Cprime_minus_B_prauc": _summarize_diff(prauc_samples[lgb_cp] - prauc_samples[lgb_b]),
        "lgb_Cprime_minus_B_f2": _summarize_diff(f2_samples[lgb_cp] - f2_samples[lgb_b]),
        "lgb_Cprime_minus_C_prauc": _summarize_diff(prauc_samples[lgb_cp] - prauc_samples[lgb_c]),
        "lgb_Cprime_minus_C_f2": _summarize_diff(f2_samples[lgb_cp] - f2_samples[lgb_c]),
        # LogReg (Set C та Set C') vs Adaptive 14-day та vs Set B
        "lr_C_minus_adaptive_prauc": _summarize_diff(prauc_samples[lr_c] - prauc_samples[adapt]),
        "lr_C_minus_adaptive_f2": _summarize_diff(f2_samples[lr_c] - f2_samples[adapt]),
        "lr_Cprime_minus_adaptive_prauc": _summarize_diff(prauc_samples[lr_cp] - prauc_samples[adapt]),
        "lr_Cprime_minus_adaptive_f2": _summarize_diff(f2_samples[lr_cp] - f2_samples[adapt]),
        "lr_Cprime_minus_B_prauc": _summarize_diff(prauc_samples[lr_cp] - prauc_samples[lr_b]),
        "lr_Cprime_minus_B_f2": _summarize_diff(f2_samples[lr_cp] - f2_samples[lr_b]),
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
    fixed_ref_recall: float | None = None,
) -> tuple[pd.DataFrame, dict, str]:
    """Оцінює всі моделі на заданому часовому зрізі."""
    th_map = {name: th for name, _, th in configs}
    ci_map, paired_diffs = day_block_bootstrap_all_models(
        dates_array, y_true, probas, th_map, n_bootstraps=n_bootstraps
    )

    adapt_name = "Baseline 3: Adaptive 14-day Rolling"
    adapt_preds = (probas[adapt_name] >= th_map[adapt_name]).astype(int)
    ref_recall = float(fixed_ref_recall) if fixed_ref_recall is not None else float(recall_score(y_true, adapt_preds, zero_division=0))
    ref_precision = float(precision_score(y_true, adapt_preds, zero_division=0))

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
    """Проводить оцінку всіх моделей (A, B, C, C') на Validation та 4 тестових зрізах із Day-Block Bootstrap (N=500)."""
    os.makedirs("reports/official/figures", exist_ok=True)
    val_df, test_df, thresholds, models = load_test_data_and_models()

    y_val = val_df["target_onset_next_1h"].values
    dates_val = val_df["date_kyiv"].values
    n_val_days = len(np.unique(dates_val))
    print(f"Розмір Val:  {len(val_df):,} годин ({n_val_days} діб), позитивних: {y_val.sum():,} ({y_val.mean():.4%})")

    y_test = test_df["target_onset_next_1h"].values
    dates_test = test_df["date_kyiv"].values
    n_unique_days = len(np.unique(dates_test))
    print(f"Розмір Test: {len(test_df):,} годин ({n_unique_days} діб), позитивних: {y_test.sum():,} ({y_test.mean():.4%})")

    probas_val = {
        "Baseline 1: True Naive Persistence": val_df["lagged_onset_1h"].values.astype(float),
        "Baseline 2: Historical Lookup Table": val_df["lookup_prob"].values,
        "Baseline 3: Adaptive 14-day Rolling": val_df["adaptive_rate_14d"].values,
        "LogReg (Set A: No Spatial)": models["lr_a"].predict_proba(models["scaler_a"].transform(val_df[FEATURE_SET_A]))[:, 1],
        "LogReg (Set B: With Spatial)": models["lr_b"].predict_proba(models["scaler_b"].transform(val_df[FEATURE_SET_B]))[:, 1],
        "LogReg (Set C: Spatial + Adaptive)": models["lr_c"].predict_proba(models["scaler_c"].transform(val_df[FEATURE_SET_C]))[:, 1],
        "LogReg (Set C': Adaptive + Lags)": models["lr_cp"].predict_proba(models["scaler_cp"].transform(val_df[FEATURE_SET_C_PRIME]))[:, 1],
        "LightGBM (Set A: No Spatial)": models["lgb_a"].predict_proba(val_df[FEATURE_SET_A])[:, 1],
        "LightGBM (Set B: With Spatial)": models["lgb_b"].predict_proba(val_df[FEATURE_SET_B])[:, 1],
        "LightGBM (Set C: Spatial + Adaptive)": models["lgb_c"].predict_proba(val_df[FEATURE_SET_C])[:, 1],
        "LightGBM (Set C': Adaptive + Lags)": models["lgb_cp"].predict_proba(val_df[FEATURE_SET_C_PRIME])[:, 1],
    }

    probas = {
        "Baseline 1: True Naive Persistence": test_df["lagged_onset_1h"].values.astype(float),
        "Baseline 2: Historical Lookup Table": test_df["lookup_prob"].values,
        "Baseline 3: Adaptive 14-day Rolling": test_df["adaptive_rate_14d"].values,
        "LogReg (Set A: No Spatial)": models["lr_a"].predict_proba(models["scaler_a"].transform(test_df[FEATURE_SET_A]))[:, 1],
        "LogReg (Set B: With Spatial)": models["lr_b"].predict_proba(models["scaler_b"].transform(test_df[FEATURE_SET_B]))[:, 1],
        "LogReg (Set C: Spatial + Adaptive)": models["lr_c"].predict_proba(models["scaler_c"].transform(test_df[FEATURE_SET_C]))[:, 1],
        "LogReg (Set C': Adaptive + Lags)": models["lr_cp"].predict_proba(models["scaler_cp"].transform(test_df[FEATURE_SET_C_PRIME]))[:, 1],
        "LightGBM (Set A: No Spatial)": models["lgb_a"].predict_proba(test_df[FEATURE_SET_A])[:, 1],
        "LightGBM (Set B: With Spatial)": models["lgb_b"].predict_proba(test_df[FEATURE_SET_B])[:, 1],
        "LightGBM (Set C: Spatial + Adaptive)": models["lgb_c"].predict_proba(test_df[FEATURE_SET_C])[:, 1],
        "LightGBM (Set C': Adaptive + Lags)": models["lgb_cp"].predict_proba(test_df[FEATURE_SET_C_PRIME])[:, 1],
    }

    configs = [
        ("Baseline 1: True Naive Persistence", "None", 0.500),
        ("Baseline 2: Historical Lookup Table", "None", thresholds["Baseline 2: Historical Lookup Table"]),
        ("Baseline 3: Adaptive 14-day Rolling", "None", thresholds["Baseline 3: Adaptive 14-day Rolling"]),
        ("LogReg (Set A: No Spatial)", "A (Time, Burden & Cluster)", thresholds["LogReg (Set A: No Spatial)"]),
        ("LogReg (Set B: With Spatial)", "B (A + Spatial at T)", thresholds["LogReg (Set B: With Spatial)"]),
        ("LogReg (Set C: Spatial + Adaptive)", "C (B + Adaptive, Nat & Lags)", thresholds["LogReg (Set C: Spatial + Adaptive)"]),
        ("LogReg (Set C': Adaptive + Lags)", "C' (B + Adaptive & Lags only)", thresholds["LogReg (Set C': Adaptive + Lags)"]),
        ("LightGBM (Set A: No Spatial)", "A (Time, Burden & Cluster)", thresholds["LightGBM (Set A: No Spatial)"]),
        ("LightGBM (Set B: With Spatial)", "B (A + Spatial at T)", thresholds["LightGBM (Set B: With Spatial)"]),
        ("LightGBM (Set C: Spatial + Adaptive)", "C (B + Adaptive, Nat & Lags)", thresholds["LightGBM (Set C: Spatial + Adaptive)"]),
        ("LightGBM (Set C': Adaptive + Lags)", "C' (B + Adaptive & Lags only)", thresholds["LightGBM (Set C': Adaptive + Lags)"]),
    ]

    cols_show = [
        "model", "frozen_threshold", "precision", "recall",
        "f2_score", "f2_ci_95", "pr_auc", "pr_auc_ci_95",
        "brier_score", "prec_at_adapt_recall",
    ]

    # 0. VALIDATION (Day-Block Bootstrap N=500 + формальний вибір моделі за PR-AUC на Val)
    print("🔄 0. Запуск парного Day-Block Bootstrap (500 ітерацій за добами) на Validation...")
    df_val_results, val_paired_diffs, _ = evaluate_slice(
        dates_val, y_val, probas_val, configs, n_bootstraps=500
    )
    df_val_results["threshold"] = df_val_results["frozen_threshold"]
    prespec_mask = ~df_val_results["model"].str.contains("Set C'")
    val_winner_idx = df_val_results.loc[prespec_mask, "pr_auc"].idxmax()
    val_winner_model = str(df_val_results.loc[val_winner_idx, "model"])

    def _status_label(m: str) -> str:
        if m == val_winner_model:
            return "selected on Val by PR-AUC (rule formulated after initial Test inspection)"
        if "Set C'" in m:
            return "post-hoc sensitivity experiment (Set C')"
        return "candidate model (Val PR-AUC selection rule formulated after initial Test inspection)"

    df_val_results["selection_status"] = df_val_results["model"].map(_status_label)
    df_val_results.to_csv("reports/official/ml_validation_benchmark.csv", index=False)

    val_summary = {
        "val_total_rows": int(len(val_df)),
        "val_total_days": int(n_val_days),
        "val_positives": int(y_val.sum()),
        "val_base_rate": float(y_val.mean()),
        "selection_rule": "arg max PR-AUC on Validation among candidate models (rule formulated after initial Test inspection) (Sets A, B, C)",
        "early_stopping_subsplit": "Internal chronological split at tail of Train (last 60 days of Train with 2h embargo after Train_Fit; Val 100% untouched by early stopping)",
        "selected_model_on_val": val_winner_model,
        "selected_model_val_pr_auc": float(df_val_results.loc[val_winner_idx, "pr_auc"]),
        "selected_model_val_f2": float(df_val_results.loc[val_winner_idx, "f2_score"]),
        "paired_day_bootstrap_500_val": val_paired_diffs,
    }
    with open("reports/official/val_evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 115)
    print("0. БЕНЧМАРК НА ВАЛІДАЦІЙНІЙ ВИБІРЦІ (VAL: 2024-05 — 2025-05, Day-Block Bootstrap N=500):")
    print(f"   🏆 Переможець за правилом arg max Val PR-AUC: {val_winner_model}")
    print("=" * 115)
    print(df_val_results[cols_show + ["selection_status"]].to_string(index=False))

    # 1. ПОВНИЙ TEST
    print("\n🔄 1. Запуск парного Day-Block Bootstrap (500 ітерацій за добами) на повному Test...")
    df_results, paired_diffs, best_base_name = evaluate_slice(
        dates_test, y_test, probas, configs, n_bootstraps=500, fixed_ref_recall=None
    )
    df_results["selection_status"] = df_results["model"].map(_status_label)
    df_results.to_csv("reports/official/ml_models_benchmark.csv", index=False)

    print("\n" + "=" * 115)
    print("1. ФІНАЛЬНИЙ БЕНЧМАРК НА ПОВНІЙ ТЕСТОВІЙ ВИБІРЦІ (TEST: 2025-06 — 2026-09, Day-Block Bootstrap N=500):")
    print("=" * 115)
    print(df_results[cols_show + ["selection_status"]].to_string(index=False))

    print("\n📊 Парні різниці на повному Test (Day-Block Bootstrap N=500):")
    for k, v in paired_diffs.items():
        print(f"   {k}: mean={v['mean']:+.4f}, 95% CI=[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")

    # 2. TEST ДО 2025-08-01
    aug_cutoff = pd.Timestamp("2025-08-01 00:00:00", tz="UTC")
    pre_mask = (test_df["timestamp_utc"] < aug_cutoff).values
    y_pre = y_test[pre_mask]
    dates_pre = dates_test[pre_mask]
    probas_pre = {k: v[pre_mask] for k, v in probas.items()}

    print(
        f"\n⏱️ 2. Оцінка на Test ДО 2025-08-01 (2025-06-01 — 2025-07-31, "
        f"рядків: {pre_mask.sum():,}, діб: {len(np.unique(dates_pre))}, стартів: {y_pre.sum():,}, базова частка: {y_pre.mean():.4%}):"
    )
    df_pre, pre_diffs, _ = evaluate_slice(
        dates_pre, y_pre, probas_pre, configs, n_bootstraps=500, fixed_ref_recall=None
    )
    df_pre.to_csv("reports/official/ml_test_pre_aug2025_benchmark.csv", index=False)
    print(df_pre[cols_show].to_string(index=False))

    # 3. TEST ПІСЛЯ 2025-08-01
    post_mask = (test_df["timestamp_utc"] >= aug_cutoff).values
    y_post = y_test[post_mask]
    dates_post = dates_test[post_mask]
    probas_post = {k: v[post_mask] for k, v in probas.items()}

    print(
        f"\n⏱️ 3. Оцінка на Test ПІСЛЯ 2025-08-01 (2025-08-01 — 2026-09-07, "
        f"рядків: {post_mask.sum():,}, діб: {len(np.unique(dates_post))}, стартів: {y_post.sum():,}, базова частка: {y_post.mean():.4%}):"
    )
    df_post, post_diffs, _ = evaluate_slice(
        dates_post, y_post, probas_post, configs, n_bootstraps=500, fixed_ref_recall=None
    )
    df_post.to_csv("reports/official/ml_test_post_aug2025_benchmark.csv", index=False)
    print(df_post[cols_show].to_string(index=False))

    # 4. ОСТАННІ 60 ДНІВ
    test_max_time = test_df["timestamp_utc"].max()
    recent_cutoff = test_max_time - pd.Timedelta(days=60)
    recent_mask = (test_df["timestamp_utc"] >= recent_cutoff).values
    y_recent = y_test[recent_mask]
    dates_recent = dates_test[recent_mask]
    probas_recent = {k: v[recent_mask] for k, v in probas.items()}

    print(
        f"\n⏱️ 4. Оцінка на останніх 60 днях ({recent_cutoff.date()} — {test_max_time.date()}, "
        f"рядків: {recent_mask.sum():,}, стартів: {y_recent.sum():,}, базова частка: {y_recent.mean():.4%}):"
    )
    df_recent, recent_diffs, _ = evaluate_slice(
        dates_recent, y_recent, probas_recent, configs, n_bootstraps=500, fixed_ref_recall=None
    )
    df_recent.to_csv("reports/official/ml_recent_60d_benchmark.csv", index=False)
    print(df_recent[cols_show].to_string(index=False))

    lgb_p_c = probas["LightGBM (Set C: Spatial + Adaptive)"]
    quantiles_c = np.quantile(lgb_p_c, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    stats_summary = {
        "test_total_rows": int(len(test_df)),
        "test_total_days": int(n_unique_days),
        "test_positives": int(y_test.sum()),
        "test_base_rate": float(y_test.mean()),
        "test_pre_aug2025_rows": int(pre_mask.sum()),
        "test_pre_aug2025_positives": int(y_pre.sum()),
        "test_pre_aug2025_base_rate": float(y_pre.mean()),
        "test_post_aug2025_rows": int(post_mask.sum()),
        "test_post_aug2025_positives": int(y_post.sum()),
        "test_post_aug2025_base_rate": float(y_post.mean()),
        "recent_60d_rows": int(recent_mask.sum()),
        "recent_60d_positives": int(y_recent.sum()),
        "recent_60d_base_rate": float(y_recent.mean()),
        "selected_model_on_val": val_winner_model,
        "paired_day_bootstrap_500_test": paired_diffs,
        "paired_day_bootstrap_500_pre_aug2025": pre_diffs,
        "paired_day_bootstrap_500_post_aug2025": post_diffs,
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
    with open("reports/official/test_evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    # Feature Importance для LightGBM (Set C)
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

    df_imp.to_csv("reports/official/feature_importance_lightgbm.csv", index=False)

    plot_test_figures(
        y_test,
        probas,
        y_recent,
        probas_recent,
        df_imp,
        df_results,
        best_base_name,
        figures_dir="reports/official/figures",
        dataset_tag="",
    )
    print("Всі графіки та звіти збережено успішно.")


def plot_test_figures(
    y_test,
    probas,
    y_recent,
    probas_recent,
    df_imp,
    df_results,
    best_base_name,
    figures_dir: str = "reports/official/figures",
    dataset_tag: str = "",
):
    """Створює графіки PR-кривих (повний Test + останні 60 днів), Feature Importance, Lift та Reliability Diagram."""
    os.makedirs(figures_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    title_suffix = f" [{dataset_tag}]" if dataset_tag else ""

    colors = {
        "Baseline 2: Historical Lookup Table": "#7f7f7f",
        "Baseline 3: Adaptive 14-day Rolling": "#2ca02c",
        "LogReg (Set A: No Spatial)": "#9ecae1",
        "LogReg (Set B: With Spatial)": "#3182bd",
        "LogReg (Set C: Spatial + Adaptive)": "#08519c",
        "LogReg (Set C': Adaptive + Lags)": "#6a51a3",
        "LightGBM (Set A: No Spatial)": "#fdae6b",
        "LightGBM (Set B: With Spatial)": "#e6550d",
        "LightGBM (Set C: Spatial + Adaptive)": "#d62728",
        "LightGBM (Set C': Adaptive + Lags)": "#8c2d04",
    }
    model_order = [
        "Baseline 2: Historical Lookup Table",
        "Baseline 3: Adaptive 14-day Rolling",
        "LogReg (Set A: No Spatial)",
        "LogReg (Set B: With Spatial)",
        "LogReg (Set C: Spatial + Adaptive)",
        "LogReg (Set C': Adaptive + Lags)",
        "LightGBM (Set A: No Spatial)",
        "LightGBM (Set B: With Spatial)",
        "LightGBM (Set C: Spatial + Adaptive)",
        "LightGBM (Set C': Adaptive + Lags)",
    ]

    # Графік 1: Двопанельні Precision-Recall Curves (Повний Test та Останні 60 днів)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.2), dpi=300)

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
            linewidth=2.3 if ("Set C" in name or "Adaptive" in name) else 1.5,
        )
    ax1.set_title(f"А. Повний Test{title_suffix}", fontsize=12.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Recall (Повнота виявлення)", fontsize=11)
    ax1.set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.legend(loc="upper right", frameon=True, fontsize=8.0)

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
            linewidth=2.3 if ("Set C" in name or "Adaptive" in name) else 1.5,
        )
    ax2.set_title(f"Б. Останні 60 днів{title_suffix}", fontsize=12.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Recall (Повнота виявлення)", fontsize=11)
    ax2.set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.legend(loc="upper right", frameon=True, fontsize=8.0)

    fig.suptitle(f"Precision-Recall криві (Ablation A vs B vs C vs C'): Повний Test та останні 60 днів{title_suffix}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "10_test_pr_curves_comparison.png"))
    plt.close(fig)

    # Графік 2: Feature Importance (Gain) для LightGBM (Set C)
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)
    top_imp = df_imp.copy()
    palette = []
    for f in top_imp["feature"]:
        if "neighbor" in f or "national" in f:
            palette.append("#d62728")
        elif "adaptive" in f or "hours_since" in f or "onsets_last" in f or "burden" in f:
            palette.append("#1f77b4")
        elif "cluster" in f:
            palette.append("#2ca02c")
        else:
            palette.append("#ff7f0e")

    ax.barh(top_imp["feature_ua"][::-1], (top_imp["gain"] / 1000)[::-1], color=palette[::-1], edgecolor="black", alpha=0.85)
    ax.set_title(f"Важливість ознак у LightGBM Set C (Gain × 1,000){title_suffix}\nПросторові сигнали, адаптивна пам'ять та календар", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Gain (× 1,000)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "11_feature_importance_lightgbm.png"))
    plt.close(fig)

    # Графік 3: Lift над НАЙКРАЩИМ бейзлайном (Adaptive 14-day)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    ml_models = df_results[df_results["model"].str.contains("LogReg|LightGBM")].copy()
    x = np.arange(len(ml_models))
    width = 0.38

    rects1 = ax.bar(x - width / 2, ml_models["lift_f2_over_best_baseline_pct"], width, label=f"Lift F2 (%) над {best_base_name}", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width / 2, ml_models["lift_prauc_over_best_baseline_pct"], width, label=f"Lift PR-AUC (%) над {best_base_name}", color="#2ca02c", edgecolor="black")

    ax.set_title(f"Відносна різниця метрик порівняно з найкращим бейзлайном ({best_base_name}){title_suffix}", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Різниця (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ml_models["model"], rotation=20, ha="right", fontsize=9)
    min_val = float(min(0.0, ml_models["lift_f2_over_best_baseline_pct"].min(), ml_models["lift_prauc_over_best_baseline_pct"].min()))
    max_val = float(max(10.0, ml_models["lift_f2_over_best_baseline_pct"].max(), ml_models["lift_prauc_over_best_baseline_pct"].max()))
    ax.set_ylim([min_val - 5.0 if min_val < 0 else 0.0, max_val + 6.0])
    ax.axhline(0, color="gray", linestyle="--")
    ax.legend(frameon=True, fontsize=10)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:+.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:+.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "12_lift_over_baselines.png"))
    plt.close(fig)

    # Графік 4: Reliability Diagram (Калібрувальна крива для LightGBM Set B, Set C, Set C')
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=300)
    for m_label, color, marker in [
        ("LightGBM (Set B: With Spatial)", "#e6550d", "s"),
        ("LightGBM (Set C: Spatial + Adaptive)", "#d62728", "o"),
        ("LightGBM (Set C': Adaptive + Lags)", "#8c2d04", "^"),
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probas[m_label], n_bins=10)
        ax.plot(prob_pred, prob_true, marker=marker, linewidth=2, color=color, label=m_label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ідеальне калібрування")
    ax.set_title(f"Reliability Diagram (Калібрування ймовірностей LightGBM на Test){title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Прогнозована ймовірність", fontsize=11)
    ax.set_ylabel("Фактична частка подій", fontsize=11)
    ax.set_xlim([0.0, 0.70])
    ax.set_ylim([0.0, 0.70])
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "13_calibration_reliability_diagram.png"))
    plt.close(fig)


if __name__ == "__main__":
    run_test_evaluation()
