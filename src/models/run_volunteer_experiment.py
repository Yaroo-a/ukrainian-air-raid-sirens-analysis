"""Повний експериментальний пайплайн для другого незалежного джерела даних:
volunteer_data_uk.csv (волонтерський датасет eTryvoga).

Реалізує:
1. Train-only кластеризацію (2022-03-15 .. 2024-04-30) та побудову Feature Store для 3 каузальних варіантів обробки naive==True:
   - (a) naive_as_is (за замовчуванням у сирих даних: finished_at = started_at + 30 хв)
   - (b) drop_naive (видалення записів із naive == True)
   - (c) naive_fixed_D (каузальна заміна finished_at = started_at + D_oblast, де D_oblast — медіанна тривалість із Train)
2. Аналіз зсуву розподілу ознак (Train / Val / Test та всередині Test до/після 2025-08-01) на волонтерських даних.
3. Повний протокол навчання та оцінки (Baselines 1-3, LogReg A/B/C/C', LightGBM A/B/C/C', Day-Block Bootstrap N=500
   на Val, Повному Test, До/Після 2025-08-01 та Останніх 60 днях) для базового варіанта (a) naive_as_is.
4. Формальний вибір моделі за правилом arg max Val PR-AUC (серед заздалегідь визначених моделей A, B, C).
5. Аналіз чутливості між варіантами (a) naive_as_is, (b) drop_naive, (c) naive_fixed_D для обраної на Val моделі
   та адаптивного бейзлайну (Adaptive 14-day Rolling).
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.features.cyclicity import analyze_temporal_cyclicity
from src.features.clustering import generate_train_only_clusters, perform_regional_clustering
from src.features.propagation import analyze_spatial_propagation
from src.features.feature_store import generate_feature_store
from src.features.analyze_feature_shift import analyze_feature_shift
from src.models.baselines import run_baselines_evaluation
from src.models.train_models import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_C_PRIME,
    evaluate_threshold_grid,
    train_logistic_regression,
    train_lightgbm_model,
    plot_curves,
)
from src.models.evaluate_test import (
    evaluate_slice,
    day_block_bootstrap_all_models,
    plot_test_figures,
)


def prepare_lookup_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Додає hour_kyiv, date_kyiv та lookup_prob (навчений виключно на train)."""
    out = df.copy()
    kyiv_dt = out["timestamp_utc"].dt.tz_convert("Europe/Kyiv")
    out["hour_kyiv"] = kyiv_dt.dt.hour
    out["date_kyiv"] = kyiv_dt.dt.date

    train_sub = out[out["split_set"] == "train"]
    lookup = (
        train_sub.groupby(["day_of_week", "hour_kyiv", "cluster_id"])["target_onset_next_1h"]
        .mean()
        .reset_index()
        .rename(columns={"target_onset_next_1h": "lookup_prob"})
    )
    global_mean = float(train_sub["target_onset_next_1h"].mean())
    out = out.merge(lookup, on=["day_of_week", "hour_kyiv", "cluster_id"], how="left")
    out["lookup_prob"] = out["lookup_prob"].fillna(global_mean)
    return out


def run_full_protocol_naive_as_is(fs_path: Path):
    """Виконує повний протокол навчання та оцінки для варіанта (a) naive_as_is."""
    print("\n" + "=" * 95)
    print("🚀 ЗАПУСК ПОВНОГО ПРОТОКОЛУ НА ВОЛОНТЕРСЬКИХ ДАНИХ: ВАРІАНТ (a) naive_as_is")
    print("=" * 95)

    vol_dir = PROJECT_ROOT / "reports" / "volunteer"
    vol_fig_dir = vol_dir / "figures"
    vol_dir.mkdir(parents=True, exist_ok=True)
    vol_fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(fs_path)
    df = prepare_lookup_and_dates(df)

    train_df = df[df["split_set"] == "train"].copy()
    val_df = df[df["split_set"] == "val"].copy()
    test_df = df[df["split_set"] == "test"].copy()

    y_train = train_df["target_onset_next_1h"].values
    y_val = val_df["target_onset_next_1h"].values
    y_test = test_df["target_onset_next_1h"].values

    dates_val = val_df["date_kyiv"].values
    dates_test = test_df["date_kyiv"].values

    # 1. Підбір порогів на Validation для бейзлайнів
    best_b2, _ = evaluate_threshold_grid(y_val, val_df["lookup_prob"].values)
    best_b3, _ = evaluate_threshold_grid(y_val, val_df["adaptive_rate_14d"].values)

    # 2. Навчання LogReg та LightGBM (A, B, C, C')
    lr_a, scaler_a = train_logistic_regression(train_df, y_train, FEATURE_SET_A)
    p_val_lr_a = lr_a.predict_proba(scaler_a.transform(val_df[FEATURE_SET_A]))[:, 1]
    best_lr_a, curve_lr_a = evaluate_threshold_grid(y_val, p_val_lr_a)

    lr_b, scaler_b = train_logistic_regression(train_df, y_train, FEATURE_SET_B)
    p_val_lr_b = lr_b.predict_proba(scaler_b.transform(val_df[FEATURE_SET_B]))[:, 1]
    best_lr_b, curve_lr_b = evaluate_threshold_grid(y_val, p_val_lr_b)

    lr_c, scaler_c = train_logistic_regression(train_df, y_train, FEATURE_SET_C)
    p_val_lr_c = lr_c.predict_proba(scaler_c.transform(val_df[FEATURE_SET_C]))[:, 1]
    best_lr_c, curve_lr_c = evaluate_threshold_grid(y_val, p_val_lr_c)

    lr_cp, scaler_cp = train_logistic_regression(train_df, y_train, FEATURE_SET_C_PRIME)
    p_val_lr_cp = lr_cp.predict_proba(scaler_cp.transform(val_df[FEATURE_SET_C_PRIME]))[:, 1]
    best_lr_cp, curve_lr_cp = evaluate_threshold_grid(y_val, p_val_lr_cp)

    lgb_a = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_A)
    p_val_lgb_a = lgb_a.predict_proba(val_df[FEATURE_SET_A])[:, 1]
    best_lgb_a, curve_lgb_a = evaluate_threshold_grid(y_val, p_val_lgb_a)

    lgb_b = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_B)
    p_val_lgb_b = lgb_b.predict_proba(val_df[FEATURE_SET_B])[:, 1]
    best_lgb_b, curve_lgb_b = evaluate_threshold_grid(y_val, p_val_lgb_b)

    lgb_c = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_C)
    p_val_lgb_c = lgb_c.predict_proba(val_df[FEATURE_SET_C])[:, 1]
    best_lgb_c, curve_lgb_c = evaluate_threshold_grid(y_val, p_val_lgb_c)

    lgb_cp = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_C_PRIME)
    p_val_lgb_cp = lgb_cp.predict_proba(val_df[FEATURE_SET_C_PRIME])[:, 1]
    best_lgb_cp, curve_lgb_cp = evaluate_threshold_grid(y_val, p_val_lgb_cp)

    os.makedirs("models", exist_ok=True)
    joblib.dump(lgb_a, "models/volunteer_lightgbm_nospatial.joblib")
    joblib.dump(lgb_b, "models/volunteer_lightgbm_spatial.joblib")
    joblib.dump(lgb_c, "models/volunteer_lightgbm_adaptive.joblib")
    joblib.dump(lgb_cp, "models/volunteer_lightgbm_c_prime.joblib")

    thresholds = {
        "Baseline 1: True Naive Persistence": 0.500,
        "Baseline 2: Historical Lookup Table": float(best_b2["threshold"]),
        "Baseline 3: Adaptive 14-day Rolling": float(best_b3["threshold"]),
        "LogReg (Set A: No Spatial)": float(best_lr_a["threshold"]),
        "LogReg (Set B: With Spatial)": float(best_lr_b["threshold"]),
        "LogReg (Set C: Spatial + Adaptive)": float(best_lr_c["threshold"]),
        "LogReg (Set C': Adaptive + Lags)": float(best_lr_cp["threshold"]),
        "LightGBM (Set A: No Spatial)": float(best_lgb_a["threshold"]),
        "LightGBM (Set B: With Spatial)": float(best_lgb_b["threshold"]),
        "LightGBM (Set C: Spatial + Adaptive)": float(best_lgb_c["threshold"]),
        "LightGBM (Set C': Adaptive + Lags)": float(best_lgb_cp["threshold"]),
    }
    with open("models/volunteer_optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    # Побудова графіків валідації (08 та 09) для волонтерських даних
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
        optimal_thresholds=thresholds,
        figures_dir=str(vol_fig_dir),
        dataset_tag="Volunteer",
    )

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

    probas_val = {
        "Baseline 1: True Naive Persistence": val_df["lagged_onset_1h"].values.astype(float),
        "Baseline 2: Historical Lookup Table": val_df["lookup_prob"].values,
        "Baseline 3: Adaptive 14-day Rolling": val_df["adaptive_rate_14d"].values,
        "LogReg (Set A: No Spatial)": p_val_lr_a,
        "LogReg (Set B: With Spatial)": p_val_lr_b,
        "LogReg (Set C: Spatial + Adaptive)": p_val_lr_c,
        "LogReg (Set C': Adaptive + Lags)": p_val_lr_cp,
        "LightGBM (Set A: No Spatial)": p_val_lgb_a,
        "LightGBM (Set B: With Spatial)": p_val_lgb_b,
        "LightGBM (Set C: Spatial + Adaptive)": p_val_lgb_c,
        "LightGBM (Set C': Adaptive + Lags)": p_val_lgb_cp,
    }

    probas_test = {
        "Baseline 1: True Naive Persistence": test_df["lagged_onset_1h"].values.astype(float),
        "Baseline 2: Historical Lookup Table": test_df["lookup_prob"].values,
        "Baseline 3: Adaptive 14-day Rolling": test_df["adaptive_rate_14d"].values,
        "LogReg (Set A: No Spatial)": lr_a.predict_proba(scaler_a.transform(test_df[FEATURE_SET_A]))[:, 1],
        "LogReg (Set B: With Spatial)": lr_b.predict_proba(scaler_b.transform(test_df[FEATURE_SET_B]))[:, 1],
        "LogReg (Set C: Spatial + Adaptive)": lr_c.predict_proba(scaler_c.transform(test_df[FEATURE_SET_C]))[:, 1],
        "LogReg (Set C': Adaptive + Lags)": lr_cp.predict_proba(scaler_cp.transform(test_df[FEATURE_SET_C_PRIME]))[:, 1],
        "LightGBM (Set A: No Spatial)": lgb_a.predict_proba(test_df[FEATURE_SET_A])[:, 1],
        "LightGBM (Set B: With Spatial)": lgb_b.predict_proba(test_df[FEATURE_SET_B])[:, 1],
        "LightGBM (Set C: Spatial + Adaptive)": lgb_c.predict_proba(test_df[FEATURE_SET_C])[:, 1],
        "LightGBM (Set C': Adaptive + Lags)": lgb_cp.predict_proba(test_df[FEATURE_SET_C_PRIME])[:, 1],
    }

    # 3. Оцінка на Validation (Day-Block Bootstrap N=500) та вибір моделі за правилом arg max Val PR-AUC
    print("🔄 [Volunteer] Запуск Day-Block Bootstrap (N=500) на Validation...")
    df_val_res, val_diffs, _ = evaluate_slice(dates_val, y_val, probas_val, configs, n_bootstraps=500)
    df_val_res["threshold"] = df_val_res["frozen_threshold"]

    prespec_mask = ~df_val_res["model"].str.contains("Set C'")
    val_winner_idx = df_val_res.loc[prespec_mask, "pr_auc"].idxmax()
    val_winner_model = str(df_val_res.loc[val_winner_idx, "model"])

    def _status_label(m: str) -> str:
        if m == val_winner_model:
            return "pre-specified winner (selected on Val by PR-AUC)"
        if "Set C'" in m:
            return "post-hoc sensitivity experiment (Set C')"
        return "pre-specified candidate"

    df_val_res["selection_status"] = df_val_res["model"].map(_status_label)
    df_val_res.to_csv(vol_dir / "volunteer_ml_validation_benchmark.csv", index=False)

    # 4. Оцінка на Повному Test
    print("🔄 [Volunteer] Запуск Day-Block Bootstrap (N=500) на повному Test...")
    df_test_res, test_diffs, best_base_name = evaluate_slice(dates_test, y_test, probas_test, configs, n_bootstraps=500)
    df_test_res["selection_status"] = df_test_res["model"].map(_status_label)
    df_test_res.to_csv(vol_dir / "volunteer_ml_models_benchmark.csv", index=False)

    # 5. Оцінка на Test ДО та ПІСЛЯ 2025-08-01
    aug_cutoff = pd.Timestamp("2025-08-01 00:00:00", tz="UTC")
    pre_mask = (test_df["timestamp_utc"] < aug_cutoff).values
    post_mask = (test_df["timestamp_utc"] >= aug_cutoff).values

    df_pre_res, pre_diffs, _ = evaluate_slice(
        dates_test[pre_mask],
        y_test[pre_mask],
        {k: v[pre_mask] for k, v in probas_test.items()},
        configs,
        n_bootstraps=500,
    )
    df_pre_res.to_csv(vol_dir / "volunteer_ml_test_pre_aug2025_benchmark.csv", index=False)

    df_post_res, post_diffs, _ = evaluate_slice(
        dates_test[post_mask],
        y_test[post_mask],
        {k: v[post_mask] for k, v in probas_test.items()},
        configs,
        n_bootstraps=500,
    )
    df_post_res.to_csv(vol_dir / "volunteer_ml_test_post_aug2025_benchmark.csv", index=False)

    # 6. Оцінка на Останніх 60 днях
    test_max_time = test_df["timestamp_utc"].max()
    recent_cutoff = test_max_time - pd.Timedelta(days=60)
    rec_mask = (test_df["timestamp_utc"] >= recent_cutoff).values
    y_recent = y_test[rec_mask]
    probas_recent = {k: v[rec_mask] for k, v in probas_test.items()}

    df_rec_res, rec_diffs, _ = evaluate_slice(
        dates_test[rec_mask],
        y_recent,
        probas_recent,
        configs,
        n_bootstraps=500,
    )
    df_rec_res.to_csv(vol_dir / "volunteer_ml_recent_60d_benchmark.csv", index=False)

    # 7. Feature Importance та графіки оцінки на Test (10, 11, 12, 13) для Volunteer
    importance_gain = lgb_c.booster_.feature_importance(importance_type="gain")
    importance_split = lgb_c.booster_.feature_importance(importance_type="split")
    feat_names_ua = {
        "adaptive_rate_14d": "Адаптивна частота за 14 діб (год × регіон)",
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
    df_imp.to_csv(vol_dir / "volunteer_feature_importance_lightgbm.csv", index=False)

    plot_test_figures(
        y_test,
        probas_test,
        y_recent,
        probas_recent,
        df_imp,
        df_test_res,
        best_base_name,
        figures_dir=str(vol_fig_dir),
        dataset_tag="Volunteer",
    )

    cols_show = [
        "model", "frozen_threshold", "precision", "recall",
        "f2_score", "f2_ci_95", "pr_auc", "pr_auc_ci_95",
        "brier_score", "prec_at_adapt_recall",
    ]
    print("\n" + "=" * 115)
    print(f"📊 [VOLUNTEER naive_as_is] ВАЛІДАЦІЯ (Переможець за Val PR-AUC: {val_winner_model}):")
    print("=" * 115)
    print(df_val_res[cols_show].to_string(index=False))

    print("\n" + "=" * 115)
    print("📊 [VOLUNTEER naive_as_is] ПОВНИЙ TEST (2025-06-01 — 2026-10-11):")
    print("=" * 115)
    print(df_test_res[cols_show].to_string(index=False))

    print("\n" + "=" * 115)
    print(f"📊 [VOLUNTEER naive_as_is] ОСТАННІ 60 ДНІВ ({recent_cutoff.date()} — {test_max_time.date()}):")
    print("=" * 115)
    print(df_rec_res[cols_show].to_string(index=False))

    details = {
        "train_rows": int(len(train_df)),
        "train_positives": int(y_train.sum()),
        "train_base_rate": float(y_train.mean()),
        "val_rows": int(len(val_df)),
        "val_days": int(len(np.unique(dates_val))),
        "val_positives": int(y_val.sum()),
        "val_base_rate": float(y_val.mean()),
        "test_rows": int(len(test_df)),
        "test_days": int(len(np.unique(dates_test))),
        "test_positives": int(y_test.sum()),
        "test_base_rate": float(y_test.mean()),
        "test_pre_aug2025_rows": int(pre_mask.sum()),
        "test_pre_aug2025_positives": int(y_test[pre_mask].sum()),
        "test_pre_aug2025_base_rate": float(y_test[pre_mask].mean()),
        "test_post_aug2025_rows": int(post_mask.sum()),
        "test_post_aug2025_positives": int(y_test[post_mask].sum()),
        "test_post_aug2025_base_rate": float(y_test[post_mask].mean()),
        "recent_60d_rows": int(rec_mask.sum()),
        "recent_60d_positives": int(y_test[rec_mask].sum()),
        "recent_60d_base_rate": float(y_test[rec_mask].mean()),
        "selected_model_on_val": val_winner_model,
        "paired_day_bootstrap_500_val": val_diffs,
        "paired_day_bootstrap_500_test": test_diffs,
        "paired_day_bootstrap_500_pre_aug2025": pre_diffs,
        "paired_day_bootstrap_500_post_aug2025": post_diffs,
        "paired_day_bootstrap_500_recent_60d": rec_diffs,
    }
    with open(vol_dir / "volunteer_evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    return val_winner_model


def evaluate_single_variant_for_winner(
    variant_code: str,
    variant_label: str,
    fs_path: Path,
) -> list[dict]:
    """
    Виконує аналіз чутливості для обраної на Val моделі (LightGBM Set C)
    та базового орієнтира (Adaptive 14-day Rolling) на заданому варіанті обробки naive.
    """
    df = pd.read_parquet(fs_path)
    df = prepare_lookup_and_dates(df)

    train_df = df[df["split_set"] == "train"].copy()
    val_df = df[df["split_set"] == "val"].copy()
    test_df = df[df["split_set"] == "test"].copy()

    y_train = train_df["target_onset_next_1h"].values
    y_val = val_df["target_onset_next_1h"].values
    y_test = test_df["target_onset_next_1h"].values

    dates_val = val_df["date_kyiv"].values
    dates_test = test_df["date_kyiv"].values

    best_adapt, _ = evaluate_threshold_grid(y_val, val_df["adaptive_rate_14d"].values)
    lgb_c = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_C)
    p_val_lgb_c = lgb_c.predict_proba(val_df[FEATURE_SET_C])[:, 1]
    best_lgb_c, _ = evaluate_threshold_grid(y_val, p_val_lgb_c)

    p_test_adapt = test_df["adaptive_rate_14d"].values
    p_test_lgb_c = lgb_c.predict_proba(test_df[FEATURE_SET_C])[:, 1]

    # Швидкий парний bootstrap для 2 моделей (Adaptive 14d та LightGBM Set C)
    rng = np.random.RandomState(42)
    unique_days, day_inv = np.unique(dates_test, return_inverse=True)
    n_days = len(unique_days)
    day_idx = [np.where(day_inv == d)[0] for d in range(n_days)]
    sampled = rng.randint(0, n_days, size=(500, n_days))

    from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score

    day_weights = np.array(
        [np.bincount(sampled[b], minlength=n_days) for b in range(500)],
        dtype=np.float64,
    )

    def _boot_stats(y_arr, p_arr, th):
        preds = (p_arr >= th).astype(int)
        tp_d = np.array([int(((preds[i] == 1) & (y_arr[i] == 1)).sum()) for i in day_idx])
        fp_d = np.array([int(((preds[i] == 1) & (y_arr[i] == 0)).sum()) for i in day_idx])
        fn_d = np.array([int(((preds[i] == 0) & (y_arr[i] == 1)).sum()) for i in day_idx])
        tp_b = tp_d[sampled].sum(axis=1)
        fp_b = fp_d[sampled].sum(axis=1)
        fn_b = fn_d[sampled].sum(axis=1)
        denom = 5 * tp_b + 4 * fn_b + fp_b
        f2_b = np.where(denom > 0, (5 * tp_b) / denom, 0.0)

        order = np.argsort(p_arr, kind="mergesort")[::-1]
        p_s = p_arr[order]
        y_s = y_arr.astype(np.float64)[order]
        d_s = day_inv[order]
        thresh_idx = np.where(np.r_[np.diff(p_s) != 0, True])[0]

        pr_b = np.zeros(500, dtype=np.float64)
        for b in range(500):
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
            pr_b[b] = float(np.sum(d_rec * prec))
        return f2_b, pr_b

    f2_adapt_b, pr_adapt_b = _boot_stats(y_test, p_test_adapt, float(best_adapt["threshold"]))
    f2_lgb_b, pr_lgb_b = _boot_stats(y_test, p_test_lgb_c, float(best_lgb_c["threshold"]))

    diff_pr = pr_lgb_b - pr_adapt_b
    diff_f2 = f2_lgb_b - f2_adapt_b

    rows = []
    for m_name, p_val_arr, p_test_arr, th, f2_b, pr_b in [
        ("Baseline 3: Adaptive 14-day Rolling", val_df["adaptive_rate_14d"].values, p_test_adapt, float(best_adapt["threshold"]), f2_adapt_b, pr_adapt_b),
        ("LightGBM (Set C: Spatial + Adaptive)", p_val_lgb_c, p_test_lgb_c, float(best_lgb_c["threshold"]), f2_lgb_b, pr_lgb_b),
    ]:
        preds_val = (p_val_arr >= th).astype(int)
        preds_test = (p_test_arr >= th).astype(int)
        rows.append({
            "naive_variant": variant_code,
            "naive_description": variant_label,
            "model": m_name,
            "calm_rows_total": int(len(df)),
            "calm_rows_test": int(len(test_df)),
            "test_positives": int(y_test.sum()),
            "test_base_rate": round(float(y_test.mean()), 5),
            "test_mean_hours_since_last_alert": round(float(test_df["hours_since_last_alert"].mean()), 2),
            "test_mean_burden_24h": round(float(test_df["burden_last_24h"].mean()), 4),
            "val_threshold": round(th, 4),
            "val_pr_auc": round(float(average_precision_score(y_val, p_val_arr)), 4),
            "val_f2": round(float(fbeta_score(y_val, preds_val, beta=2, zero_division=0)), 4),
            "test_precision": round(float(precision_score(y_test, preds_test, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, preds_test, zero_division=0)), 4),
            "test_f2": round(float(fbeta_score(y_test, preds_test, beta=2, zero_division=0)), 4),
            "test_f2_ci_95": f"[{np.percentile(f2_b, 2.5):.3f}, {np.percentile(f2_b, 97.5):.3f}]",
            "test_pr_auc": round(float(average_precision_score(y_test, p_test_arr)), 4),
            "test_pr_auc_ci_95": f"[{np.percentile(pr_b, 2.5):.3f}, {np.percentile(pr_b, 97.5):.3f}]",
            "delta_prauc_lgbC_minus_adapt": round(float(diff_pr.mean()), 4) if "LightGBM" in m_name else 0.0,
            "delta_prauc_ci_95": f"[{np.percentile(diff_pr, 2.5):+.4f}, {np.percentile(diff_pr, 97.5):+.4f}]" if "LightGBM" in m_name else "-",
            "delta_f2_lgbC_minus_adapt": round(float(diff_f2.mean()), 4) if "LightGBM" in m_name else 0.0,
            "delta_f2_ci_95": f"[{np.percentile(diff_f2, 2.5):+.4f}, {np.percentile(diff_f2, 97.5):+.4f}]" if "LightGBM" in m_name else "-",
        })
    return rows


def main():
    vol_dir = PROJECT_ROOT / "reports" / "volunteer"
    vol_fig_dir = vol_dir / "figures"
    vol_dir.mkdir(parents=True, exist_ok=True)
    vol_fig_dir.mkdir(parents=True, exist_ok=True)

    clean_as_is = PROJECT_ROOT / "data" / "processed" / "volunteer_alerts_clean_as_is.parquet"

    # 0. EDA та просторовий аналіз (графіки 01..06) для волонтерського датасету (naive_as_is)
    print("\n📊 [Volunteer] Побудова графіків сезонності, кластеризації та поширення (01..06)...")
    analyze_temporal_cyclicity(clean_as_is, vol_fig_dir)
    perform_regional_clustering(clean_as_is, vol_dir)
    analyze_spatial_propagation(clean_as_is, vol_dir)

    # 1. Кластери та Feature Store для трьох каузальних варіантів
    variants = [
        (
            "naive_as_is",
            "(а) naive як є (finished_at = started_at + 30 хв)",
            clean_as_is,
            vol_dir / "volunteer_clusters_train_only.csv",
            PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_as_is.parquet",
        ),
        (
            "drop_naive",
            "(б) без наївних тривог (drop naive == True)",
            PROJECT_ROOT / "data" / "processed" / "volunteer_alerts_clean_drop_naive.parquet",
            vol_dir / "volunteer_clusters_train_only_drop_naive.csv",
            PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_drop_naive.parquet",
        ),
        (
            "naive_fixed_D",
            "(в) каузальний naive_fixed_D (finished_at = started_at + D_oblast із Train)",
            PROJECT_ROOT / "data" / "processed" / "volunteer_alerts_clean_fixed_d.parquet",
            vol_dir / "volunteer_clusters_train_only_fixed_d.csv",
            PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_fixed_d.parquet",
        ),
    ]

    for code, label, clean_pq, clust_csv, fs_pq in variants:
        print(f"\n⚙️ Побудова Train-only кластерів та Feature Store для [{code}]...")
        generate_train_only_clusters(clean_pq, clust_csv)
        generate_feature_store(clean_pq, clust_csv, fs_pq)

    # Оцінка бейзлайнів та побудова графіка 07 для волонтерського датасету
    run_baselines_evaluation(
        ml_features_parquet=PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_as_is.parquet",
        output_dir=vol_dir,
    )

    # 2. Аналіз зсуву розподілу ознак на волонтерських даних (naive_as_is)
    print("\n📉 Аналіз зсуву розподілу ознак на волонтерському Feature Store (naive_as_is)...")
    analyze_feature_shift(
        parquet_path=PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_as_is.parquet",
        out_splits_csv=vol_dir / "volunteer_feature_shift_summary.csv",
        out_subperiods_csv=vol_dir / "volunteer_feature_shift_test_subperiods.csv",
    )

    # 3. Повний протокол лише для (а) naive_as_is
    val_winner = run_full_protocol_naive_as_is(
        PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_as_is.parquet"
    )
    print(f"\n🏆 Обрана на Val модель (volunteer): {val_winner}")

    # 4. Чутливість між (а), (б), (в) лише для обраної на Val моделі (LightGBM Set C) та Adaptive 14-day
    print("\n🔬 Розрахунок чутливості до обробки naive == True (варіанти а, б, в)...")
    sens_rows = []
    for code, label, _, _, fs_pq in variants:
        sens_rows.extend(evaluate_single_variant_for_winner(code, label, fs_pq))

    sens_df = pd.DataFrame(sens_rows)
    sens_csv = vol_dir / "volunteer_naive_sensitivity.csv"
    sens_json = vol_dir / "volunteer_naive_sensitivity.json"
    sens_df.to_csv(sens_csv, index=False)
    sens_json.write_text(json.dumps(sens_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 115)
    print("📊 АНАЛІЗ ЧУТЛИВОСТІ ДО НАЇВНИХ ТРИВОГ (VOLUNTEER DATASET: ВАРІАНТИ А, Б, В):")
    print("=" * 115)
    print(sens_df[[
        "naive_variant", "model", "calm_rows_test", "test_mean_hours_since_last_alert",
        "test_mean_burden_24h", "val_pr_auc", "test_pr_auc", "test_pr_auc_ci_95",
        "test_f2", "test_f2_ci_95", "delta_prauc_ci_95"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

