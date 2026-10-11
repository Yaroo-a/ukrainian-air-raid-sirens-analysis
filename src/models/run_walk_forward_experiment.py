"""Walk-Forward (ковзне щомісячне перенавчання) для перевірки гіпотези нестаціонарності
та виходу ознак за межі навчального діапазону на інтервалі прогнозування [T, T+1h).

Заздалегідь зафіксований протокол (Pre-Specified Protocol):
1. Головна гіпотеза (Primary Pre-Specified Hypothesis):
   - Модель `LightGBM (Set C) [WF Rolling 12m]` (12 місяців ковзного Train + 2 місяці Val перед кожним
     тестовим місяцем M з 2-годинним embargo) проти справедливого бейзлайну
     `Baseline 3: Adaptive 14-day Rolling (WF p*_M)` за парною різницею ΔPR-AUC (Day-Block Bootstrap, N=500),
     окремо на двох незалежних джерелах:
     (а) Official (`data/processed/ml_features_hourly.parquet`)
     (б) Volunteer (`data/processed/volunteer_ml_features_hourly_fixed_d.parquet`, каузальний варіант naive_fixed_D).
2. Усі інші конфігурації позначені явно:
   - `LogReg (Set C) [WF Rolling 12m]` — exploratory (pre-specified Set C)
   - `Expanding Window (+ 2m Val)` — exploratory secondary window configuration
   - `Set D (Set C + Relative Ratios)` — post-hoc exploratory feature set (сконструйований після перегляду Test).
3. Для кожного місяця M у Test (починаючи з 2025-06):
   - Тестове вікно M: [T_start(M), T_end(M))
   - Внутрішня валідація Val_M (останні 2 місяці перед M з 2-годинним embargo для інтервалу цілі [T, T+1h)):
     [T_start(M) - 2 months, T_start(M) - 2 hours]
   - Навчальне вікно Train_M (з 2-годинним embargo перед Val_M):
     * Rolling 12m: [T_start(M) - 14 months, T_start(M) - 2 months - 2 hours]
     * Expanding:   [T_min, T_start(M) - 2 months - 2 hours]
   - Жодного повторного перенавчання на Val_M немає: модель навчається на Train_M (з early stopping на Val_M),
     поріг p*_M обирається на Val_M (максимізація F2) як для ML-моделей, так і для Adaptive 14-day бейзлайну.
4. Метрики:
   - Об'єднаний (pooled out-of-fold) PR-AUC з 95% Day-Block Bootstrap ДІ (N=500)
   - Середнє помісячних PR-AUC (`mean_monthly_pr_auc`)
   - Precision при тому ж Recall, що в `Adaptive 14-day (WF p*_M)` (`prec_at_adapt_recall`)
   - F2-Score при p*_M з 95% ДІ та довідковий рядок `Predict-All (p* = 0.0)` F2
   - Пряме порівняння `(WF − Adaptive)` проти `(Static − Adaptive)` та парна різниця `(WF − Static)`.
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.models.train_models import FEATURE_SET_C
from src.models.evaluate_test import (
    compute_equal_operating_points,
    day_block_bootstrap_all_models,
)

# Пост-хок набір ознак Set D: Set C + 3 безрозмірні/центровані відносні ознаки
POST_HOC_RELATIVE_FEATURES = [
    "rel_activity_24h_vs_14d",
    "adaptive_ratio_7d_vs_30d",
    "adaptive_diff_14d_minus_30d",
]
FEATURE_SET_D_POST_HOC = FEATURE_SET_C + POST_HOC_RELATIVE_FEATURES


def add_post_hoc_set_d_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Каузально додає 3 відносні ознаки Set D (post-hoc), які обчислюються
    виключно з поточних каузальних вікон у точці T без жодних групових статистик.
    """
    out = df.copy()
    eps = 1e-4
    out["rel_activity_24h_vs_14d"] = np.clip(
        out["onsets_last_24h"] / (24.0 * out["adaptive_rate_14d"] + eps),
        0.0,
        20.0,
    ).astype(np.float32)
    out["adaptive_ratio_7d_vs_30d"] = np.clip(
        (out["adaptive_rate_7d"] + eps) / (out["adaptive_rate_30d"] + eps),
        0.0,
        20.0,
    ).astype(np.float32)
    out["adaptive_diff_14d_minus_30d"] = (
        out["adaptive_rate_14d"] - out["adaptive_rate_30d"]
    ).astype(np.float32)
    return out


def fast_threshold_search_f2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Векторизований пошук порогу p*, що максимізує F2 на валідаційній вибірці (сітка з train_models.py)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.70, 175)
    y_b = (y_true == 1)[:, None]
    preds = y_prob[:, None] >= thresholds[None, :]

    tp = np.sum(preds & y_b, axis=0, dtype=np.float64)
    fp = np.sum(preds & (~y_b), axis=0, dtype=np.float64)
    fn = np.sum((~preds) & y_b, axis=0, dtype=np.float64)

    denom = 5.0 * tp + 4.0 * fn + fp
    f2 = np.where(denom > 0, (5.0 * tp) / denom, 0.0)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

    best_idx = int(np.argmax(f2))
    return {
        "threshold": float(thresholds[best_idx]),
        "f2_score": float(f2[best_idx]),
        "precision": float(prec[best_idx]),
        "recall": float(rec[best_idx]),
    }


def get_walk_forward_split_masks(
    timestamps_utc: pd.Series,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    window_mode: str = "rolling_12m",
    embargo_hours: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, pd.Timestamp]]:
    """
    Повертає булеві маски (train_mask, val_mask, test_mask) та словник часових меж
    для заданого тестового місяця M із суворим embargo = 2 години між Train -> Val та Val -> Test
    (оскільки ціль прогнозується на інтервалі [T, T+1h)).
    """
    embargo = pd.Timedelta(hours=embargo_hours)
    val_end = month_start - embargo
    val_start = month_start - pd.DateOffset(months=2)
    train_end = val_start - embargo

    if window_mode == "rolling_12m":
        train_start = month_start - pd.DateOffset(months=14)
    elif window_mode == "expanding":
        train_start = timestamps_utc.min()
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")

    ts = timestamps_utc
    train_mask = ((ts >= train_start) & (ts <= train_end)).values
    val_mask = ((ts >= val_start) & (ts <= val_end)).values
    test_mask = ((ts >= month_start) & (ts < month_end)).values

    bounds = {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": month_start,
        "test_end": month_end,
    }
    return train_mask, val_mask, test_mask, bounds


def fit_fold_logreg(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    features: list[str],
    random_state: int = 42,
):
    """Навчає LogisticRegression строго на X_train та обирає поріг p*_M на X_val (параметри train_models.py)."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[features])
    X_va_s = scaler.transform(X_val[features])

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_tr_s, y_train)
    p_val = clf.predict_proba(X_va_s)[:, 1]
    th_info = fast_threshold_search_f2(y_val, p_val)
    return clf, scaler, th_info["threshold"], p_val


def fit_fold_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    features: list[str],
    n_jobs: int = 4,
    random_state: int = 42,
):
    """Навчає LightGBM строго на X_train (early stopping на X_val) з тими самими гіперпараметрами, що в train_models.py."""
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=n_jobs,
        deterministic=(n_jobs == 1),
        verbosity=-1,
    )
    model.fit(
        X_train[features],
        y_train,
        eval_set=[(X_val[features], y_val)],
        eval_names=["val"],
        callbacks=[early_stopping(stopping_rounds=30, verbose=False), log_evaluation(period=0)],
    )
    p_val = model.predict_proba(X_val[features])[:, 1]
    th_info = fast_threshold_search_f2(y_val, p_val)
    return model, th_info["threshold"], p_val


def evaluate_walk_forward_predictions(
    dates_arr: np.ndarray,
    months_arr: np.ndarray,
    y_true: np.ndarray,
    probas_dict: dict[str, np.ndarray],
    preds_binary_dict: dict[str, np.ndarray],
    model_meta: list[tuple[str, str, str]],
    ref_model_name: str = "Baseline 3: Adaptive 14-day (WF p*_M)",
    n_bootstraps: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """
    Обчислює зведену таблицю метрик для всіх моделей (Static + Walk-Forward):
    - pooled PR-AUC + 95% Day-Block Bootstrap CI
    - mean_monthly_pr_auc (середнє помісячних PR-AUC)
    - F2-Score (при помісячних порогах p*_M або замороженому порозі p*) + 95% CI
    - Precision, Recall, Brier Score, Precision @ Adaptive(WF) Recall
    - Рядок Predict-All (p* = 0.0) F2
    - Парні різниці (WF - Adaptive), (Static - Adaptive) та (WF - Static).
    """
    unique_days, day_inv = np.unique(dates_arr, return_inverse=True)
    n_days = len(unique_days)
    day_indices = [np.where(day_inv == d)[0] for d in range(n_days)]

    rng = np.random.RandomState(42)
    sampled_days = rng.randint(0, n_days, size=(n_bootstraps, n_days))
    day_weights = np.array(
        [np.bincount(sampled_days[b], minlength=n_days) for b in range(n_bootstraps)],
        dtype=np.float64,
    )
    y_float = y_true.astype(np.float64)

    def _boot_model(name: str):
        p_vals = probas_dict[name]
        preds = preds_binary_dict[name]

        tp_d = np.array([int(((preds[i] == 1) & (y_true[i] == 1)).sum()) for i in day_indices])
        fp_d = np.array([int(((preds[i] == 1) & (y_true[i] == 0)).sum()) for i in day_indices])
        fn_d = np.array([int(((preds[i] == 0) & (y_true[i] == 1)).sum()) for i in day_indices])

        tp_b = tp_d[sampled_days].sum(axis=1)
        fp_b = fp_d[sampled_days].sum(axis=1)
        fn_b = fn_d[sampled_days].sum(axis=1)
        denom = 5 * tp_b + 4 * fn_b + fp_b
        f2_b = np.where(denom > 0, (5 * tp_b) / denom, 0.0)

        if "Predict-All" in name:
            return name, f2_b, np.full(n_bootstraps, np.nan)

        order = np.argsort(p_vals, kind="mergesort")[::-1]
        p_s = p_vals[order]
        y_s = y_float[order]
        d_s = day_inv[order]
        thresh_idx = np.where(np.r_[np.diff(p_s) != 0, True])[0]

        pr_b = np.zeros(n_bootstraps, dtype=np.float64)
        for b in range(n_bootstraps):
            w_s = day_weights[b, d_s]
            tp_cum = np.cumsum(y_s * w_s)[thresh_idx]
            total_pos = tp_cum[-1]
            if total_pos <= 0:
                continue
            tot_cum = np.cumsum(w_s)[thresh_idx]
            keep = np.r_[tot_cum[0] > 0, np.diff(tot_cum) > 0]
            tps = tp_cum[keep]
            tots = tot_cum[keep]
            prec = tps / tots
            rec = tps / total_pos
            d_rec = np.r_[rec[0], np.diff(rec)]
            pr_b[b] = float(np.sum(d_rec * prec))
        return name, f2_b, pr_b

    from joblib import Parallel, delayed
    boot_out = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_boot_model)(name) for name, _, _ in model_meta
    )
    boot_f2 = {name: f2_b for name, f2_b, _ in boot_out}
    boot_pr = {name: pr_b for name, _, pr_b in boot_out}

    ref_preds = preds_binary_dict[ref_model_name]
    ref_recall = recall_score(y_true, ref_preds, zero_division=0)
    ref_prec = precision_score(y_true, ref_preds, zero_division=0)

    unique_months = sorted(np.unique(months_arr))

    rows = []
    for name, fset_label, status_label in model_meta:
        p_vals = probas_dict[name]
        preds = preds_binary_dict[name]

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)

        if "Predict-All" in name:
            pooled_pr = np.nan
            mean_m_pr = np.nan
            brier = np.nan
            p_at_ref_r = prec
            pr_ci_str = "-"
            d_pr_mean = np.nan
            d_pr_ci_str = "-"
        else:
            pooled_pr = float(average_precision_score(y_true, p_vals))
            m_prs = [
                float(average_precision_score(y_true[months_arr == m], p_vals[months_arr == m]))
                for m in unique_months
                if (months_arr == m).sum() > 0 and y_true[months_arr == m].sum() > 0
            ]
            mean_m_pr = float(np.mean(m_prs))
            brier = float(brier_score_loss(y_true, p_vals))
            p_at_ref_r, _ = compute_equal_operating_points(y_true, p_vals, ref_recall, ref_prec)
            pr_b = boot_pr[name]
            pr_ci_str = f"[{np.percentile(pr_b, 2.5):.3f}, {np.percentile(pr_b, 97.5):.3f}]"

            diff_pr_adapt = pr_b - boot_pr[ref_model_name]
            d_pr_mean = float(np.mean(diff_pr_adapt))
            d_pr_ci_str = f"[{np.percentile(diff_pr_adapt, 2.5):+.4f}, {np.percentile(diff_pr_adapt, 97.5):+.4f}]"

        f2_b = boot_f2[name]
        f2_ci_str = f"[{np.percentile(f2_b, 2.5):.3f}, {np.percentile(f2_b, 97.5):.3f}]"
        diff_f2_adapt = f2_b - boot_f2[ref_model_name]
        d_f2_mean = float(np.mean(diff_f2_adapt))
        d_f2_ci_str = f"[{np.percentile(diff_f2_adapt, 2.5):+.4f}, {np.percentile(diff_f2_adapt, 97.5):+.4f}]"

        rows.append({
            "model": name,
            "feature_set": fset_label,
            "protocol_status": status_label,
            "pooled_pr_auc": pooled_pr,
            "pr_auc_ci_95": pr_ci_str,
            "mean_monthly_pr_auc": mean_m_pr,
            "delta_prauc_vs_adapt_wf": d_pr_mean,
            "delta_prauc_vs_adapt_wf_ci_95": d_pr_ci_str,
            "prec_at_adapt_recall": p_at_ref_r,
            "precision": prec,
            "recall": rec,
            "f2_score": f2,
            "f2_ci_95": f2_ci_str,
            "delta_f2_vs_adapt_wf": d_f2_mean,
            "delta_f2_vs_adapt_wf_ci_95": d_f2_ci_str,
            "brier_score": brier,
        })

    df_res = pd.DataFrame(rows)

    # Пряме порівняння (WF - Adaptive) проти (Static - Adaptive) та (WF - Static)
    direct_comparisons = {}
    wf_vs_static_pairs = [
        (
            "LightGBM Set C (Rolling 12m vs Static)",
            "LightGBM (Set C) [WF Rolling 12m]",
            "LightGBM (Set C) [Static Initial Train]",
        ),
        (
            "LightGBM Set C (Expanding vs Static)",
            "LightGBM (Set C) [WF Expanding]",
            "LightGBM (Set C) [Static Initial Train]",
        ),
        (
            "LogReg Set C (Rolling 12m vs Static)",
            "LogReg (Set C) [WF Rolling 12m]",
            "LogReg (Set C) [Static Initial Train]",
        ),
        (
            "LogReg Set C (Expanding vs Static)",
            "LogReg (Set C) [WF Expanding]",
            "LogReg (Set C) [Static Initial Train]",
        ),
        (
            "LightGBMRolling12m Set D (post-hoc) vs Set C",
            "LightGBM (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]",
            "LightGBM (Set C) [WF Rolling 12m]",
        ),
    ]
    for label, m_wf, m_stat in wf_vs_static_pairs:
        diff_wf_adapt_pr = boot_pr[m_wf] - boot_pr[ref_model_name]
        diff_stat_adapt_pr = boot_pr[m_stat] - boot_pr[ref_model_name]
        diff_wf_stat_pr = boot_pr[m_wf] - boot_pr[m_stat]

        diff_wf_adapt_f2 = boot_f2[m_wf] - boot_f2[ref_model_name]
        diff_stat_adapt_f2 = boot_f2[m_stat] - boot_f2[ref_model_name]
        diff_wf_stat_f2 = boot_f2[m_wf] - boot_f2[m_stat]

        direct_comparisons[label] = {
            "wf_model": m_wf,
            "reference_model": m_stat,
            "wf_minus_adapt_prauc_mean": float(np.mean(diff_wf_adapt_pr)),
            "wf_minus_adapt_prauc_ci_95": [float(np.percentile(diff_wf_adapt_pr, 2.5)), float(np.percentile(diff_wf_adapt_pr, 97.5))],
            "ref_minus_adapt_prauc_mean": float(np.mean(diff_stat_adapt_pr)),
            "ref_minus_adapt_prauc_ci_95": [float(np.percentile(diff_stat_adapt_pr, 2.5)), float(np.percentile(diff_stat_adapt_pr, 97.5))],
            "wf_minus_ref_prauc_mean": float(np.mean(diff_wf_stat_pr)),
            "wf_minus_ref_prauc_ci_95": [float(np.percentile(diff_wf_stat_pr, 2.5)), float(np.percentile(diff_wf_stat_pr, 97.5))],
            "wf_minus_adapt_f2_mean": float(np.mean(diff_wf_adapt_f2)),
            "wf_minus_adapt_f2_ci_95": [float(np.percentile(diff_wf_adapt_f2, 2.5)), float(np.percentile(diff_wf_adapt_f2, 97.5))],
            "ref_minus_adapt_f2_mean": float(np.mean(diff_stat_adapt_f2)),
            "ref_minus_adapt_f2_ci_95": [float(np.percentile(diff_stat_adapt_f2, 2.5)), float(np.percentile(diff_stat_adapt_f2, 97.5))],
            "wf_minus_ref_f2_mean": float(np.mean(diff_wf_stat_f2)),
            "wf_minus_ref_f2_ci_95": [float(np.percentile(diff_wf_stat_f2, 2.5)), float(np.percentile(diff_wf_stat_f2, 97.5))],
        }

    return df_res, direct_comparisons


def run_walk_forward_for_dataset(
    dataset_name: str,
    parquet_path: Path,
    out_dir: Path,
    file_prefix: str = "",
) -> dict:
    """Виконує повний Walk-Forward експеримент для одного джерела даних (Official або Volunteer)."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 105)
    print(f"🚀 WALK-FORWARD ЕКСПЕРИМЕНТ НА ДЖЕРЕЛІ [{dataset_name.upper()}] ({parquet_path.name})")
    print("=" * 105)

    df = pd.read_parquet(parquet_path)
    df = add_post_hoc_set_d_features(df)
    kyiv_dt = df["timestamp_utc"].dt.tz_convert("Europe/Kyiv")
    df["date_kyiv"] = kyiv_dt.dt.date
    df["year_month_utc"] = df["timestamp_utc"].dt.strftime("%Y-%m")

    # 1. Статичні еталонні моделі (використовуємо ті самі заморожені файли моделей та порогів з основного звіту)
    import joblib
    init_train = df[df["split_set"] == "train"]
    init_val = df[df["split_set"] == "val"]
    y_init_tr = init_train["target_onset_next_1h"].values
    y_init_va = init_val["target_onset_next_1h"].values

    if file_prefix == "volunteer":
        frozen_lgb_path = PROJECT_ROOT / "models" / "volunteer_lightgbm_adaptive.joblib"
        frozen_th_path = PROJECT_ROOT / "models" / "volunteer_optimal_thresholds.json"
    else:
        frozen_lgb_path = PROJECT_ROOT / "models" / "lightgbm_adaptive.joblib"
        frozen_th_path = PROJECT_ROOT / "models" / "optimal_thresholds.json"

    with open(frozen_th_path, "r", encoding="utf-8") as f:
        frozen_ths = json.load(f)

    static_adapt_th = float(frozen_ths["Baseline 3: Adaptive 14-day Rolling"])
    static_lgb_c = joblib.load(frozen_lgb_path)
    static_lgb_c_th = float(frozen_ths["LightGBM (Set C: Spatial + Adaptive)"])
    print(f"📦 Завантажено заморожену статичну модель: {frozen_lgb_path.name} (p*={static_lgb_c_th:.4f})")

    if file_prefix != "volunteer" and (PROJECT_ROOT / "models" / "logreg_adaptive.joblib").exists():
        static_lr_c = joblib.load(PROJECT_ROOT / "models" / "logreg_adaptive.joblib")
        static_sc_c = joblib.load(PROJECT_ROOT / "models" / "scaler_adaptive.joblib")
        static_lr_c_th = float(frozen_ths["LogReg (Set C: Spatial + Adaptive)"])
    else:
        static_lr_c, static_sc_c, static_lr_c_th, _ = fit_fold_logreg(
            init_train, y_init_tr, init_val, y_init_va, FEATURE_SET_C
        )

    static_train_max_adapt = float(init_train["adaptive_rate_14d"].max())
    static_train_p99_adapt = float(np.percentile(init_train["adaptive_rate_14d"].values, 99.0))

    # 2. Визначаємо всі тестові місяці M (починаючи з 2025-06)
    test_only = df[df["split_set"] == "test"]
    test_months = sorted(test_only["year_month_utc"].unique())
    print(f"📅 Тестових місяців: {len(test_months)} ({test_months[0]} .. {test_months[-1]})")

    monthly_diag_rows = []
    oof_chunks = []

    for m_str in test_months:
        m_start = pd.Timestamp(f"{m_str}-01 00:00:00", tz="UTC")
        m_end = m_start + pd.DateOffset(months=1)

        tr_roll_mask, va_mask, te_mask, bounds_roll = get_walk_forward_split_masks(
            df["timestamp_utc"], m_start, m_end, window_mode="rolling_12m", embargo_hours=2
        )
        tr_exp_mask, _, _, bounds_exp = get_walk_forward_split_masks(
            df["timestamp_utc"], m_start, m_end, window_mode="expanding", embargo_hours=2
        )

        tr_roll_df = df.loc[tr_roll_mask]
        tr_exp_df = df.loc[tr_exp_mask]
        va_df = df.loc[va_mask]
        te_df = df.loc[te_mask].copy()

        y_tr_roll = tr_roll_df["target_onset_next_1h"].values
        y_tr_exp = tr_exp_df["target_onset_next_1h"].values
        y_va = va_df["target_onset_next_1h"].values
        y_te = te_df["target_onset_next_1h"].values

        # Каузальна перевірка часових меж усередині кожного місяця M
        assert tr_roll_df["timestamp_utc"].max() <= bounds_roll["val_start"] - pd.Timedelta(hours=2)
        assert va_df["timestamp_utc"].max() <= m_start - pd.Timedelta(hours=2)
        assert te_df["timestamp_utc"].min() >= m_start

        # Поріг справедливого бейзлайну Adaptive 14-day на Val_M
        wf_adapt_th = fast_threshold_search_f2(y_va, va_df["adaptive_rate_14d"].values)["threshold"]

        # Навчання моделей на Rolling 12m (головна конфігурація)
        roll_lgb_c, roll_lgb_c_th, _ = fit_fold_lightgbm(tr_roll_df, y_tr_roll, va_df, y_va, FEATURE_SET_C)
        roll_lr_c, roll_sc_c, roll_lr_c_th, _ = fit_fold_logreg(tr_roll_df, y_tr_roll, va_df, y_va, FEATURE_SET_C)
        roll_lgb_d, roll_lgb_d_th, _ = fit_fold_lightgbm(tr_roll_df, y_tr_roll, va_df, y_va, FEATURE_SET_D_POST_HOC)
        roll_lr_d, roll_sc_d, roll_lr_d_th, _ = fit_fold_logreg(tr_roll_df, y_tr_roll, va_df, y_va, FEATURE_SET_D_POST_HOC)

        # Навчання моделей на Expanding Window (додаткова конфігурація)
        exp_lgb_c, exp_lgb_c_th, _ = fit_fold_lightgbm(tr_exp_df, y_tr_exp, va_df, y_va, FEATURE_SET_C)
        exp_lr_c, exp_sc_c, exp_lr_c_th, _ = fit_fold_logreg(tr_exp_df, y_tr_exp, va_df, y_va, FEATURE_SET_C)
        exp_lgb_d, exp_lgb_d_th, _ = fit_fold_lightgbm(tr_exp_df, y_tr_exp, va_df, y_va, FEATURE_SET_D_POST_HOC)
        exp_lr_d, exp_sc_d, exp_lr_d_th, _ = fit_fold_logreg(tr_exp_df, y_tr_exp, va_df, y_va, FEATURE_SET_D_POST_HOC)

        # Прогнози на тестовому місяці M
        p_adapt = te_df["adaptive_rate_14d"].values
        p_stat_lgb_c = static_lgb_c.predict_proba(te_df[FEATURE_SET_C])[:, 1]
        p_stat_lr_c = static_lr_c.predict_proba(static_sc_c.transform(te_df[FEATURE_SET_C]))[:, 1]

        p_roll_lgb_c = roll_lgb_c.predict_proba(te_df[FEATURE_SET_C])[:, 1]
        p_roll_lr_c = roll_lr_c.predict_proba(roll_sc_c.transform(te_df[FEATURE_SET_C]))[:, 1]
        p_roll_lgb_d = roll_lgb_d.predict_proba(te_df[FEATURE_SET_D_POST_HOC])[:, 1]
        p_roll_lr_d = roll_lr_d.predict_proba(roll_sc_d.transform(te_df[FEATURE_SET_D_POST_HOC]))[:, 1]

        p_exp_lgb_c = exp_lgb_c.predict_proba(te_df[FEATURE_SET_C])[:, 1]
        p_exp_lr_c = exp_lr_c.predict_proba(exp_sc_c.transform(te_df[FEATURE_SET_C]))[:, 1]
        p_exp_lgb_d = exp_lgb_d.predict_proba(te_df[FEATURE_SET_D_POST_HOC])[:, 1]
        p_exp_lr_d = exp_lr_d.predict_proba(exp_sc_d.transform(te_df[FEATURE_SET_D_POST_HOC]))[:, 1]

        # Діагностика виходу adaptive_rate_14d за межі Train
        roll_max_adapt = float(tr_roll_df["adaptive_rate_14d"].max())
        roll_p99_adapt = float(np.percentile(tr_roll_df["adaptive_rate_14d"].values, 99.0))
        exp_max_adapt = float(tr_exp_df["adaptive_rate_14d"].max())
        exp_p99_adapt = float(np.percentile(tr_exp_df["adaptive_rate_14d"].values, 99.0))

        base_rate_m = float(y_te.mean())
        predict_all_f2_m = (5.0 * base_rate_m) / (4.0 * base_rate_m + 1.0) if base_rate_m > 0 else 0.0

        monthly_diag_rows.append({
            "month": m_str,
            "train_roll12m_rows": int(len(tr_roll_df)),
            "val_2m_rows": int(len(va_df)),
            "test_rows": int(len(te_df)),
            "test_positives": int(y_te.sum()),
            "actual_base_rate": round(base_rate_m, 5),
            "predict_all_f2": round(predict_all_f2_m, 4),
            # Частота виходу за межі статичного vs ковзного Train
            "static_train_max_adapt14d": round(static_train_max_adapt, 4),
            "static_train_p99_adapt14d": round(static_train_p99_adapt, 4),
            "share_gt_static_train_max": round(float((p_adapt > static_train_max_adapt).mean()), 4),
            "share_gt_static_train_p99": round(float((p_adapt > static_train_p99_adapt).mean()), 4),
            "roll12m_train_max_adapt14d": round(roll_max_adapt, 4),
            "roll12m_train_p99_adapt14d": round(roll_p99_adapt, 4),
            "share_gt_roll12m_train_max": round(float((p_adapt > roll_max_adapt).mean()), 4),
            "share_gt_roll12m_train_p99": round(float((p_adapt > roll_p99_adapt).mean()), 4),
            "exp_train_max_adapt14d": round(exp_max_adapt, 4),
            "share_gt_exp_train_max": round(float((p_adapt > exp_max_adapt).mean()), 4),
            # Середній прогноз проти фактичної частоти
            "mean_pred_adapt14d": round(float(p_adapt.mean()), 5),
            "mean_pred_static_lgb_c": round(float(p_stat_lgb_c.mean()), 5),
            "mean_pred_wf_roll12m_lgb_c": round(float(p_roll_lgb_c.mean()), 5),
            "mean_pred_wf_roll12m_lgb_d": round(float(p_roll_lgb_d.mean()), 5),
            "mean_pred_wf_roll12m_lr_c": round(float(p_roll_lr_c.mean()), 5),
            "mean_pred_wf_exp_lgb_c": round(float(p_exp_lgb_c.mean()), 5),
            # Пороги p*_M
            "th_adapt_wf": round(wf_adapt_th, 4),
            "th_wf_roll12m_lgb_c": round(roll_lgb_c_th, 4),
            "th_wf_roll12m_lr_c": round(roll_lr_c_th, 4),
            "th_wf_roll12m_lgb_d": round(roll_lgb_d_th, 4),
            # Помісячні PR-AUC
            "prauc_adapt": round(float(average_precision_score(y_te, p_adapt)), 4),
            "prauc_static_lgb_c": round(float(average_precision_score(y_te, p_stat_lgb_c)), 4),
            "prauc_wf_roll12m_lgb_c": round(float(average_precision_score(y_te, p_roll_lgb_c)), 4),
            "prauc_wf_roll12m_lgb_d": round(float(average_precision_score(y_te, p_roll_lgb_d)), 4),
            "prauc_wf_roll12m_lr_c": round(float(average_precision_score(y_te, p_roll_lr_c)), 4),
            "prauc_wf_roll12m_lr_d": round(float(average_precision_score(y_te, p_roll_lr_d)), 4),
            "prauc_wf_exp_lgb_c": round(float(average_precision_score(y_te, p_exp_lgb_c)), 4),
            "prauc_wf_exp_lgb_d": round(float(average_precision_score(y_te, p_exp_lgb_d)), 4),
            "prauc_wf_exp_lr_c": round(float(average_precision_score(y_te, p_exp_lr_c)), 4),
            "prauc_static_lr_c": round(float(average_precision_score(y_te, p_stat_lr_c)), 4),
            # Помісячні F2
            "f2_adapt_wf": round(float(fbeta_score(y_te, (p_adapt >= wf_adapt_th).astype(int), beta=2, zero_division=0)), 4),
            "f2_static_lgb_c": round(float(fbeta_score(y_te, (p_stat_lgb_c >= static_lgb_c_th).astype(int), beta=2, zero_division=0)), 4),
            "f2_wf_roll12m_lgb_c": round(float(fbeta_score(y_te, (p_roll_lgb_c >= roll_lgb_c_th).astype(int), beta=2, zero_division=0)), 4),
            "f2_wf_roll12m_lgb_d": round(float(fbeta_score(y_te, (p_roll_lgb_d >= roll_lgb_d_th).astype(int), beta=2, zero_division=0)), 4),
            "f2_wf_roll12m_lr_c": round(float(fbeta_score(y_te, (p_roll_lr_c >= roll_lr_c_th).astype(int), beta=2, zero_division=0)), 4),
        })

        oof_chunks.append(pd.DataFrame({
            "timestamp_utc": te_df["timestamp_utc"].values,
            "date_kyiv": te_df["date_kyiv"].values,
            "year_month_utc": te_df["year_month_utc"].values,
            "y_true": y_te,
            "p_pred_all": np.ones_like(p_adapt),
            "b_pred_all": np.ones_like(y_te),
            "p_adapt": p_adapt,
            "b_adapt_static": (p_adapt >= static_adapt_th).astype(int),
            "b_adapt_wf": (p_adapt >= wf_adapt_th).astype(int),
            "p_stat_lgb_c": p_stat_lgb_c,
            "b_stat_lgb_c": (p_stat_lgb_c >= static_lgb_c_th).astype(int),
            "p_stat_lr_c": p_stat_lr_c,
            "b_stat_lr_c": (p_stat_lr_c >= static_lr_c_th).astype(int),
            "p_roll_lgb_c": p_roll_lgb_c,
            "b_roll_lgb_c": (p_roll_lgb_c >= roll_lgb_c_th).astype(int),
            "p_roll_lr_c": p_roll_lr_c,
            "b_roll_lr_c": (p_roll_lr_c >= roll_lr_c_th).astype(int),
            "p_roll_lgb_d": p_roll_lgb_d,
            "b_roll_lgb_d": (p_roll_lgb_d >= roll_lgb_d_th).astype(int),
            "p_roll_lr_d": p_roll_lr_d,
            "b_roll_lr_d": (p_roll_lr_d >= roll_lr_d_th).astype(int),
            "p_exp_lgb_c": p_exp_lgb_c,
            "b_exp_lgb_c": (p_exp_lgb_c >= exp_lgb_c_th).astype(int),
            "p_exp_lr_c": p_exp_lr_c,
            "b_exp_lr_c": (p_exp_lr_c >= exp_lr_c_th).astype(int),
            "p_exp_lgb_d": p_exp_lgb_d,
            "b_exp_lgb_d": (p_exp_lgb_d >= exp_lgb_d_th).astype(int),
            "p_exp_lr_d": p_exp_lr_d,
            "b_exp_lr_d": (p_exp_lr_d >= exp_lr_d_th).astype(int),
        }))
        print(
            f"   ✔ Місяць {m_str}: base_rate={base_rate_m:.3f} | >StaticMax={monthly_diag_rows[-1]['share_gt_static_train_max']:.1%} "
            f"-> >Roll12mMax={monthly_diag_rows[-1]['share_gt_roll12m_train_max']:.1%} | "
            f"PR-AUC: Adapt={monthly_diag_rows[-1]['prauc_adapt']:.4f}, "
            f"Static_LGB_C={monthly_diag_rows[-1]['prauc_static_lgb_c']:.4f}, "
            f"WF_Roll12m_LGB_C={monthly_diag_rows[-1]['prauc_wf_roll12m_lgb_c']:.4f}"
        )

    monthly_df = pd.DataFrame(monthly_diag_rows)
    oof_df = pd.concat(oof_chunks, ignore_index=True)

    model_meta = [
        ("Reference: Predict-All (p* = 0.0)", "None", "reference (base-rate F2 floor)"),
        ("Baseline 3: Adaptive 14-day (Static Val p*)", "None", "reference baseline (frozen Val p*)"),
        ("Baseline 3: Adaptive 14-day (WF p*_M)", "None", "PRIMARY FAIR BASELINE (WF p*_M)"),
        ("LightGBM (Set C) [Static Initial Train]", "C (22 features)", "reference static model (Val winner)"),
        ("LogReg (Set C) [Static Initial Train]", "C (22 features)", "reference static model"),
        ("LightGBM (Set C) [WF Rolling 12m]", "C (22 features)", "PRIMARY PRE-SPECIFIED HYPOTHESIS"),
        ("LogReg (Set C) [WF Rolling 12m]", "C (22 features)", "exploratory (pre-specified Set C)"),
        ("LightGBM (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]", "D (25 features, post-hoc)", "post-hoc exploratory (Set D)"),
        ("LogReg (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]", "D (25 features, post-hoc)", "post-hoc exploratory (Set D)"),
        ("LightGBM (Set C) [WF Expanding]", "C (22 features)", "exploratory (expanding window)"),
        ("LogReg (Set C) [WF Expanding]", "C (22 features)", "exploratory (expanding window)"),
        ("LightGBM (Set D: +Rel Ratios) [WF Expanding, post-hoc]", "D (25 features, post-hoc)", "post-hoc exploratory (Set D, expanding)"),
        ("LogReg (Set D: +Rel Ratios) [WF Expanding, post-hoc]", "D (25 features, post-hoc)", "post-hoc exploratory (Set D, expanding)"),
    ]

    def _build_dicts(sub_oof: pd.DataFrame):
        probas = {
            "Reference: Predict-All (p* = 0.0)": sub_oof["p_pred_all"].values,
            "Baseline 3: Adaptive 14-day (Static Val p*)": sub_oof["p_adapt"].values,
            "Baseline 3: Adaptive 14-day (WF p*_M)": sub_oof["p_adapt"].values,
            "LightGBM (Set C) [Static Initial Train]": sub_oof["p_stat_lgb_c"].values,
            "LogReg (Set C) [Static Initial Train]": sub_oof["p_stat_lr_c"].values,
            "LightGBM (Set C) [WF Rolling 12m]": sub_oof["p_roll_lgb_c"].values,
            "LogReg (Set C) [WF Rolling 12m]": sub_oof["p_roll_lr_c"].values,
            "LightGBM (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]": sub_oof["p_roll_lgb_d"].values,
            "LogReg (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]": sub_oof["p_roll_lr_d"].values,
            "LightGBM (Set C) [WF Expanding]": sub_oof["p_exp_lgb_c"].values,
            "LogReg (Set C) [WF Expanding]": sub_oof["p_exp_lr_c"].values,
            "LightGBM (Set D: +Rel Ratios) [WF Expanding, post-hoc]": sub_oof["p_exp_lgb_d"].values,
            "LogReg (Set D: +Rel Ratios) [WF Expanding, post-hoc]": sub_oof["p_exp_lr_d"].values,
        }
        preds_bin = {
            "Reference: Predict-All (p* = 0.0)": sub_oof["b_pred_all"].values,
            "Baseline 3: Adaptive 14-day (Static Val p*)": sub_oof["b_adapt_static"].values,
            "Baseline 3: Adaptive 14-day (WF p*_M)": sub_oof["b_adapt_wf"].values,
            "LightGBM (Set C) [Static Initial Train]": sub_oof["b_stat_lgb_c"].values,
            "LogReg (Set C) [Static Initial Train]": sub_oof["b_stat_lr_c"].values,
            "LightGBM (Set C) [WF Rolling 12m]": sub_oof["b_roll_lgb_c"].values,
            "LogReg (Set C) [WF Rolling 12m]": sub_oof["b_roll_lr_c"].values,
            "LightGBM (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]": sub_oof["b_roll_lgb_d"].values,
            "LogReg (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]": sub_oof["b_roll_lr_d"].values,
            "LightGBM (Set C) [WF Expanding]": sub_oof["b_exp_lgb_c"].values,
            "LogReg (Set C) [WF Expanding]": sub_oof["b_exp_lr_c"].values,
            "LightGBM (Set D: +Rel Ratios) [WF Expanding, post-hoc]": sub_oof["b_exp_lgb_d"].values,
            "LogReg (Set D: +Rel Ratios) [WF Expanding, post-hoc]": sub_oof["b_exp_lr_d"].values,
        }
        return probas, preds_bin

    # 3. Оцінка на Повному Test (Out-of-Fold 2025-06 .. end)
    print(f"\n🔄 [{dataset_name}] Day-Block Bootstrap (N=500) на повному Out-of-Fold Test...")
    probas_full, preds_full = _build_dicts(oof_df)
    df_full_res, direct_full = evaluate_walk_forward_predictions(
        oof_df["date_kyiv"].values,
        oof_df["year_month_utc"].values,
        oof_df["y_true"].values,
        probas_full,
        preds_full,
        model_meta,
        n_bootstraps=500,
    )

    # 4. Оцінка на Останніх 60 днях
    max_ts = pd.to_datetime(oof_df["timestamp_utc"]).max()
    rec_cutoff = max_ts - pd.Timedelta(days=60)
    rec_oof = oof_df[pd.to_datetime(oof_df["timestamp_utc"]) >= rec_cutoff].copy()
    print(f"🔄 [{dataset_name}] Day-Block Bootstrap (N=500) на Останніх 60 днях ({rec_cutoff.date()} — {max_ts.date()})...")
    probas_rec, preds_rec = _build_dicts(rec_oof)
    df_rec_res, direct_rec = evaluate_walk_forward_predictions(
        rec_oof["date_kyiv"].values,
        rec_oof["year_month_utc"].values,
        rec_oof["y_true"].values,
        probas_rec,
        preds_rec,
        model_meta,
        n_bootstraps=500,
    )

    elapsed_sec = round(time.perf_counter() - t0, 2)

    # Збереження артефактів
    p_fx = f"{file_prefix}_" if file_prefix else ""
    monthly_csv = out_dir / f"{p_fx}walk_forward_monthly_metrics.csv"
    full_csv = out_dir / f"{p_fx}walk_forward_overall_benchmark.csv"
    rec_csv = out_dir / f"{p_fx}walk_forward_recent_60d_benchmark.csv"
    details_json = out_dir / f"{p_fx}walk_forward_details.json"

    monthly_df.to_csv(monthly_csv, index=False)
    df_full_res.to_csv(full_csv, index=False)
    df_rec_res.to_csv(rec_csv, index=False)

    details = {
        "dataset": dataset_name,
        "parquet_path": str(parquet_path.relative_to(PROJECT_ROOT)),
        "target_interval_definition": "[T, T+1h)",
        "embargo_hours": 2,
        "primary_hypothesis": {
            "model": "LightGBM (Set C) [WF Rolling 12m]",
            "comparator": "Baseline 3: Adaptive 14-day (WF p*_M)",
            "metric": "Paired Delta PR-AUC (Day-Block Bootstrap N=500, seed=42)",
        },
        "measured_runtime_seconds": elapsed_sec,
        "n_test_months": len(test_months),
        "test_months": test_months,
        "full_test_rows": int(len(oof_df)),
        "full_test_positives": int(oof_df["y_true"].sum()),
        "full_test_base_rate": float(oof_df["y_true"].mean()),
        "recent_60d_rows": int(len(rec_oof)),
        "recent_60d_positives": int(rec_oof["y_true"].sum()),
        "recent_60d_base_rate": float(rec_oof["y_true"].mean()),
        "direct_comparisons_full_test": direct_full,
        "direct_comparisons_recent_60d": direct_rec,
    }
    details_json.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cols_print = [
        "model", "protocol_status", "pooled_pr_auc", "pr_auc_ci_95",
        "mean_monthly_pr_auc", "delta_prauc_vs_adapt_wf_ci_95",
        "prec_at_adapt_recall", "f2_score", "f2_ci_95",
    ]
    print("\n" + "=" * 125)
    print(f"📊 [{dataset_name.upper()}] WALK-FORWARD ПІДСУМОК НА ПОВНОМУ TEST (Час виконання: {elapsed_sec:.2f} с):")
    print("=" * 125)
    print(df_full_res[cols_print].to_string(index=False))

    print("\n" + "=" * 125)
    print(f"📊 [{dataset_name.upper()}] WALK-FORWARD ПІДСУМОК НА ОСТАННІХ 60 ДНЯХ:")
    print("=" * 125)
    print(df_rec_res[cols_print].to_string(index=False))

    print("\n📌 ПРЯМЕ ПОРІВНЯННЯ (WF − Adaptive) проти (Static − Adaptive) та (WF − Static):")
    for k, v in direct_full.items():
        print(
            f"   [Full Test]  {k}: (WF-Adapt)={v['wf_minus_adapt_prauc_mean']:+.4f} {v['wf_minus_adapt_prauc_ci_95']} vs "
            f"(Ref-Adapt)={v['ref_minus_adapt_prauc_mean']:+.4f} {v['ref_minus_adapt_prauc_ci_95']} | "
            f"(WF-Ref)={v['wf_minus_ref_prauc_mean']:+.4f} {v['wf_minus_ref_prauc_ci_95']}"
        )
    for k, v in direct_rec.items():
        print(
            f"   [Recent 60d] {k}: (WF-Adapt)={v['wf_minus_adapt_prauc_mean']:+.4f} {v['wf_minus_adapt_prauc_ci_95']} vs "
            f"(Ref-Adapt)={v['ref_minus_adapt_prauc_mean']:+.4f} {v['ref_minus_adapt_prauc_ci_95']} | "
            f"(WF-Ref)={v['wf_minus_ref_prauc_mean']:+.4f} {v['wf_minus_ref_prauc_ci_95']}"
        )

    return {
        "full_df": df_full_res,
        "rec_df": df_rec_res,
        "monthly_df": monthly_df,
        "details": details,
    }


def plot_walk_forward_comparison(off_res: dict, vol_res: dict, comp_dir: Path) -> None:
    """Будує 4-панельний графік 19: діагностика гіпотези та помісячна динаміка Walk-Forward vs Static."""
    fig_dir = comp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11.5), dpi=300)

    # Panel A: Volunteer monthly PR-AUC (Adaptive vs Static LGB C vs WF Rolling 12m LGB C vs WF LGB D)
    m_vol = vol_res["monthly_df"]
    xv = np.arange(len(m_vol))
    ax1 = axes[0, 0]
    ax1.plot(xv, m_vol["prauc_adapt"], marker="o", lw=2.3, color="#2ca02c", label="Baseline 3: Adaptive 14-day")
    ax1.plot(xv, m_vol["prauc_static_lgb_c"], marker="s", lw=2.0, ls="--", color="#7f7f7f", label="Static LightGBM (Set C)")
    ax1.plot(xv, m_vol["prauc_wf_roll12m_lgb_c"], marker="^", lw=2.5, color="#d62728", label="WF Rolling 12m LightGBM (Set C) [Primary]")
    ax1.plot(xv, m_vol["prauc_wf_roll12m_lgb_d"], marker="D", lw=2.0, ls=":", color="#8c2d04", label="WF Rolling 12m LightGBM (Set D, post-hoc)")
    ax1.plot(xv, m_vol["prauc_wf_roll12m_lr_c"], marker="v", lw=2.0, color="#08519c", label="WF Rolling 12m LogReg (Set C)")
    ax1.set_title("А. Volunteer (naive_fixed_D): Помісячний PR-AUC (Static vs Walk-Forward)", fontsize=12.5, fontweight="bold")
    ax1.set_ylabel("PR-AUC у межах місяця M", fontsize=11)
    ax1.set_xticks(xv)
    ax1.set_xticklabels(m_vol["month"], rotation=35, ha="right", fontsize=9)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.8)

    # Panel B: Official monthly PR-AUC
    m_off = off_res["monthly_df"]
    xo = np.arange(len(m_off))
    ax2 = axes[0, 1]
    ax2.plot(xo, m_off["prauc_adapt"], marker="o", lw=2.3, color="#2ca02c", label="Baseline 3: Adaptive 14-day")
    ax2.plot(xo, m_off["prauc_static_lgb_c"], marker="s", lw=2.0, ls="--", color="#7f7f7f", label="Static LightGBM (Set C)")
    ax2.plot(xo, m_off["prauc_wf_roll12m_lgb_c"], marker="^", lw=2.5, color="#d62728", label="WF Rolling 12m LightGBM (Set C) [Primary]")
    ax2.plot(xo, m_off["prauc_wf_roll12m_lgb_d"], marker="D", lw=2.0, ls=":", color="#8c2d04", label="WF Rolling 12m LightGBM (Set D, post-hoc)")
    ax2.plot(xo, m_off["prauc_wf_roll12m_lr_c"], marker="v", lw=2.0, color="#08519c", label="WF Rolling 12m LogReg (Set C)")
    ax2.set_title("Б. Official: Помісячний PR-AUC (Static vs Walk-Forward)", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("PR-AUC у межах місяця M", fontsize=11)
    ax2.set_xticks(xo)
    ax2.set_xticklabels(m_off["month"], rotation=35, ha="right", fontsize=9)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.8)

    # Panel C: Diagnostics on Volunteer - share of rows exceeding Train max & P99
    ax3 = axes[1, 0]
    ax3.plot(xv, m_vol["share_gt_static_train_p99"] * 100, marker="o", lw=2.2, ls="--", color="#e6550d", label="% рядків > P99(Static Train 2022–2024)")
    ax3.plot(xv, m_vol["share_gt_static_train_max"] * 100, marker="s", lw=2.2, color="#d62728", label="% рядків > Max(Static Train = 0.414)")
    ax3.plot(xv, m_vol["share_gt_roll12m_train_p99"] * 100, marker="^", lw=2.2, ls="--", color="#3182bd", label="% рядків > P99(Rolling 12m Train_M)")
    ax3.plot(xv, m_vol["share_gt_roll12m_train_max"] * 100, marker="D", lw=2.4, color="#08519c", label="% рядків > Max(Rolling 12m Train_M)")
    ax3.set_title("В. Діагностика виходу adaptive_rate_14d за межі Train (Volunteer)", fontsize=12.5, fontweight="bold")
    ax3.set_ylabel("Частка спокійних годин у місяці M (%)", fontsize=11)
    ax3.set_xticks(xv)
    ax3.set_xticklabels(m_vol["month"], rotation=35, ha="right", fontsize=9)
    ax3.legend(loc="upper left", frameon=True, fontsize=9)

    # Panel D: Mean Predicted Probability vs Actual Base Rate across months (Volunteer)
    ax4 = axes[1, 1]
    ax4.plot(xv, m_vol["actual_base_rate"] * 100, marker="o", lw=2.8, color="black", label="Фактична частота стартів (Base Rate %)")
    ax4.plot(xv, m_vol["mean_pred_adapt14d"] * 100, marker="s", lw=2.2, color="#2ca02c", label="Середній прогноз Adaptive 14-day (%)")
    ax4.plot(xv, m_vol["mean_pred_static_lgb_c"] * 100, marker="v", lw=2.0, ls="--", color="#7f7f7f", label="Середній прогноз Static LightGBM Set C (%)")
    ax4.plot(xv, m_vol["mean_pred_wf_roll12m_lgb_c"] * 100, marker="^", lw=2.3, color="#d62728", label="Середній прогноз WF Rolling 12m LightGBM Set C (%)")
    ax4.set_title("Г. Фактична частота стартів vs Середній прогноз по місяцях (Volunteer)", fontsize=12.5, fontweight="bold")
    ax4.set_ylabel("Частота / Ймовірність (%)", fontsize=11)
    ax4.set_xticks(xv)
    ax4.set_xticklabels(m_vol["month"], rotation=35, ha="right", fontsize=9)
    ax4.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle(
        "Експеримент Walk-Forward (Ковзне перенавчання 12m + 2m Val): перевірка гіпотези нестаціонарності",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig_path = fig_dir / "19_walk_forward_vs_static_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"🎨 Збережено графік Walk-Forward: {fig_path}")


def main():
    t_total_start = time.perf_counter()
    off_dir = PROJECT_ROOT / "reports" / "official"
    vol_dir = PROJECT_ROOT / "reports" / "volunteer"
    comp_dir = PROJECT_ROOT / "reports" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    off_res = run_walk_forward_for_dataset(
        dataset_name="Official",
        parquet_path=PROJECT_ROOT / "data" / "processed" / "ml_features_hourly.parquet",
        out_dir=off_dir,
        file_prefix="",
    )
    vol_res = run_walk_forward_for_dataset(
        dataset_name="Volunteer (naive_fixed_D)",
        parquet_path=PROJECT_ROOT / "data" / "processed" / "volunteer_ml_features_hourly_fixed_d.parquet",
        out_dir=vol_dir,
        file_prefix="volunteer",
    )

    # Об'єднана порівняльна таблиця між Official та Volunteer
    frames = []
    for slice_label, df_o, df_v in [
        ("Full Test (Out-of-Fold)", off_res["full_df"], vol_res["full_df"]),
        ("Recent 60d", off_res["rec_df"], vol_res["rec_df"]),
    ]:
        keep = [
            "model", "protocol_status", "pooled_pr_auc", "pr_auc_ci_95",
            "mean_monthly_pr_auc", "delta_prauc_vs_adapt_wf", "delta_prauc_vs_adapt_wf_ci_95",
            "prec_at_adapt_recall", "f2_score", "f2_ci_95",
        ]
        m = df_o[keep].rename(columns={c: f"off_{c}" for c in keep if c not in ("model", "protocol_status")}).merge(
            df_v[keep].drop(columns=["protocol_status"]).rename(columns={c: f"vol_{c}" for c in keep if c not in ("model", "protocol_status")}),
            on="model",
            how="inner",
        )
        m.insert(0, "slice", slice_label)
        frames.append(m)

    comp_df = pd.concat(frames, ignore_index=True)
    comp_csv = comp_dir / "official_vs_volunteer_walk_forward_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n💾 Збережено зведену таблицю Walk-Forward (Official vs Volunteer): {comp_csv}")

    plot_walk_forward_comparison(off_res, vol_res, comp_dir)
    total_sec = round(time.perf_counter() - t_total_start, 2)
    print(f"\n⏱️ Загальний виміряний час виконання Walk-Forward (обидва джерела): {total_sec:.2f} с")


if __name__ == "__main__":
    main()
