"""
Модуль навчання моделей машинного навчання та тюнінгу порогів (Фаза 2.3).
Реалізує:
1. Logistic Regression (L2, class_weight='balanced')
2. LightGBM (Gradient Boosted Decision Trees)
3. Ablation Study:
   - Підмножина A: Тільки часові та історичні фактори (No Spatial)
   - Підмножина B: Повний набір з просторовими лагами сусідів (With Spatial)
4. Пошук оптимального порогу p* = argmax F2(p) виключно на Validation
5. Збереження навчених моделей та артефактів для Phase 2.4 (Test evaluation)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score
)
import lightgbm as lgb

FEATURE_SET_A = [
    'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'month',
    'burden_last_6h', 'burden_last_24h',
    'cluster_1_rear', 'cluster_2_center', 'cluster_3_front'
]

FEATURE_SET_B = FEATURE_SET_A + [
    'neighbors_active_count', 'neighbor_any_active', 'neighbor_active_ratio'
]


def load_datasets(parquet_path: str = "data/processed/ml_features_hourly.parquet"):
    """Завантажує набори Train та Validation з паркету."""
    print(f"Завантаження даних із {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    train_df = df[df['split_set'] == 'train'].copy()
    val_df = df[df['split_set'] == 'val'].copy()
    
    print(f"Train розмір: {train_df.shape}, Val розмір: {val_df.shape}")
    print(f"Train позитивні: {train_df['target_onset_next_1h'].sum()} ({train_df['target_onset_next_1h'].mean():.4%})")
    print(f"Val позитивні: {val_df['target_onset_next_1h'].sum()} ({val_df['target_onset_next_1h'].mean():.4%})")
    return train_df, val_df


def evaluate_threshold_grid(y_true, y_probas, thresholds=None):
    """
    Сканує сітку порогів та знаходить оптимальний поріг p* для максимізації F2.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 150)
        
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
            'threshold': th,
            'precision': p,
            'recall': r,
            'f2_score': f2,
            'pr_auc': pr_auc,
            'brier_score': brier
        }
        records.append(row)
        
        if f2 > best_f2:
            best_f2 = f2
            best_record = row
            
    df_curve = pd.DataFrame(records)
    return best_record, df_curve


def train_logistic_regression(X_train, y_train, feature_names):
    """Навчає масштабовану логістичну регресію."""
    print(f"Навчання Logistic Regression ({len(feature_names)} ознак)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_names])
    
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names):
    """
    Навчає LightGBM з ранньою зупинкою на Validation вибірці.
    Використовує стандартний logloss для збереження калібрування ймовірностей.
    """
    print(f"Навчання LightGBM ({len(feature_names)} ознак)...")
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
        verbosity=-1
    )
    
    model.fit(
        X_train[feature_names],
        y_train,
        eval_set=[(X_val[feature_names], y_val)],
        eval_names=['val'],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    return model


def run_pipeline():
    """Повний цикл навчання, валідації та оцінки для фази 2.3."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    train_df, val_df = load_datasets()
    
    y_train = train_df['target_onset_next_1h'].values
    y_val = val_df['target_onset_next_1h'].values
    
    # 1. Завантаження результатів бейзлайнів
    baselines_df = pd.read_csv("reports/baselines_benchmark.csv")
    
    # 2. Модель 1A: Logistic Regression (No Spatial)
    lr_a, scaler_a = train_logistic_regression(train_df, y_train, FEATURE_SET_A)
    p_val_lr_a = lr_a.predict_proba(scaler_a.transform(val_df[FEATURE_SET_A]))[:, 1]
    best_lr_a, curve_lr_a = evaluate_threshold_grid(y_val, p_val_lr_a)
    joblib.dump(lr_a, "models/logreg_nospatial.joblib")
    joblib.dump(scaler_a, "models/scaler_nospatial.joblib")
    
    # 3. Модель 1B: Logistic Regression (With Spatial)
    lr_b, scaler_b = train_logistic_regression(train_df, y_train, FEATURE_SET_B)
    p_val_lr_b = lr_b.predict_proba(scaler_b.transform(val_df[FEATURE_SET_B]))[:, 1]
    best_lr_b, curve_lr_b = evaluate_threshold_grid(y_val, p_val_lr_b)
    joblib.dump(lr_b, "models/logreg_spatial.joblib")
    joblib.dump(scaler_b, "models/scaler_spatial.joblib")
    
    # 4. Модель 2A: LightGBM (No Spatial)
    lgb_a = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_A)
    p_val_lgb_a = lgb_a.predict_proba(val_df[FEATURE_SET_A])[:, 1]
    best_lgb_a, curve_lgb_a = evaluate_threshold_grid(y_val, p_val_lgb_a)
    joblib.dump(lgb_a, "models/lightgbm_nospatial.joblib")
    
    # 5. Модель 2B: LightGBM (With Spatial)
    lgb_b = train_lightgbm_model(train_df, y_train, val_df, y_val, FEATURE_SET_B)
    p_val_lgb_b = lgb_b.predict_proba(val_df[FEATURE_SET_B])[:, 1]
    best_lgb_b, curve_lgb_b = evaluate_threshold_grid(y_val, p_val_lgb_b)
    joblib.dump(lgb_b, "models/lightgbm_spatial.joblib")
    
    # Збереження порогів
    optimal_thresholds = {
        'Historical Lookup': float(baselines_df.loc[baselines_df['model'].str.contains('Historical'), 'threshold'].iloc[0]),
        'LogReg (No Spatial)': float(best_lr_a['threshold']),
        'LogReg (With Spatial)': float(best_lr_b['threshold']),
        'LightGBM (No Spatial)': float(best_lgb_a['threshold']),
        'LightGBM (With Spatial)': float(best_lgb_b['threshold']),
    }
    with open("models/optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)
        
    # Формування підсумкової таблиці бенчмарку на Validation
    benchmark_rows = []
    
    # Додаємо бейзлайни
    for _, r in baselines_df.iterrows():
        benchmark_rows.append({
            'model': r['model'],
            'feature_set': 'None',
            'threshold': r['threshold'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f2_score': r['f2_score'],
            'pr_auc': r['pr_auc'],
            'brier_score': r['brier_score']
        })
        
    # Додаємо ML моделі
    models_meta = [
        ("LogReg (No Spatial)", "A (Time & Lags)", best_lr_a),
        ("LogReg (With Spatial)", "B (Time, Lags & Neighbors)", best_lr_b),
        ("LightGBM (No Spatial)", "A (Time & Lags)", best_lgb_a),
        ("LightGBM (With Spatial)", "B (Time, Lags & Neighbors)", best_lgb_b),
    ]
    
    for name, fset, res in models_meta:
        benchmark_rows.append({
            'model': name,
            'feature_set': fset,
            'threshold': res['threshold'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f2_score': res['f2_score'],
            'pr_auc': res['pr_auc'],
            'brier_score': res['brier_score']
        })
        
    res_df = pd.DataFrame(benchmark_rows)
    res_df.to_csv("reports/ml_validation_benchmark.csv", index=False)
    print("\n" + "="*80)
    print("ПІДСУМКОВА ТАБЛИЦЯ ВАЛІДАЦІЇ (Validation Set):")
    print("="*80)
    print(res_df.to_string(index=False))
    
    # 6. Побудова графіків
    plot_curves(
        y_val,
        probas_dict={
            'LogReg (No Spatial)': p_val_lr_a,
            'LogReg (With Spatial)': p_val_lr_b,
            'LightGBM (No Spatial)': p_val_lgb_a,
            'LightGBM (With Spatial)': p_val_lgb_b
        },
        curves_dict={
            'LogReg (No Spatial)': curve_lr_a,
            'LogReg (With Spatial)': curve_lr_b,
            'LightGBM (No Spatial)': curve_lgb_a,
            'LightGBM (With Spatial)': curve_lgb_b
        },
        optimal_thresholds=optimal_thresholds
    )
    print("Графіки збережено успішно.")


def plot_curves(y_val, probas_dict, curves_dict, optimal_thresholds):
    """Будує Precision-Recall криві та графіки F2 від порогу."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Графік 1: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    colors = {
        'LogReg (No Spatial)': '#1f77b4',
        'LogReg (With Spatial)': '#004c6d',
        'LightGBM (No Spatial)': '#ff7f0e',
        'LightGBM (With Spatial)': '#d62728'
    }
    
    # No skill baseline (частота класу)
    base_rate = np.mean(y_val)
    ax.axhline(base_rate, color='gray', linestyle='--', label=f'Випадковий вибір (Base rate = {base_rate:.3f})')
    
    for name, probas in probas_dict.items():
        precision, recall, _ = precision_recall_curve(y_val, probas)
        auc_score = average_precision_score(y_val, probas)
        ax.plot(
            recall, precision,
            label=f"{name} (PR-AUC = {auc_score:.4f})",
            color=colors.get(name, 'black'),
            linewidth=2.2
        )
        
    ax.set_title("Порівняння Precision-Recall кривих на вибірці Validation\n(Ablation Study: Вплив просторових лагів)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (Повнота виявлення початку тривог)", fontsize=12)
    ax.set_ylabel("Precision (Точність сповіщень)", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 0.70])
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig("reports/figures/08_ml_pr_curves_validation.png")
    plt.close(fig)
    
    # Графік 2: F2 Score vs Classification Threshold
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    for name, curve in curves_dict.items():
        opt_th = optimal_thresholds[name]
        best_f2 = curve.loc[(curve['threshold'] - opt_th).abs().idxmin(), 'f2_score']
        line, = ax.plot(
            curve['threshold'], curve['f2_score'],
            label=f"{name} (max F2 = {best_f2:.4f} @ p*={opt_th:.3f})",
            color=colors.get(name, 'black'),
            linewidth=2.0
        )
        ax.scatter([opt_th], [best_f2], color=colors.get(name, 'black'), s=70, zorder=5)
        
    ax.set_title("Підбір оптимального порогу класифікації $p^*$ на Validation\n(Максимізація $F_2$-Score для пріоритету виявлення загроз)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Поріг класифікації ($p$)", fontsize=12)
    ax.set_ylabel("$F_2$-Score (пріоритет Recall)", fontsize=12)
    ax.set_xlim([0.01, 0.50])
    ax.legend(loc='lower center', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig("reports/figures/09_threshold_tuning_f2.png")
    plt.close(fig)


if __name__ == "__main__":
    run_pipeline()
