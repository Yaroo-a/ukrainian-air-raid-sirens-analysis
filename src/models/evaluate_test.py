"""Модуль фінальної оцінки на вибірці Test (червень 2025 - вересень 2026) (Фаза 2.4 - Refactored).

Реалізує:
1. Оцінку моделей на вибірці Test із зафіксованими на Validation порогами p*.
2. Узгодження часових поясів (Київський час dt.tz_convert("Europe/Kyiv").dt.hour ідентично до Val).
3. Оцінку True Naive Persistence (T-1) та Adaptive 14-day Rolling Baseline.
4. Розрахунок чесного Lift over BEST Baseline на Test.
5. Reliability Diagram (калібрувальна крива) для LightGBM.
6. Оцінку на свіжому зрізі (останні 60 днів вибірки: липень-вересень 2026).
7. Довірчі інтервали (Bootstrap 95% CI) для PR-AUC та F2.
8. Побудову оновлених графіків та підсумкової таблиці reports/ml_models_benchmark.csv.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

FEATURE_SET_A = [
    'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'month',
    'burden_last_6h', 'burden_last_24h',
    'cluster_1_low_activity', 'cluster_2_medium_activity', 'cluster_3_high_activity'
]

FEATURE_SET_B = FEATURE_SET_A + [
    'neighbors_active_count', 'neighbor_any_active', 'neighbor_active_ratio'
]


def load_test_data_and_models(parquet_path: str = "data/processed/ml_features_hourly.parquet"):
    """Завантажує тестовий набір даних, таблицю бейзлайну та натреновані моделі."""
    print("Завантаження даних та моделей...")
    df = pd.read_parquet(parquet_path)
    train_df = df[df['split_set'] == 'train'].copy()
    test_df = df[df['split_set'] == 'test'].copy()
    
    with open("models/optimal_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    # Таблиця шансів Historical Lookup з Train (Київський час!)
    train_df["hour_kyiv"] = train_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    test_df["hour_kyiv"] = test_df["timestamp_utc"].dt.tz_convert("Europe/Kyiv").dt.hour
    
    lookup = train_df.groupby(['day_of_week', 'hour_kyiv', 'cluster_id'])['target_onset_next_1h'].mean().reset_index()
    lookup.rename(columns={'target_onset_next_1h': 'lookup_prob'}, inplace=True)
    global_mean = train_df['target_onset_next_1h'].mean()
    
    test_merged = test_df.merge(lookup, on=['day_of_week', 'hour_kyiv', 'cluster_id'], how='left')
    test_merged['lookup_prob'] = test_merged['lookup_prob'].fillna(global_mean)
    
    models = {
        'scaler_a': joblib.load('models/scaler_nospatial.joblib'),
        'scaler_b': joblib.load('models/scaler_spatial.joblib'),
        'lr_a': joblib.load('models/logreg_nospatial.joblib'),
        'lr_b': joblib.load('models/logreg_spatial.joblib'),
        'lgb_a': joblib.load('models/lightgbm_nospatial.joblib'),
        'lgb_b': joblib.load('models/lightgbm_spatial.joblib'),
    }
    
    return test_merged, thresholds, models


def bootstrap_metric_ci(y_true, y_prob, threshold, n_bootstraps=200, random_state=42):
    """Розраховує 95% довірчий інтервал через Bootstrap."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    f2_list = []
    prauc_list = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        if y_t.sum() == 0:
            continue
        preds = (y_p >= threshold).astype(int)
        f2_list.append(fbeta_score(y_t, preds, beta=2, zero_division=0))
        prauc_list.append(average_precision_score(y_t, y_p))
        
    return {
        'f2_ci_low': np.percentile(f2_list, 2.5),
        'f2_ci_high': np.percentile(f2_list, 97.5),
        'prauc_ci_low': np.percentile(prauc_list, 2.5),
        'prauc_ci_high': np.percentile(prauc_list, 97.5)
    }


def run_test_evaluation():
    """Проводить тестування всіх моделей і зберігає фінальні звіти."""
    os.makedirs("reports/figures", exist_ok=True)
    test_df, thresholds, models = load_test_data_and_models()
    
    y_test = test_df['target_onset_next_1h'].values
    print(f"Розмір Test: {len(test_df):,}, позитивних: {y_test.sum():,} ({y_test.mean():.4%})")
    
    # 1. Прогнози ймовірностей
    probas = {}
    
    # True Naive Persistence (лаг T-1 з повної сітки)
    probas['Baseline 1: True Naive Persistence'] = test_df['lagged_onset_1h'].values.astype(float)
    
    # Historical Lookup (узгоджений Київський час)
    probas['Baseline 2: Historical Lookup Table'] = test_df['lookup_prob'].values
    
    # Adaptive 14-day Rolling Baseline
    probas['Baseline 3: Adaptive 14-day Rolling'] = test_df['adaptive_rate_14d'].values
    
    # Logistic Regression
    probas['LogReg (No Spatial)'] = models['lr_a'].predict_proba(models['scaler_a'].transform(test_df[FEATURE_SET_A]))[:, 1]
    probas['LogReg (With Spatial)'] = models['lr_b'].predict_proba(models['scaler_b'].transform(test_df[FEATURE_SET_B]))[:, 1]
    
    # LightGBM
    probas['LightGBM (No Spatial)'] = models['lgb_a'].predict_proba(test_df[FEATURE_SET_A])[:, 1]
    probas['LightGBM (With Spatial)'] = models['lgb_b'].predict_proba(test_df[FEATURE_SET_B])[:, 1]
    
    # 2. Оцінка метрик із зафіксованими порогами
    configs = [
        ('Baseline 1: True Naive Persistence', 'None', 0.500),
        ('Baseline 2: Historical Lookup Table', 'None', thresholds['Baseline 2: Historical Lookup Table']),
        ('Baseline 3: Adaptive 14-day Rolling', 'None', thresholds['Baseline 3: Adaptive 14-day Rolling']),
        ('LogReg (No Spatial)', 'A (Time & Lags)', thresholds['LogReg (No Spatial)']),
        ('LogReg (With Spatial)', 'B (Time, Lags & Neighbors)', thresholds['LogReg (With Spatial)']),
        ('LightGBM (No Spatial)', 'A (Time & Lags)', thresholds['LightGBM (No Spatial)']),
        ('LightGBM (With Spatial)', 'B (Time, Lags & Neighbors)', thresholds['LightGBM (With Spatial)']),
    ]
    
    results = []
    
    for name, fset, th in configs:
        p_vals = probas[name]
        preds = (p_vals >= th).astype(int)
        
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
        pr_auc = average_precision_score(y_test, p_vals) if 'Naive' not in name else np.nan
        brier = brier_score_loss(y_test, p_vals)
        
        ci_stats = bootstrap_metric_ci(y_test, p_vals, th, n_bootstraps=50) if 'Naive' not in name else {'f2_ci_low': np.nan, 'f2_ci_high': np.nan, 'prauc_ci_low': np.nan, 'prauc_ci_high': np.nan}
        
        results.append({
            'model': name,
            'feature_set': fset,
            'frozen_threshold': th,
            'precision': prec,
            'recall': rec,
            'f2_score': f2,
            'f2_ci_95': f"[{ci_stats['f2_ci_low']:.3f}, {ci_stats['f2_ci_high']:.3f}]" if not np.isnan(ci_stats['f2_ci_low']) else "-",
            'pr_auc': pr_auc,
            'pr_auc_ci_95': f"[{ci_stats['prauc_ci_low']:.3f}, {ci_stats['prauc_ci_high']:.3f}]" if not np.isnan(ci_stats['prauc_ci_low']) else "-",
            'brier_score': brier
        })
        
    df_results = pd.DataFrame(results)
    
    # 3. Чесний Lift над НАЙКРАЩИМ бейзлайном на Test (Baseline 3: Adaptive 14-day Rolling)
    best_base_f2 = df_results[df_results['model'].str.contains('Baseline')]['f2_score'].max()
    best_base_name = df_results.loc[df_results['f2_score'] == best_base_f2, 'model'].iloc[0]
    best_base_prauc = df_results.loc[df_results['model'] == best_base_name, 'pr_auc'].iloc[0]
    
    print(f"\n🏆 Найкращий бейзлайн на Test: {best_base_name} (F2 = {best_base_f2:.4f}, PR-AUC = {best_base_prauc:.4f})")
    
    df_results['lift_f2_over_best_baseline_pct'] = ((df_results['f2_score'] - best_base_f2) / best_base_f2) * 100
    df_results['lift_prauc_over_best_baseline_pct'] = ((df_results['pr_auc'] - best_base_prauc) / best_base_prauc) * 100
    
    df_results.to_csv("reports/ml_models_benchmark.csv", index=False)
    print("\n" + "="*95)
    print("ФІНАЛЬНИЙ БЕНЧМАРК НА ТЕСТОВІЙ ВИБІРЦІ (TEST: 2025-06 - 2026-09):")
    print("="*95)
    print(df_results[['model', 'feature_set', 'frozen_threshold', 'precision', 'recall', 'f2_score', 'f2_ci_95', 'pr_auc', 'brier_score', 'lift_f2_over_best_baseline_pct']].to_string(index=False))
    
    # 4. Оцінка на останньому часовому періоді (останні 60 днів)
    test_max_time = test_df['timestamp_utc'].max()
    recent_cutoff = test_max_time - pd.Timedelta(days=60)
    recent_mask = test_df['timestamp_utc'] >= recent_cutoff
    print(f"\n⏱️ Оцінка на останніх 60 днях (від {recent_cutoff.date()} до {test_max_time.date()}, рядків: {recent_mask.sum():,}):")
    y_recent = y_test[recent_mask]
    for m in ['Baseline 3: Adaptive 14-day Rolling', 'LightGBM (With Spatial)']:
        th = thresholds.get(m, 0.10)
        p_sub = probas[m][recent_mask]
        preds_sub = (p_sub >= th).astype(int)
        f2_rec = fbeta_score(y_recent, preds_sub, beta=2, zero_division=0)
        prauc_rec = average_precision_score(y_recent, p_sub)
        print(f"   {m:35s}: F2 = {f2_rec:.4f}, PR-AUC = {prauc_rec:.4f}, Positives = {y_recent.sum()}")
        
    # 5. Feature Importance для LightGBM
    lgb_spatial = models['lgb_b']
    importance_gain = lgb_spatial.booster_.feature_importance(importance_type='gain')
    importance_split = lgb_spatial.booster_.feature_importance(importance_type='split')
    
    feat_names_ua = {
        'burden_last_24h': 'Навантаження за 24 год (добовий темп)',
        'cluster_3_high_activity': 'Кластер 3 (Висока активність / Фронт & Кордон)',
        'neighbors_active_count': 'Кількість активних сусідів (RQ3)',
        'month': 'Місяць (сезонність)',
        'hour_sin': 'Година доби (синус - час)',
        'hour_cos': 'Година доби (косинус)',
        'cluster_1_low_activity': 'Кластер 1 (Низька активність / Тил)',
        'neighbor_active_ratio': 'Частка активних сусідів (RQ3)',
        'cluster_2_medium_activity': 'Кластер 2 (Середня активність / Центр)',
        'day_of_week': 'День тижня (тижневий ритм)',
        'neighbor_any_active': 'Чи активний бодай один сусід (RQ3)',
        'burden_last_6h': 'Навантаження за 6 год (тактичний темп)',
        'is_weekend': 'Вихідний день'
    }
    
    df_imp = pd.DataFrame({
        'feature': FEATURE_SET_B,
        'feature_ua': [feat_names_ua.get(f, f) for f in FEATURE_SET_B],
        'gain': importance_gain,
        'split': importance_split
    }).sort_values('gain', ascending=False)
    
    df_imp.to_csv("reports/feature_importance_lightgbm.csv", index=False)
    
    # 6. Графіки
    plot_test_figures(y_test, probas, df_imp, df_results, best_base_name)
    print("Всі графіки та звіти збережено успішно.")


def plot_test_figures(y_test, probas, df_imp, df_results, best_base_name):
    """Створює графіки PR-кривих на Test, Feature Importance, Lift та Reliability Diagram."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Графік 1: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    colors = {
        'Baseline 2: Historical Lookup Table': '#7f7f7f',
        'Baseline 3: Adaptive 14-day Rolling': '#2ca02c',
        'LogReg (No Spatial)': '#1f77b4',
        'LogReg (With Spatial)': '#004c6d',
        'LightGBM (No Spatial)': '#ff7f0e',
        'LightGBM (With Spatial)': '#d62728'
    }
    
    base_rate = np.mean(y_test)
    ax.axhline(base_rate, color='darkgray', linestyle='--', label=f'Base rate = {base_rate:.3%}')
    
    for name in ['Baseline 2: Historical Lookup Table', 'Baseline 3: Adaptive 14-day Rolling',
                 'LogReg (No Spatial)', 'LogReg (With Spatial)',
                 'LightGBM (No Spatial)', 'LightGBM (With Spatial)']:
        p_vals = probas[name]
        prec, rec, _ = precision_recall_curve(y_test, p_vals)
        auc_score = average_precision_score(y_test, p_vals)
        ax.plot(
            rec, prec,
            label=f"{name} (PR-AUC = {auc_score:.4f})",
            color=colors.get(name, 'black'),
            linewidth=2.2 if 'LightGBM' in name else 1.8
        )
        
    ax.set_title("Precision-Recall криві на вибірці TEST (2025–2026)\nЧесне змагання з адаптивним бейзлайном", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (Повнота виявлення)", fontsize=11)
    ax.set_ylabel("Precision (Точність сповіщень)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 0.60])
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig("reports/figures/10_test_pr_curves_comparison.png")
    plt.close(fig)
    
    # Графік 2: Feature Importance (Gain)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    top_imp = df_imp.copy()
    palette = []
    for f in top_imp['feature']:
        if 'neighbor' in f:
            palette.append('#d62728')  # Червоний: Сусіди
        elif 'cluster' in f:
            palette.append('#2ca02c')  # Зелений: Кластери
        elif 'burden' in f:
            palette.append('#1f77b4')  # Синій: Навантаження
        else:
            palette.append('#ff7f0e')  # Помаранчевий: Календар
            
    ax.barh(top_imp['feature_ua'][::-1], (top_imp['gain'] / 1000)[::-1], color=palette[::-1], edgecolor='black', alpha=0.85)
    ax.set_title("Важливість ознак у LightGBM (Gain × 1,000)\nПорівняння темпу, простору та календаря", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Total Gain (× 1,000)", fontsize=11)
    fig.tight_layout()
    fig.savefig("reports/figures/11_feature_importance_lightgbm.png")
    plt.close(fig)
    
    # Графік 3: Lift над НАЙКРАЩИМ бейзлайном
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ml_models = df_results[df_results['model'].str.contains('LogReg|LightGBM')].copy()
    x = np.arange(len(ml_models))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, ml_models['lift_f2_over_best_baseline_pct'], width, label=f'Lift F2 (%) над {best_base_name}', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, ml_models['lift_prauc_over_best_baseline_pct'], width, label=f'Lift PR-AUC (%) над {best_base_name}', color='#2ca02c', edgecolor='black')
    
    ax.set_title(f"Чесний приріст над найкращим бейзлайном ({best_base_name}) на Test", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Lift (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ml_models['model'], rotation=15, ha='right', fontsize=10)
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend(frameon=True, fontsize=10)
    
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:+.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:+.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    fig.tight_layout()
    fig.savefig("reports/figures/12_lift_over_baselines.png")
    plt.close(fig)
    
    # Графік 4: Reliability Diagram (Калібрувальна крива)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    prob_true, prob_pred = calibration_curve(y_test, probas['LightGBM (With Spatial)'], n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#d62728', label='LightGBM (With Spatial)')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ідеальне калібрування')
    ax.set_title("Reliability Diagram (Калібрування ймовірностей LightGBM)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Прогнозована ймовірність", fontsize=11)
    ax.set_ylabel("Фактична частка подій", fontsize=11)
    ax.set_xlim([0.0, 0.40])
    ax.set_ylim([0.0, 0.40])
    ax.legend(loc='upper left', frameon=True)
    fig.tight_layout()
    fig.savefig("reports/figures/13_calibration_reliability_diagram.png")
    plt.close(fig)


if __name__ == "__main__":
    run_test_evaluation()
