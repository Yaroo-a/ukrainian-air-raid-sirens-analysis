"""
Модуль фінальної оцінки на вибірці Test (червень 2025 - вересень 2026) (Фаза 2.4).
Реалізує:
1. Оцінку моделей на вибірці Test із зафіксованими на Validation порогами p*.
2. Розрахунок Lift over Baseline для F2-score та PR-AUC.
3. Аналіз важливості ознак (Feature Importance - Gain) для LightGBM.
4. Побудову порівняльних графіків PR-кривих, важливості ознак та Lift.
5. Збереження фінальної таблиці reports/ml_models_benchmark.csv.
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

FEATURE_SET_A = [
    'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'month',
    'is_active_t', 'burden_last_6h', 'burden_last_24h',
    'cluster_1_rear', 'cluster_2_center', 'cluster_3_front'
]

FEATURE_SET_B = FEATURE_SET_A + [
    'neighbors_active_count', 'neighbor_any_active', 'neighbor_active_ratio'
]


def load_test_data_and_models(parquet_path: str = "data/processed/ml_features_hourly.parquet"):
    """Завантажує тестовий набір даних, таблицю бейзлайну та натреновані моделі."""
    print("Завантаження даних та моделей...")
    df = pd.read_parquet(parquet_path)
    train_df = df[df['split_set'] == 'train']
    test_df = df[df['split_set'] == 'test'].copy()
    
    with open("models/optimal_thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    # Таблиця шансів Historical Lookup з Train
    lookup = train_df.groupby(['day_of_week', train_df['timestamp_utc'].dt.hour, 'cluster_id'])['target_onset_next_1h'].mean().reset_index()
    lookup.rename(columns={'timestamp_utc': 'hour', 'target_onset_next_1h': 'lookup_prob'}, inplace=True)
    global_mean = train_df['target_onset_next_1h'].mean()
    
    test_df['hour'] = test_df['timestamp_utc'].dt.hour
    test_merged = test_df.merge(lookup, on=['day_of_week', 'hour', 'cluster_id'], how='left')
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


def run_test_evaluation():
    """Проводить тестування всіх моделей і зберігає фінальні звіти."""
    os.makedirs("reports/figures", exist_ok=True)
    test_df, thresholds, models = load_test_data_and_models()
    
    y_test = test_df['target_onset_next_1h'].values
    print(f"Розмір Test: {len(test_df)}, позитивних: {y_test.sum()} ({y_test.mean():.4%})")
    
    # 1. Розрахунок прогнозів імовірностей
    probas = {}
    
    # Naive Persistence (0/1 прогноз)
    probas['Baseline 1: Naive Persistence'] = test_df['is_active_t'].values.astype(float)
    
    # Historical Lookup
    probas['Baseline 2: Historical Lookup'] = test_df['lookup_prob'].values
    
    # Logistic Regression
    probas['LogReg (No Spatial)'] = models['lr_a'].predict_proba(models['scaler_a'].transform(test_df[FEATURE_SET_A]))[:, 1]
    probas['LogReg (With Spatial)'] = models['lr_b'].predict_proba(models['scaler_b'].transform(test_df[FEATURE_SET_B]))[:, 1]
    
    # LightGBM
    probas['LightGBM (No Spatial)'] = models['lgb_a'].predict_proba(test_df[FEATURE_SET_A])[:, 1]
    probas['LightGBM (With Spatial)'] = models['lgb_b'].predict_proba(test_df[FEATURE_SET_B])[:, 1]
    
    # 2. Оцінка метрик із зафіксованими порогами
    results = []
    
    # Модельні конфігурації
    configs = [
        ('Baseline 1: Naive Persistence', 'None', 0.500),
        ('Baseline 2: Historical Lookup', 'None', thresholds['Historical Lookup']),
        ('LogReg (No Spatial)', 'A (Time & Lags)', thresholds['LogReg (No Spatial)']),
        ('LogReg (With Spatial)', 'B (Time, Lags & Neighbors)', thresholds['LogReg (With Spatial)']),
        ('LightGBM (No Spatial)', 'A (Time & Lags)', thresholds['LightGBM (No Spatial)']),
        ('LightGBM (With Spatial)', 'B (Time, Lags & Neighbors)', thresholds['LightGBM (With Spatial)']),
    ]
    
    for name, fset, th in configs:
        p_vals = probas[name]
        preds = (p_vals >= th).astype(int)
        
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
        pr_auc = average_precision_score(y_test, p_vals) if name != 'Baseline 1: Naive Persistence' else np.nan
        brier = brier_score_loss(y_test, p_vals)
        
        results.append({
            'model': name,
            'feature_set': fset,
            'frozen_threshold': th,
            'precision': prec,
            'recall': rec,
            'f2_score': f2,
            'pr_auc': pr_auc,
            'brier_score': brier
        })
        
    df_results = pd.DataFrame(results)
    
    # 3. Розрахунок Lift over Baseline 2 (Historical Lookup)
    base_f2 = df_results.loc[df_results['model'] == 'Baseline 2: Historical Lookup', 'f2_score'].iloc[0]
    base_prauc = df_results.loc[df_results['model'] == 'Baseline 2: Historical Lookup', 'pr_auc'].iloc[0]
    
    df_results['lift_f2_pct'] = ((df_results['f2_score'] - base_f2) / base_f2) * 100
    df_results['lift_prauc_pct'] = ((df_results['pr_auc'] - base_prauc) / base_prauc) * 100
    
    df_results.to_csv("reports/ml_models_benchmark.csv", index=False)
    print("\n" + "="*90)
    print("ФІНАЛЬНИЙ БЕНЧМАРК НА ТЕСТОВІЙ ВИБІРЦІ (TEST: 2025-06 - 2026-09):")
    print("="*90)
    print(df_results.to_string(index=False))
    
    # 4. Feature Importance для LightGBM
    lgb_spatial = models['lgb_b']
    importance_gain = lgb_spatial.booster_.feature_importance(importance_type='gain')
    importance_split = lgb_spatial.booster_.feature_importance(importance_type='split')
    
    feat_names_ua = {
        'burden_last_24h': 'Навантаження за 24 год (добовий імпульс)',
        'cluster_3_front': 'Прифронтовий кластер 3 (RQ2)',
        'hour_sin': 'Година доби (синус - неперервний час)',
        'month': 'Місяць (сезонність)',
        'hour_cos': 'Година доби (косинус)',
        'burden_last_6h': 'Навантаження за 6 год (тактичний темп)',
        'neighbors_active_count': 'Кількість активних сусідів (RQ3 лаги)',
        'day_of_week': 'День тижня (тижнева циклічність)',
        'is_active_t': 'Статус тривоги в поточний момент t',
        'neighbor_active_ratio': 'Частка активних сусідів (RQ3)',
        'neighbor_any_active': 'Чи активний бодай один сусід (RQ3)',
        'cluster_1_rear': 'Тиловий кластер 1 (RQ2)',
        'cluster_2_center': 'Центральний кластер 2 (RQ2)',
        'is_weekend': 'Вихідний день'
    }
    
    df_imp = pd.DataFrame({
        'feature': FEATURE_SET_B,
        'feature_ua': [feat_names_ua.get(f, f) for f in FEATURE_SET_B],
        'gain': importance_gain,
        'split': importance_split
    }).sort_values('gain', ascending=False)
    
    df_imp.to_csv("reports/feature_importance_lightgbm.csv", index=False)
    
    # 5. Побудова графіків
    plot_test_figures(y_test, probas, df_imp, df_results)
    print("Всі графіки та звіти збережено успішно.")


def plot_test_figures(y_test, probas, df_imp, df_results):
    """Створює графіки PR-кривих на Test, Feature Importance та Lift."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # --- Графік 1: Precision-Recall Curves на Test ---
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    colors = {
        'Baseline 2: Historical Lookup': '#7f7f7f',
        'LogReg (No Spatial)': '#1f77b4',
        'LogReg (With Spatial)': '#004c6d',
        'LightGBM (No Spatial)': '#ff7f0e',
        'LightGBM (With Spatial)': '#d62728'
    }
    
    base_rate = np.mean(y_test)
    ax.axhline(base_rate, color='darkgray', linestyle='--', linewidth=1.5,
               label=f'Випадковий базовий рівень (Base rate = {base_rate:.3%})')
    
    for name in ['Baseline 2: Historical Lookup', 'LogReg (No Spatial)', 'LogReg (With Spatial)',
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
        
    ax.set_title("Фінальне порівняння Precision-Recall кривих на вибірці TEST (2025–2026)\nЧесний іспит на майбутньому часовому горизонті", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (Повнота виявлення нових тривог)", fontsize=12)
    ax.set_ylabel("Precision (Точність сповіщень)", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 0.60])
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig("reports/figures/10_test_pr_curves_comparison.png")
    plt.close(fig)
    
    # --- Графік 2: Feature Importance (Gain) для LightGBM ---
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    
    top_imp = df_imp.copy()
    # Визначаємо категорії ознак кольорами
    palette = []
    for f in top_imp['feature']:
        if 'neighbor' in f:
            palette.append('#d62728')  # Червоний: Просторові сусіди
        elif 'cluster' in f:
            palette.append('#2ca02c')  # Зелений: Кластери
        elif 'burden' in f or f == 'is_active_t':
            palette.append('#1f77b4')  # Синій: Історичний темп / пам'ять
        else:
            palette.append('#ff7f0e')  # Помаранчевий: Час доби / календар
            
    bars = ax.barh(top_imp['feature_ua'][::-1], (top_imp['gain'] / 1000)[::-1], color=palette[::-1], edgecolor='black', alpha=0.85)
    
    ax.set_title("Важливість ознак у LightGBM (Feature Importance by Total Gain)\nЩо визначає загрозу: час, історія чи поведінка сусідів?", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Total Gain (× 1,000) — сумарний внесок у якість розщеплень", fontsize=12)
    
    # Легенда категорій
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Історичний темп / Пам\'ять атак'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Регіональний географічний кластер (RQ2)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Часова та календарна циклічність (RQ1)'),
        Patch(facecolor='#d62728', edgecolor='black', label='Просторові лаги сусідів (RQ3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig("reports/figures/11_feature_importance_lightgbm.png")
    plt.close(fig)
    
    # --- Графік 3: Lift over Baseline ---
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ml_models = df_results[df_results['model'].str.contains('LogReg|LightGBM')].copy()
    
    x = np.arange(len(ml_models))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, ml_models['lift_f2_pct'], width, label='Lift у F2-Score (%)', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, ml_models['lift_prauc_pct'], width, label='Lift у PR-AUC (%)', color='#2ca02c', edgecolor='black')
    
    ax.set_title("Відносний приріст якості моделей над бейзлайном (Lift over Baseline)\nТестовий період (Test: червень 2025 – вересень 2026)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Приріст над Historical Lookup (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ml_models['model'], rotation=15, ha='right', fontsize=10)
    ax.legend(frameon=True, fontsize=11)
    
    # Підписи стовпчиків
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"+{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"+{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax.set_ylim(0, max(ml_models['lift_prauc_pct']) * 1.15)
    fig.tight_layout()
    fig.savefig("reports/figures/12_lift_over_baselines.png")
    plt.close(fig)


if __name__ == "__main__":
    run_test_evaluation()
