#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    JUPYTER=".venv/bin/jupyter"
else
    PYTHON="python3"
    JUPYTER="jupyter"
fi

START_TS=$(date +%s)
echo "================================================================================"
echo "🚀 ЗАПУСК ПОВНОГО ВІДТВОРЮВАНОГО ПАЙПЛАЙНУ (run_all.sh)"
echo "   Python: $($PYTHON --version)"
echo "================================================================================"

echo -e "\n[1/9] Верифікація сирих джерел даних (verify_upstream_source.py)..."
"$PYTHON" src/data/verify_upstream_source.py

echo -e "\n[2/9] Очищення та агрегація інтервалів (cleaner.py: Official + Volunteer a/b/c)..."
"$PYTHON" src/data/cleaner.py

echo -e "\n[3/9] Порівняння офіційного та волонтерського джерел (compare_official_vs_volunteer.py)..."
"$PYTHON" src/data/compare_official_vs_volunteer.py

echo -e "\n[4/9] Дослідницькі модулі RQ1-RQ3 та Train-only кластери (cyclicity, clustering, propagation)..."
"$PYTHON" src/features/cyclicity.py
"$PYTHON" src/features/clustering.py
"$PYTHON" src/features/propagation.py

echo -e "\n[5/9] Генерація Feature Store та аналіз зсуву ознак (feature_store.py, analyze_feature_shift.py)..."
"$PYTHON" src/features/feature_store.py
"$PYTHON" src/features/analyze_feature_shift.py

echo -e "\n[6/9] Навчання бейзлайнів, моделей A/B/C/C' та оцінка з Day-Block Bootstrap N=500 (Official)..."
"$PYTHON" src/models/baselines.py
"$PYTHON" src/models/train_models.py
"$PYTHON" src/models/evaluate_test.py

echo -e "\n[7/9] Повний експеримент, аналіз чутливості на волонтерських даних та міжджерельне порівняння..."
"$PYTHON" src/models/run_volunteer_experiment.py
"$PYTHON" src/models/compare_official_vs_volunteer_metrics.py

echo -e "\n[8/9] Повне виконання всіх 5 Jupyter Notebooks з нуля (Restart & Run All)..."
"$JUPYTER" nbconvert --to notebook --execute --inplace \
    notebooks/01_data_profiling_and_cleaning.ipynb \
    notebooks/02_rq1_temporal_cyclicity.ipynb \
    notebooks/03_rq2_regional_clustering.ipynb \
    notebooks/04_rq3_spatial_propagation_routes.ipynb \
    notebooks/05_ml_forecasting_experiment.ipynb

echo -e "\n[9/9] Запуск тестів каузальної інваріантності та структури артефактів..."
"$PYTHON" tests/test_leakage_invariance.py
"$PYTHON" tests/verify_pipeline.py

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "================================================================================"
echo "✅ ПОВНИЙ ПАЙПЛАЙН УСПІШНО ВИКОНАНО ЗА ${ELAPSED} с (${MINS} хв ${SECS} с)"
echo "================================================================================"
