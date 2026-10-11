"""End-to-End Automated Pipeline & Artifact Verification (Quality Gate 5).

Verifies:
1. Parquet dataset integrity & row count (alerts_clean_oblast.parquet & ml_features_hourly.parquet).
2. Output integrity of all analytical and ML modules.
3. Existence and valid file size of all 13 publication figures.
4. Existence and non-emptiness of all summary CSV reports (including clusters_train_only.csv and ml_recent_60d_benchmark.csv).
5. Syntactic and structural validity of all 5 Jupyter Notebooks.
"""

from pathlib import Path
import json
import pandas as pd


def verify_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    ml_file = project_root / "data" / "processed" / "ml_features_hourly.parquet"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    notebooks_dir = project_root / "notebooks"

    print("=" * 65)
    print("🚀 ФАЗА 5: АВТОМАТИЗОВАНИЙ ТЕСТ ВІДТВОРЮВАНОСТІ ПРОЄКТУ")
    print("=" * 65)

    # 1. Dataset Verification
    print("\n📦 1. Перевірка цілісності очищеного датасету Parquet та Feature Store...")
    assert data_file.exists(), "FAIL: Parquet dataset is missing!"
    df = pd.read_parquet(data_file)
    assert len(df) == 84908, f"FAIL: Unexpected row count: {len(df)}"
    assert df["oblast"].nunique() == 25, "FAIL: Oblast count mismatch!"
    assert (df[df["is_finished"]]["duration_seconds"] < 0).sum() == 0, "FAIL: Negative duration found!"
    assert set(df["split_set"].unique()) == {"train", "val", "test"}, "FAIL: Split set mismatch!"
    print(f"   ✔ Датасет alerts_clean_oblast.parquet валідний: {len(df):,} рядків, 25 регіонів.")

    assert ml_file.exists(), "FAIL: ML Feature Store Parquet is missing!"
    ml_df = pd.read_parquet(ml_file)
    assert len(ml_df) == 797428, f"FAIL: Unexpected ML Feature Store row count: {len(ml_df)}"
    print(f"   ✔ Feature Store ml_features_hourly.parquet валідний: {len(ml_df):,} спокійних годин.")

    # 2. Figures Verification
    print("\n🎨 2. Перевірка 13 графіків у reports/figures/...")
    expected_figures = [
        "01_heatmap_24x7.png",
        "02_hourly_evolution_by_year.png",
        "03_duration_distribution.png",
        "04_clustering_metrics_evaluation.png",
        "05_regional_dendrogram.png",
        "06_spatial_cross_correlation_routes.png",
        "07_baselines_performance.png",
        "08_ml_pr_curves_validation.png",
        "09_threshold_tuning_f2.png",
        "10_test_pr_curves_comparison.png",
        "11_feature_importance_lightgbm.png",
        "12_lift_over_baselines.png",
        "13_calibration_reliability_diagram.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"FAIL: Missing figure {fig_name}!"
        size_kb = fig_path.stat().st_size / 1024
        assert size_kb > 20, f"FAIL: Figure {fig_name} is too small ({size_kb:.1f} KB)!"
        print(f"   ✔ {fig_name} ({size_kb:.1f} KB) — валідно.")

    # 3. CSV Reports Verification
    print("\n📊 3. Перевірка зведених CSV-звітів...")
    expected_csvs = [
        ("regional_clusters_summary.csv", 24),
        ("clusters_train_only.csv", 24),
        ("spatial_lead_lag_summary.csv", 49),
        ("baselines_benchmark.csv", 3),
        ("ml_validation_benchmark.csv", 7),
        ("ml_models_benchmark.csv", 7),
        ("ml_recent_60d_benchmark.csv", 7),
        ("feature_importance_lightgbm.csv", 13),
    ]
    for csv_name, min_rows in expected_csvs:
        csv_path = reports_dir / csv_name
        assert csv_path.exists(), f"FAIL: Missing CSV report {csv_name}!"
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) >= min_rows, f"FAIL: {csv_name} has only {len(csv_df)} rows, expected >= {min_rows}!"
        print(f"   ✔ {csv_name} ({len(csv_df)} рядків) — валідно.")

    # 4. Notebooks Structural Integrity
    print("\n📓 4. Перевірка структури 5 дослідницьких Jupyter Notebooks...")
    expected_notebooks = [
        "01_data_profiling_and_cleaning.ipynb",
        "02_rq1_temporal_cyclicity.ipynb",
        "03_rq2_regional_clustering.ipynb",
        "04_rq3_spatial_propagation_routes.ipynb",
        "05_ml_forecasting_experiment.ipynb",
    ]
    for nb_name in expected_notebooks:
        nb_path = notebooks_dir / nb_name
        assert nb_path.exists(), f"FAIL: Missing notebook {nb_name}!"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
        assert "cells" in nb_json and len(nb_json["cells"]) >= 4, f"FAIL: Notebook {nb_name} has invalid structure!"
        print(f"   ✔ {nb_name} ({len(nb_json['cells'])} клітинок) — синтаксично коректний.")

    print("\n" + "=" * 65)
    print("🎉 ВСІ ПЕРЕВІРКИ QUALITY GATE 5 УСПІШНО ПРОЙДЕНО!")
    print("=" * 65)


if __name__ == "__main__":
    verify_pipeline()
