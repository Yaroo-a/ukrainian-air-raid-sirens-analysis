"""Перевірка наявності артефактів, розмірностей таблиць та JSON-синтаксису блокнотів (Artifact & Syntax Check).

ВАЖЛИВО: Цей скрипт НЕ є перевіркою відтворюваності числових результатів з нуля (для повного відтворення
використовуйте `bash run_all.sh`). Він перевіряє лише:
1. Наявність та кількість рядків у Parquet-файлах (основного та волонтерського датасетів).
2. Наявність і розмір (> 20 KB) усіх 31 графіків:
   - 13 у reports/official/figures/
   - 13 у reports/volunteer/figures/
   - 5 порівняльних у reports/comparison/figures/
3. Наявність і непорожність усіх CSV/JSON-звітів у reports/official/, reports/volunteer/ та reports/comparison/.
4. Синтаксичну валідність JSON-структури всіх 5 Jupyter Notebooks (наявність масиву cells).
"""

from pathlib import Path
import json
import pandas as pd


def verify_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    ml_file = project_root / "data" / "processed" / "ml_features_hourly.parquet"
    vol_data_file = project_root / "data" / "processed" / "volunteer_alerts_clean_as_is.parquet"
    vol_ml_file = project_root / "data" / "processed" / "volunteer_ml_features_hourly_as_is.parquet"

    off_dir = project_root / "reports" / "official"
    vol_dir = project_root / "reports" / "volunteer"
    comp_dir = project_root / "reports" / "comparison"
    notebooks_dir = project_root / "notebooks"

    print("=" * 75)
    print("🔍 ПЕРЕВІРКА НАЯВНОСТІ АРТЕФАКТІВ, РОЗМІРНОСТЕЙ ТА JSON-СИНТАКСИСУ БЛОКНОТІВ")
    print("   (Примітка: перевірка структури файлів; повний перезапуск виконує bash run_all.sh)")
    print("=" * 75)

    # 1. Dataset Verification
    print("\n📦 1. Перевірка цілісності Parquet-файлів та Feature Store (Official & Volunteer)...")
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

    assert vol_data_file.exists(), "FAIL: Volunteer Parquet dataset is missing!"
    vdf = pd.read_parquet(vol_data_file)
    assert len(vdf) == 113525, f"FAIL: Unexpected volunteer row count: {len(vdf)}"
    print(f"   ✔ Датасет volunteer_alerts_clean_as_is.parquet валідний: {len(vdf):,} рядків.")

    assert vol_ml_file.exists(), "FAIL: Volunteer ML Feature Store Parquet is missing!"
    vml_df = pd.read_parquet(vol_ml_file)
    assert len(vml_df) > 700000, f"FAIL: Unexpected Volunteer ML Feature Store row count: {len(vml_df)}"
    print(f"   ✔ Feature Store volunteer_ml_features_hourly_as_is.parquet валідний: {len(vml_df):,} спокійних годин.")

    # 2. Figures Verification (Official, Volunteer, Comparison)
    print("\n🎨 2. Перевірка графіків у reports/{official,volunteer,comparison}/figures/...")
    expected_13_figures = [
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
    for sub_label, fig_folder in [("official", off_dir / "figures"), ("volunteer", vol_dir / "figures")]:
        for fig_name in expected_13_figures:
            fig_path = fig_folder / fig_name
            assert fig_path.exists(), f"FAIL: Missing figure {sub_label}/figures/{fig_name}!"
            size_kb = fig_path.stat().st_size / 1024
            assert size_kb > 20, f"FAIL: Figure {sub_label}/figures/{fig_name} is too small ({size_kb:.1f} KB)!"
        print(f"   ✔ reports/{sub_label}/figures/: усі 13 графіків (01..13) наявні та валідні.")

    expected_comp_figures = [
        "14_comparison_pr_auc_official_vs_volunteer.png",
        "15_comparison_f2_official_vs_volunteer.png",
        "16_comparison_feature_shift_official_vs_volunteer.png",
        "17_comparison_alert_counts_and_durations.png",
        "18_volunteer_naive_sensitivity_comparison.png",
        "19_walk_forward_vs_static_comparison.png",
    ]
    for fig_name in expected_comp_figures:
        fig_path = comp_dir / "figures" / fig_name
        assert fig_path.exists(), f"FAIL: Missing comparison figure {fig_name}!"
        size_kb = fig_path.stat().st_size / 1024
        assert size_kb > 20, f"FAIL: Comparison figure {fig_name} is too small ({size_kb:.1f} KB)!"
        print(f"   ✔ reports/comparison/figures/{fig_name} ({size_kb:.1f} KB) — валідно.")

    # 3. CSV Reports Verification
    print("\n📊 3. Перевірка структурованих CSV-звітів (official/, volunteer/, comparison/)...")
    expected_csvs = [
        # Official
        (off_dir / "regional_clusters_summary.csv", 24),
        (off_dir / "clusters_train_only.csv", 24),
        (off_dir / "spatial_lead_lag_summary.csv", 49),
        (off_dir / "feature_shift_summary.csv", 10),
        (off_dir / "feature_shift_test_subperiods.csv", 10),
        (off_dir / "baselines_benchmark.csv", 3),
        (off_dir / "ml_validation_benchmark.csv", 11),
        (off_dir / "ml_models_benchmark.csv", 11),
        (off_dir / "ml_test_pre_aug2025_benchmark.csv", 11),
        (off_dir / "ml_test_post_aug2025_benchmark.csv", 11),
        (off_dir / "ml_recent_60d_benchmark.csv", 11),
        (off_dir / "feature_importance_lightgbm.csv", 22),
        (off_dir / "walk_forward_monthly_metrics.csv", 16),
        (off_dir / "walk_forward_overall_benchmark.csv", 13),
        (off_dir / "walk_forward_recent_60d_benchmark.csv", 13),
        # Volunteer
        (vol_dir / "volunteer_train_median_durations.csv", 24),
        (vol_dir / "regional_clusters_summary.csv", 24),
        (vol_dir / "volunteer_clusters_train_only.csv", 24),
        (vol_dir / "spatial_lead_lag_summary.csv", 49),
        (vol_dir / "baselines_benchmark.csv", 3),
        (vol_dir / "volunteer_feature_shift_summary.csv", 10),
        (vol_dir / "volunteer_feature_shift_test_subperiods.csv", 10),
        (vol_dir / "volunteer_ml_validation_benchmark.csv", 11),
        (vol_dir / "volunteer_ml_models_benchmark.csv", 11),
        (vol_dir / "volunteer_ml_test_pre_aug2025_benchmark.csv", 11),
        (vol_dir / "volunteer_ml_test_post_aug2025_benchmark.csv", 11),
        (vol_dir / "volunteer_ml_recent_60d_benchmark.csv", 11),
        (vol_dir / "volunteer_feature_importance_lightgbm.csv", 22),
        (vol_dir / "volunteer_naive_sensitivity.csv", 6),
        (vol_dir / "volunteer_walk_forward_monthly_metrics.csv", 17),
        (vol_dir / "volunteer_walk_forward_overall_benchmark.csv", 13),
        (vol_dir / "volunteer_walk_forward_recent_60d_benchmark.csv", 13),
        # Comparison
        (comp_dir / "official_vs_volunteer_comparison.csv", 5),
        (comp_dir / "official_vs_volunteer_ml_metrics_comparison.csv", 55),
        (comp_dir / "official_vs_volunteer_feature_shift_comparison.csv", 10),
        (comp_dir / "official_vs_volunteer_walk_forward_comparison.csv", 26),
    ]
    for csv_path, min_rows in expected_csvs:
        rel = csv_path.relative_to(project_root)
        assert csv_path.exists(), f"FAIL: Missing CSV report {rel}!"
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) >= min_rows, f"FAIL: {rel} has only {len(csv_df)} rows, expected >= {min_rows}!"
        print(f"   ✔ {rel} ({len(csv_df)} рядків) — валідно.")

    # 4. Notebooks Structural Integrity (JSON Syntax only)
    print("\n📓 4. Перевірка JSON-синтаксису 5 дослідницьких Jupyter Notebooks...")
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
        print(f"   ✔ {nb_name} ({len(nb_json['cells'])} клітинок) — JSON-синтаксис коректний.")

    print("\n" + "=" * 75)
    print("🎉 ПЕРЕВІРКУ НАЯВНОСТІ АРТЕФАКТІВ ТА СИНТАКСИСУ БЛОКНОТІВ УСПІШНО ПРОЙДЕНО!")
    print("=" * 75)


if __name__ == "__main__":
    verify_pipeline()
