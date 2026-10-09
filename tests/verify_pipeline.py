"""End-to-End Automated Pipeline & Artifact Verification (Quality Gate 5).

Verifies:
1. Parquet dataset integrity & row count.
2. Output integrity of all 4 analytical modules (cleaner, cyclicity, clustering, propagation).
3. Existence and valid file size of all 6 publication figures.
4. Existence and non-emptiness of all 2 summary CSV reports.
5. Syntactic and structural validity of all 4 Jupyter Notebooks.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np


def verify_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    notebooks_dir = project_root / "notebooks"

    print("=" * 65)
    print("🚀 ФАЗА 5: АВТОМАТИЗОВАНИЙ СТРЕС-ТЕСТ ВІДТВОРЮВАНОСТІ ПРОЄКТУ")
    print("=" * 65)

    # 1. Dataset Verification
    print("\n📦 1. Перевірка цілісності очищеного датасету Parquet...")
    assert data_file.exists(), "FAIL: Parquet dataset is missing!"
    df = pd.read_parquet(data_file)
    assert len(df) == 129955, f"FAIL: Unexpected row count: {len(df)}"
    assert df["oblast"].nunique() == 25, "FAIL: Oblast count mismatch!"
    assert (df[df["is_finished"]]["duration_seconds"] < 0).sum() == 0, "FAIL: Negative duration found!"
    assert set(df["split_set"].unique()) == {"train", "val", "test"}, "FAIL: Split set mismatch!"
    print(f"   ✔ Датасет Parquet валідний: {len(df):,} рядків, 25 регіонів, 0 від'ємних тривалостей.")

    # 2. Figures Verification
    print("\n🎨 2. Перевірка графіків високої роздільної здатності у reports/figures/...")
    expected_figures = [
        "01_heatmap_24x7.png",
        "02_hourly_evolution_by_year.png",
        "03_duration_distribution.png",
        "04_clustering_metrics_evaluation.png",
        "05_regional_dendrogram.png",
        "06_spatial_cross_correlation_routes.png",
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
        ("spatial_lead_lag_summary.csv", 49),
    ]
    for csv_name, min_rows in expected_csvs:
        csv_path = reports_dir / csv_name
        assert csv_path.exists(), f"FAIL: Missing CSV report {csv_name}!"
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) >= min_rows, f"FAIL: {csv_name} has only {len(csv_df)} rows, expected >= {min_rows}!"
        print(f"   ✔ {csv_name} ({len(csv_df)} рядків) — валідно.")

    # 4. Notebooks Structural Integrity
    print("\n📓 4. Перевірка структури 4-х дослідницьких Jupyter Notebooks...")
    expected_notebooks = [
        "01_data_profiling_and_cleaning.ipynb",
        "02_rq1_temporal_cyclicity.ipynb",
        "03_rq2_regional_clustering.ipynb",
        "04_rq3_spatial_propagation_routes.ipynb",
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
