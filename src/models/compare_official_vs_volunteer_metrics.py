"""Порівняльний аналіз метрик, зсуву ознак та емпіричних характеристик
між двома незалежними джерелами даних:
1. Official (official_data_uk.csv -> reports/official/)
2. Volunteer (volunteer_data_uk.csv -> reports/volunteer/)

Генерує зведені таблиці та 5 порівняльних графіків високої роздільної здатності
у директорії reports/comparison/ та reports/comparison/figures/:
- official_vs_volunteer_ml_metrics_comparison.csv
- official_vs_volunteer_feature_shift_comparison.csv
- figures/14_comparison_pr_auc_official_vs_volunteer.png
- figures/15_comparison_f2_official_vs_volunteer.png
- figures/16_comparison_feature_shift_official_vs_volunteer.png
- figures/17_comparison_alert_counts_and_durations.png
- figures/18_volunteer_naive_sensitivity_comparison.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_ci(ci_str: str, center: float) -> tuple[float, float]:
    """Повертає (err_low, err_high) із рядка виду '[0.324, 0.342]'."""
    if not isinstance(ci_str, str) or "[" not in ci_str:
        return 0.0, 0.0
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", ci_str)]
    if len(nums) != 2 or np.isnan(center):
        return 0.0, 0.0
    return max(0.0, center - nums[0]), max(0.0, nums[1] - center)


def build_ml_metrics_comparison(off_dir: Path, vol_dir: Path, comp_dir: Path) -> pd.DataFrame:
    """Об'єднує метрики всіх 11 моделей на 5 зрізах (Val, Full Test, Pre-Aug, Post-Aug, Recent 60d)."""
    slices = [
        ("Validation", "ml_validation_benchmark.csv", "volunteer_ml_validation_benchmark.csv"),
        ("Full Test", "ml_models_benchmark.csv", "volunteer_ml_models_benchmark.csv"),
        ("Test Pre-Aug 2025", "ml_test_pre_aug2025_benchmark.csv", "volunteer_ml_test_pre_aug2025_benchmark.csv"),
        ("Test Post-Aug 2025", "ml_test_post_aug2025_benchmark.csv", "volunteer_ml_test_post_aug2025_benchmark.csv"),
        ("Recent 60d", "ml_recent_60d_benchmark.csv", "volunteer_ml_recent_60d_benchmark.csv"),
    ]

    merged_frames = []
    for slice_name, off_file, vol_file in slices:
        df_off = pd.read_csv(off_dir / off_file)
        df_vol = pd.read_csv(vol_dir / vol_file)

        keep_cols = [
            "model", "feature_set", "frozen_threshold", "precision", "recall",
            "f2_score", "f2_ci_95", "pr_auc", "pr_auc_ci_95", "brier_score",
        ]
        df_off_sub = df_off[keep_cols].rename(columns={
            c: f"off_{c}" for c in keep_cols if c not in ("model", "feature_set")
        })
        df_vol_sub = df_vol[keep_cols].drop(columns=["feature_set"]).rename(columns={
            c: f"vol_{c}" for c in keep_cols if c not in ("model", "feature_set")
        })

        m = df_off_sub.merge(df_vol_sub, on="model", how="inner")
        m.insert(0, "slice", slice_name)
        m["delta_vol_minus_off_pr_auc"] = (m["vol_pr_auc"] - m["off_pr_auc"]).round(4)
        m["delta_vol_minus_off_f2"] = (m["vol_f2_score"] - m["off_f2_score"]).round(4)
        merged_frames.append(m)

    out_df = pd.concat(merged_frames, ignore_index=True)
    out_path = comp_dir / "official_vs_volunteer_ml_metrics_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"💾 Збережено зведену таблицю ML-метрик (Official vs Volunteer): {out_path}")
    return out_df


def build_feature_shift_comparison(off_dir: Path, vol_dir: Path, comp_dir: Path) -> pd.DataFrame:
    """Об'єднує таблиці зсуву ознак між Official та Volunteer."""
    off_sp = pd.read_csv(off_dir / "feature_shift_summary.csv")
    vol_sp = pd.read_csv(vol_dir / "volunteer_feature_shift_summary.csv")
    off_sub = pd.read_csv(off_dir / "feature_shift_test_subperiods.csv")
    vol_sub = pd.read_csv(vol_dir / "volunteer_feature_shift_test_subperiods.csv")

    off_full = off_sp.merge(off_sub, on="feature", how="inner")
    vol_full = vol_sp.merge(vol_sub, on="feature", how="inner")

    rename_off = {c: f"off_{c}" for c in off_full.columns if c != "feature"}
    rename_vol = {c: f"vol_{c}" for c in vol_full.columns if c != "feature"}

    comp = off_full.rename(columns=rename_off).merge(
        vol_full.rename(columns=rename_vol), on="feature", how="inner"
    )
    out_path = comp_dir / "official_vs_volunteer_feature_shift_comparison.csv"
    comp.to_csv(out_path, index=False)
    print(f"💾 Збережено зведену таблицю зсуву ознак (Official vs Volunteer): {out_path}")
    return comp


def plot_pr_auc_and_f2_comparisons(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    """Будує парні графіки 14 (PR-AUC) та 15 (F2-Score) із 95% ДІ для Official vs Volunteer."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    short_names = {
        "Baseline 1: True Naive Persistence": "B1: Naive Lag",
        "Baseline 2: Historical Lookup Table": "B2: Hist Lookup",
        "Baseline 3: Adaptive 14-day Rolling": "B3: Adaptive 14d",
        "LogReg (Set A: No Spatial)": "LogReg Set A",
        "LogReg (Set B: With Spatial)": "LogReg Set B",
        "LogReg (Set C: Spatial + Adaptive)": "LogReg Set C",
        "LogReg (Set C': Adaptive + Lags)": "LogReg Set C'",
        "LightGBM (Set A: No Spatial)": "LGBM Set A",
        "LightGBM (Set B: With Spatial)": "LGBM Set B",
        "LightGBM (Set C: Spatial + Adaptive)": "LGBM Set C (Val Win)",
        "LightGBM (Set C': Adaptive + Lags)": "LGBM Set C'",
    }

    # --- Figure 14: PR-AUC Comparison (Validation & Full Test) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.2), dpi=300)
    for ax, slice_name, panel_title in [
        (ax1, "Validation", "А. Validation (2024-05-01 — 2025-05-01)"),
        (ax2, "Full Test", "Б. Повний Test (2025-06-01 — 2026-10-11)"),
    ]:
        sub = metrics_df[
            (metrics_df["slice"] == slice_name)
            & (~metrics_df["model"].str.contains("Naive"))
        ].copy()
        labels = [short_names.get(m, m) for m in sub["model"]]
        x = np.arange(len(sub))
        w = 0.38

        off_vals = sub["off_pr_auc"].values
        vol_vals = sub["vol_pr_auc"].values
        off_errs = np.array([_parse_ci(ci, v) for ci, v in zip(sub["off_pr_auc_ci_95"], off_vals)]).T
        vol_errs = np.array([_parse_ci(ci, v) for ci, v in zip(sub["vol_pr_auc_ci_95"], vol_vals)]).T

        b1 = ax.bar(
            x - w / 2, off_vals, w, yerr=off_errs, capsize=3,
            label="Official (official_data_uk.csv)", color="#2b5c8f", edgecolor="black", alpha=0.88,
        )
        b2 = ax.bar(
            x + w / 2, vol_vals, w, yerr=vol_errs, capsize=3,
            label="Volunteer (volunteer_data_uk.csv)", color="#d95f02", edgecolor="black", alpha=0.88,
        )

        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("PR-AUC (з 95% Day-Block Bootstrap ДІ)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.5)
        ax.set_ylim(0.0, max(off_vals.max(), vol_vals.max()) + 0.08)
        ax.legend(loc="upper left", frameon=True, fontsize=10)

        for r in b1:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h + 0.012),
                        ha="center", va="bottom", fontsize=7.5, rotation=90, color="#1b3b5c", fontweight="bold")
        for r in b2:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h + 0.012),
                        ha="center", va="bottom", fontsize=7.5, rotation=90, color="#8c3800", fontweight="bold")

    fig.suptitle(
        "Порівняння PR-AUC між Official та Volunteer датасетами (Validation vs Повний Test, N=500 bootstrap)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig14_path = fig_dir / "14_comparison_pr_auc_official_vs_volunteer.png"
    fig.savefig(fig14_path)
    plt.close(fig)
    print(f"   Збережено: {fig14_path}")

    # --- Figure 15: F2-Score Comparison (Validation & Full Test) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.2), dpi=300)
    for ax, slice_name, panel_title in [
        (ax1, "Validation", "А. Validation (2024-05-01 — 2025-05-01)"),
        (ax2, "Full Test", "Б. Повний Test (2025-06-01 — 2026-10-11)"),
    ]:
        sub = metrics_df[metrics_df["slice"] == slice_name].copy()
        labels = [short_names.get(m, m) for m in sub["model"]]
        x = np.arange(len(sub))
        w = 0.38

        off_vals = sub["off_f2_score"].values
        vol_vals = sub["vol_f2_score"].values
        off_errs = np.array([_parse_ci(ci, v) for ci, v in zip(sub["off_f2_ci_95"], off_vals)]).T
        vol_errs = np.array([_parse_ci(ci, v) for ci, v in zip(sub["vol_f2_ci_95"], vol_vals)]).T

        b1 = ax.bar(
            x - w / 2, off_vals, w, yerr=off_errs, capsize=3,
            label="Official (official_data_uk.csv)", color="#2b5c8f", edgecolor="black", alpha=0.88,
        )
        b2 = ax.bar(
            x + w / 2, vol_vals, w, yerr=vol_errs, capsize=3,
            label="Volunteer (volunteer_data_uk.csv)", color="#d95f02", edgecolor="black", alpha=0.88,
        )

        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("$F_2$-Score при замороженому порозі $p^*$ (з 95% ДІ)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.5)
        ax.set_ylim(0.0, max(off_vals.max(), vol_vals.max()) + 0.09)
        ax.legend(loc="upper left", frameon=True, fontsize=10)

        for r in b1:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h + 0.012),
                        ha="center", va="bottom", fontsize=7.5, rotation=90, color="#1b3b5c", fontweight="bold")
        for r in b2:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h + 0.012),
                        ha="center", va="bottom", fontsize=7.5, rotation=90, color="#8c3800", fontweight="bold")

    fig.suptitle(
        "Порівняння F2-Score між Official та Volunteer датасетами (Validation vs Повний Test, N=500 bootstrap)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig15_path = fig_dir / "15_comparison_f2_official_vs_volunteer.png"
    fig.savefig(fig15_path)
    plt.close(fig)
    print(f"   Збережено: {fig15_path}")


def plot_feature_shift_comparison(shift_df: pd.DataFrame, metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    """Будує 4-панельний графік 16: порівняння зсуву ознак та його впливу на Set B vs Set C."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    stages = ["Train\n(2022–04.2024)", "Val\n(05.2024–05.2025)", "Test Pre-Aug\n(06–07.2025)", "Test Post-Aug\n(08.2025–10.2026)"]
    x = np.arange(len(stages))

    features_to_plot = [
        ("national_active_count", "А. Зсув national_active_count (активних областей о T)", "Середня к-сть активних областей", axes[0, 0]),
        ("hours_since_last_alert", "Б. Зсув hours_since_last_alert (годин спокою)", "Середня к-сть годин від відбою", axes[0, 1]),
        ("adaptive_rate_14d", "В. Еволюція adaptive_rate_14d (частота стартів за 14 діб)", "Середня ймовірність старту", axes[1, 0]),
    ]

    for feat, title, ylabel, ax in features_to_plot:
        row = shift_df[shift_df["feature"] == feat].iloc[0]
        off_y = [row["off_train_mean"], row["off_val_mean"], row["off_test_pre_2025_08_01_mean"], row["off_test_post_2025_08_01_mean"]]
        vol_y = [row["vol_train_mean"], row["vol_val_mean"], row["vol_test_pre_2025_08_01_mean"], row["vol_test_post_2025_08_01_mean"]]

        ax.plot(x, off_y, marker="o", linewidth=2.6, markersize=8, color="#2b5c8f",
                label=f"Official (Test/Train = {row['off_test_vs_train_ratio']:.2f}x, Post/Pre = {row['off_post_vs_pre_ratio']:.2f}x)")
        ax.plot(x, vol_y, marker="s", linewidth=2.6, markersize=8, color="#d95f02",
                label=f"Volunteer (Test/Train = {row['vol_test_vs_train_ratio']:.2f}x, Post/Pre = {row['vol_post_vs_pre_ratio']:.2f}x)")

        for xi, oy, vy in zip(x, off_y, vol_y):
            ax.annotate(f"{oy:.2f}", (xi, oy), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#1b3b5c")
            ax.annotate(f"{vy:.2f}", (xi, vy), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#8c3800")

        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=9.5)
        ax.set_ylim(0, max(max(off_y), max(vol_y)) * 1.25)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Panel D: How Set B vs Set C vs Set C' behave on Full Test (Official vs Volunteer)
    ax4 = axes[1, 1]
    full_test = metrics_df[metrics_df["slice"] == "Full Test"].set_index("model")
    models_focus = [
        ("Baseline 3: Adaptive 14-day Rolling", "Adaptive 14d"),
        ("LightGBM (Set B: With Spatial)", "LGBM Set B\n(No Adaptive/Nat)"),
        ("LightGBM (Set C: Spatial + Adaptive)", "LGBM Set C\n(Val Winner)"),
        ("LightGBM (Set C': Adaptive + Lags)", "LGBM Set C'\n(Post-hoc)"),
        ("LogReg (Set C: Spatial + Adaptive)", "LogReg Set C"),
    ]
    xf = np.arange(len(models_focus))
    w = 0.36
    off_pr = [full_test.loc[m, "off_pr_auc"] for m, _ in models_focus]
    vol_pr = [full_test.loc[m, "vol_pr_auc"] for m, _ in models_focus]

    r1 = ax4.bar(xf - w / 2, off_pr, w, label="Official Full Test PR-AUC", color="#2b5c8f", edgecolor="black")
    r2 = ax4.bar(xf + w / 2, vol_pr, w, label="Volunteer Full Test PR-AUC", color="#d95f02", edgecolor="black")
    for r in r1:
        h = r.get_height()
        ax4.annotate(f"{h:.4f}", (r.get_x() + r.get_width() / 2, h + 0.006), ha="center", fontsize=8.5, fontweight="bold", color="#1b3b5c")
    for r in r2:
        h = r.get_height()
        ax4.annotate(f"{h:.4f}", (r.get_x() + r.get_width() / 2, h + 0.006), ha="center", fontsize=8.5, fontweight="bold", color="#8c3800")

    ax4.set_title("Г. Наслідок зсуву: інверсія Set B vs Set C зникає на Volunteer", fontsize=12.5, fontweight="bold", pad=10)
    ax4.set_ylabel("Full Test PR-AUC", fontsize=10.5)
    ax4.set_xticks(xf)
    ax4.set_xticklabels([lbl for _, lbl in models_focus], fontsize=9.5)
    ax4.set_ylim(0.25, 0.45)
    ax4.legend(loc="upper left", frameon=True, fontsize=9.5)

    fig.suptitle(
        "Порівняння зсуву розподілу ознак (Covariate Shift) та стійкості Set C між Official і Volunteer",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig16_path = fig_dir / "16_comparison_feature_shift_official_vs_volunteer.png"
    fig.savefig(fig16_path)
    plt.close(fig)
    print(f"   Збережено: {fig16_path}")


def plot_empirical_alerts_comparison(comp_dir: Path, fig_dir: Path) -> None:
    """Будує графік 17: порівняння кількості тривог, медіанної тривалості та часток перетину."""
    df = pd.read_csv(comp_dir / "official_vs_volunteer_comparison.csv")
    sub = df[~df["period"].str.startswith("Full")].copy()

    short_periods = [
        "До 2025\n(03.2022–12.2024)",
        "Січ–Лип 2025\n(01.2025–07.2025)",
        "Сер–Лис 2025\n(08.2025–11.2025)",
        "Після Груд 2025\n(12.2025–09.2026)",
    ]
    x = np.arange(len(sub))
    w = 0.36

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.2), dpi=300)

    # Panel 1: Alert counts
    b1 = ax1.bar(x - w / 2, sub["official_alerts"], w, label="Official (після злиття до області)", color="#2b5c8f", edgecolor="black")
    b2 = ax1.bar(x + w / 2, sub["volunteer_alerts"], w, label="Volunteer (обласний рівень)", color="#d95f02", edgecolor="black")
    for i, (r_off, r_vol, ratio) in enumerate(zip(b1, b2, sub["vol_to_off_ratio"])):
        h = max(r_off.get_height(), r_vol.get_height())
        ax1.annotate(f"{ratio:.2f}x", (x[i], h + 900), ha="center", fontsize=9.5, fontweight="bold", color="#b30000")
    ax1.set_title("А. Кількість тривог та відношення Vol / Off", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Кількість тривог (24 області)", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_periods, fontsize=9)
    ax1.set_ylim(0, sub["volunteer_alerts"].max() * 1.18)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 2: Median duration
    d1 = ax2.bar(x - w / 2, sub["official_median_dur_min"], w, label="Official медіана (хв)", color="#2b5c8f", edgecolor="black")
    d2 = ax2.bar(x + w / 2, sub["volunteer_median_dur_min"], w, label="Volunteer медіана (хв)", color="#d95f02", edgecolor="black")
    for r in d1:
        ax2.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height() + 1.0), ha="center", fontsize=8.5, fontweight="bold")
    for r in d2:
        ax2.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height() + 1.0), ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_title("Б. Медіанна тривалість тривоги (хвилини)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Медіанна тривалість (хв)", fontsize=10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_periods, fontsize=9)
    ax2.set_ylim(0, 75)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 3: Hypothesis verification shares
    s_ongoing = sub["vol_starts_inside_ongoing_off_interval_share"] * 100
    s_obl_match = sub["vol_starts_matched_in_off_pm5m"] * 100
    s_raw_match = sub["vol_starts_matched_in_raw_off_any_level_pm5m"] * 100

    ax3.plot(x, s_ongoing, marker="o", linewidth=2.5, color="#d62728", label="Старти Vol всередині вже активної тривоги Off (%)")
    ax3.plot(x, s_raw_match, marker="s", linewidth=2.5, color="#2ca02c", label="Збіг старту Vol (±5 хв) із сирим районом/громадою Off (%)")
    ax3.plot(x, s_obl_match, marker="^", linewidth=2.2, linestyle="--", color="#2b5c8f", label="Збіг старту Vol (±5 хв) зі злитою областю Off (%)")
    for xi, v1, v2, v3 in zip(x, s_ongoing, s_raw_match, s_obl_match):
        ax3.annotate(f"{v1:.1f}%", (xi, v1 + 2.0), ha="center", fontsize=8.5, fontweight="bold", color="#d62728")
        ax3.annotate(f"{v2:.1f}%", (xi, v2 + 2.0), ha="center", fontsize=8.5, fontweight="bold", color="#1b7837")
        ax3.annotate(f"{v3:.1f}%", (xi, v3 - 4.5), ha="center", fontsize=8.5, fontweight="bold", color="#2b5c8f")

    ax3.set_title("В. Перевірка гіпотези: подрібнення vs злитий конверт", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Частка тривог Volunteer (%)", fontsize=10.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_periods, fontsize=9)
    ax3.set_ylim(15, 80)
    ax3.legend(loc="upper left", frameon=True, fontsize=8.2)

    fig.suptitle(
        "Емпіричне порівняння генеральної сукупності тривог: Official vs Volunteer (2022–2026)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig17_path = fig_dir / "17_comparison_alert_counts_and_durations.png"
    fig.savefig(fig17_path)
    plt.close(fig)
    print(f"   Збережено: {fig17_path}")


def plot_volunteer_naive_sensitivity(vol_dir: Path, fig_dir: Path) -> None:
    """Будує графік 18: чутливість метрик на Volunteer до обробки naive == True (варіанти а, б, в)."""
    df = pd.read_csv(vol_dir / "volunteer_naive_sensitivity.csv")

    variants_order = ["naive_as_is", "drop_naive", "naive_fixed_D"]
    labels = [
        "(а) naive_as_is\n(start + 30 хв)",
        "(б) drop_naive\n(без наївних)",
        "(в) naive_fixed_D\n(start + D_oblast з Train)",
    ]
    x = np.arange(len(variants_order))
    w = 0.35

    adapt = df[df["model"] == "Baseline 3: Adaptive 14-day Rolling"].set_index("naive_variant").loc[variants_order]
    lgb_c = df[df["model"] == "LightGBM (Set C: Spatial + Adaptive)"].set_index("naive_variant").loc[variants_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)

    # Panel 1: Test PR-AUC
    ad_pr = adapt["test_pr_auc"].values
    lg_pr = lgb_c["test_pr_auc"].values
    ad_err = np.array([_parse_ci(ci, v) for ci, v in zip(adapt["test_pr_auc_ci_95"], ad_pr)]).T
    lg_err = np.array([_parse_ci(ci, v) for ci, v in zip(lgb_c["test_pr_auc_ci_95"], lg_pr)]).T

    b1 = ax1.bar(x - w / 2, ad_pr, w, yerr=ad_err, capsize=4, label="Baseline 3: Adaptive 14-day", color="#2ca02c", edgecolor="black", alpha=0.85)
    b2 = ax1.bar(x + w / 2, lg_pr, w, yerr=lg_err, capsize=4, label="LightGBM (Set C: Val Winner)", color="#d62728", edgecolor="black", alpha=0.88)
    for r in b1:
        ax1.annotate(f"{r.get_height():.4f}", (r.get_x() + r.get_width() / 2, r.get_height() - 0.025), ha="center", color="white", fontweight="bold", fontsize=9.5)
    for r in b2:
        ax1.annotate(f"{r.get_height():.4f}", (r.get_x() + r.get_width() / 2, r.get_height() - 0.025), ha="center", color="white", fontweight="bold", fontsize=9.5)
    ax1.set_title("А. Чутливість Test PR-AUC до правил обробки naive", fontsize=12.5, fontweight="bold")
    ax1.set_ylabel("Test PR-AUC (з 95% ДІ)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0.32, 0.42)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Panel 2: Test F2-Score
    ad_f2 = adapt["test_f2"].values
    lg_f2 = lgb_c["test_f2"].values
    ad_f2_err = np.array([_parse_ci(ci, v) for ci, v in zip(adapt["test_f2_ci_95"], ad_f2)]).T
    lg_f2_err = np.array([_parse_ci(ci, v) for ci, v in zip(lgb_c["test_f2_ci_95"], lg_f2)]).T

    f1 = ax2.bar(x - w / 2, ad_f2, w, yerr=ad_f2_err, capsize=4, label="Baseline 3: Adaptive 14-day", color="#2ca02c", edgecolor="black", alpha=0.85)
    f2 = ax2.bar(x + w / 2, lg_f2, w, yerr=lg_f2_err, capsize=4, label="LightGBM (Set C: Val Winner)", color="#d62728", edgecolor="black", alpha=0.88)
    for r in f1:
        ax2.annotate(f"{r.get_height():.4f}", (r.get_x() + r.get_width() / 2, r.get_height() - 0.03), ha="center", color="white", fontweight="bold", fontsize=9.5)
    for r in f2:
        ax2.annotate(f"{r.get_height():.4f}", (r.get_x() + r.get_width() / 2, r.get_height() - 0.03), ha="center", color="white", fontweight="bold", fontsize=9.5)
    ax2.set_title("Б. Чутливість Test F2-Score до правил обробки naive", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("Test $F_2$-Score (з 95% ДІ)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0.50, 0.63)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    fig.suptitle(
        "Аналіз чутливості на Volunteer Dataset: (а) naive_as_is vs (б) drop_naive vs (в) каузальний naive_fixed_D",
        fontsize=13.5, fontweight="bold",
    )
    fig.tight_layout()
    fig18_path = fig_dir / "18_volunteer_naive_sensitivity_comparison.png"
    fig.savefig(fig18_path)
    plt.close(fig)
    print(f"   Збережено: {fig18_path}")


def run_all_comparisons() -> None:
    off_dir = PROJECT_ROOT / "reports" / "official"
    vol_dir = PROJECT_ROOT / "reports" / "volunteer"
    comp_dir = PROJECT_ROOT / "reports" / "comparison"
    fig_dir = comp_dir / "figures"
    comp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = build_ml_metrics_comparison(off_dir, vol_dir, comp_dir)
    shift_df = build_feature_shift_comparison(off_dir, vol_dir, comp_dir)

    plot_pr_auc_and_f2_comparisons(metrics_df, fig_dir)
    plot_feature_shift_comparison(shift_df, metrics_df, fig_dir)
    plot_empirical_alerts_comparison(comp_dir, fig_dir)
    plot_volunteer_naive_sensitivity(vol_dir, fig_dir)

    print("\n" + "=" * 115)
    print("📊 ПОРІВНЯННЯ КЛЮЧОВИХ МОДЕЛЕЙ НА FULL TEST: OFFICIAL vs VOLUNTEER")
    print("=" * 115)
    full_t = metrics_df[metrics_df["slice"] == "Full Test"][[
        "model", "off_pr_auc", "off_pr_auc_ci_95", "vol_pr_auc", "vol_pr_auc_ci_95",
        "delta_vol_minus_off_pr_auc", "off_f2_score", "vol_f2_score", "delta_vol_minus_off_f2"
    ]]
    print(full_t.to_string(index=False))


if __name__ == "__main__":
    run_all_comparisons()
