"""Регіональний описовий розріз (Exploratory Regional Breakdown) Walk-Forward проти Static та Adaptive
для всіх 24 областей окремо у двох джерелах даних (Official та Volunteer naive_fixed_D).

Методологічні правила протоколу:
1. Скрипт читає збережені Out-of-Fold прогнози з `data/processed/walk_forward_oof_predictions_*.parquet`
   та обов'язково звіряє їхній SHA-256 із записаним у `reports/*/walk_forward_details.json`.
2. Календарна доба `date_kyiv` обчислюється безпосередньо з `timestamp_utc`
   через `.dt.tz_convert("Europe/Kyiv").dt.date`.
3. Для кожної з 24 областей окремо в кожному джерелі на повному Test (2025-06 .. end) рахуються:
   - кількість спокійних годин (`n_calm_hours`), кількість стартів (`n_positives`), базова частота (`base_rate`),
   - PR-AUC у межах області для `Adaptive 14-day`, `Static LightGBM (Set C)`, `WF Rolling 12m LightGBM (Set C)`,
   - парні різниці ΔPR-AUC `(WF − Static)` та `(WF − Adaptive)` з Day-Block Bootstrap по добах `date_kyiv` (N=500, seed=42).
4. Усі 24 області залишаються в таблицях і на графіку без виключення; області з малою вибіркою у Test
   (`n_positives < 150` або `n_calm_hours < 1500`) явно позначаються прапорцем `low_sample_flag` та зірочкою `*`.
5. Групування виконується виключно за Train-only кластерами (`clusters_train_only.csv` / `volunteer_clusters_train_only.csv`),
   підписаними як «активність 2022–04.2024».
6. Увесь блок має статус `exploratory` (24 області × 2 джерела = множинні порівняння; жодних тверджень про
   статистичну значущість окремої області не робиться).
7. Зафіксовано застереження: абсолютний PR-AUC у межах області механічно залежить від її базової частоти (`base_rate`).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


CLUSTER_LABELS_UA = {
    "high_activity": "high_activity (активність 2022–04.2024)",
    "medium_activity": "medium_activity (активність 2022–04.2024)",
    "low_activity": "low_activity (активність 2022–04.2024)",
}

CLUSTER_ORDER = {
    "high_activity": 0,
    "medium_activity": 1,
    "low_activity": 2,
}

BASE_RATE_CAVEAT = (
    "Застереження: абсолютний PR-AUC у межах області механічно залежить від її базової частоти (base_rate); "
    "між областями коректно порівнювати лише парні внутрішньообласні різниці ΔPR-AUC (WF − Static) та (WF − Adaptive)."
)


def bootstrap_oblast_metrics(
    dates_kyiv: np.ndarray,
    y_true: np.ndarray,
    p_adapt: np.ndarray,
    p_stat: np.ndarray,
    p_wf: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> dict:
    """Day-Block Bootstrap (N=500, seed=42) у межах однієї області по календарних добах date_kyiv."""
    pt_adapt = float(average_precision_score(y_true, p_adapt))
    pt_stat = float(average_precision_score(y_true, p_stat))
    pt_wf = float(average_precision_score(y_true, p_wf))

    pt_wf_minus_stat = pt_wf - pt_stat
    pt_wf_minus_adapt = pt_wf - pt_adapt
    pt_stat_minus_adapt = pt_stat - pt_adapt

    unique_days, day_inv = np.unique(dates_kyiv, return_inverse=True)
    n_days = len(unique_days)
    day_buckets = [np.where(day_inv == i)[0] for i in range(n_days)]

    rng = np.random.default_rng(seed)
    b_adapt, b_stat, b_wf = [], [], []
    b_wf_stat, b_wf_adapt, b_stat_adapt = [], [], []

    for _ in range(n_bootstraps):
        sampled_days = rng.integers(0, n_days, size=n_days)
        idx = np.concatenate([day_buckets[d] for d in sampled_days])
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        pa = float(average_precision_score(yb, p_adapt[idx]))
        ps = float(average_precision_score(yb, p_stat[idx]))
        pw = float(average_precision_score(yb, p_wf[idx]))

        b_adapt.append(pa)
        b_stat.append(ps)
        b_wf.append(pw)
        b_wf_stat.append(pw - ps)
        b_wf_adapt.append(pw - pa)
        b_stat_adapt.append(ps - pa)

    def _ci_bounds(arr: list[float], pt: float) -> tuple[float, float, str]:
        if not arr:
            return pt, pt, f"[{pt:.3f}, {pt:.3f}]"
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        return lo, hi, f"[{lo:.3f}, {hi:.3f}]"

    def _ci_signed(arr: list[float], pt: float) -> tuple[float, float, str]:
        if not arr:
            return pt, pt, f"[{pt:+.4f}, {pt:+.4f}]"
        lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        return lo, hi, f"[{lo:+.4f}, {hi:+.4f}]"

    _, _, adapt_ci_str = _ci_bounds(b_adapt, pt_adapt)
    _, _, stat_ci_str = _ci_bounds(b_stat, pt_stat)
    _, _, wf_ci_str = _ci_bounds(b_wf, pt_wf)

    ws_lo, ws_hi, ws_ci_str = _ci_signed(b_wf_stat, pt_wf_minus_stat)
    wa_lo, wa_hi, wa_ci_str = _ci_signed(b_wf_adapt, pt_wf_minus_adapt)
    sa_lo, sa_hi, sa_ci_str = _ci_signed(b_stat_adapt, pt_stat_minus_adapt)

    return {
        "prauc_adapt": round(pt_adapt, 4),
        "prauc_adapt_ci_95": adapt_ci_str,
        "prauc_static_lgb_c": round(pt_stat, 4),
        "prauc_static_lgb_c_ci_95": stat_ci_str,
        "prauc_wf_roll12m_lgb_c": round(pt_wf, 4),
        "prauc_wf_roll12m_lgb_c_ci_95": wf_ci_str,
        "delta_prauc_wf_minus_static": round(pt_wf_minus_stat, 4),
        "delta_wf_minus_static_ci_low": round(ws_lo, 4),
        "delta_wf_minus_static_ci_high": round(ws_hi, 4),
        "delta_prauc_wf_minus_static_ci_95": ws_ci_str,
        "delta_prauc_wf_minus_adapt": round(pt_wf_minus_adapt, 4),
        "delta_wf_minus_adapt_ci_low": round(wa_lo, 4),
        "delta_wf_minus_adapt_ci_high": round(wa_hi, 4),
        "delta_prauc_wf_minus_adapt_ci_95": wa_ci_str,
        "delta_prauc_static_minus_adapt": round(pt_stat_minus_adapt, 4),
        "delta_prauc_static_minus_adapt_ci_95": sa_ci_str,
    }


def analyze_regional_for_source(
    dataset_name: str,
    oof_parquet_path: Path,
    details_json_path: Path,
    clusters_csv_path: Path,
    out_csv_path: Path,
    n_bootstraps: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Виконує регіональний аналіз для всіх 24 областей одного джерела з перевіркою SHA-256."""
    t0 = time.perf_counter()
    assert oof_parquet_path.exists(), f"Missing OOF parquet: {oof_parquet_path}"
    assert details_json_path.exists(), f"Missing details JSON: {details_json_path}"
    assert clusters_csv_path.exists(), f"Missing train-only clusters CSV: {clusters_csv_path}"

    actual_sha = hashlib.sha256(oof_parquet_path.read_bytes()).hexdigest()
    details = json.loads(details_json_path.read_text(encoding="utf-8"))
    expected_sha = details.get("oof_predictions_sha256")
    assert actual_sha == expected_sha, (
        f"SHA-256 mismatch for {oof_parquet_path.name}: actual={actual_sha}, expected={expected_sha}"
    )
    print(f"\n🔒 [{dataset_name}] SHA-256 OOF parquet верифіковано: {actual_sha[:16]}...")

    oof_df = pd.read_parquet(oof_parquet_path)
    required_cols = {"timestamp_utc", "oblast", "y", "p_adapt", "p_stat_lgb_c", "p_roll_lgb_c"}
    assert required_cols.issubset(oof_df.columns), f"Missing columns: {required_cols - set(oof_df.columns)}"

    # Умова (5): date_kyiv рахуємо безпосередньо з timestamp_utc
    oof_df["date_kyiv"] = (
        pd.to_datetime(oof_df["timestamp_utc"], utc=True)
        .dt.tz_convert("Europe/Kyiv")
        .dt.date
    )

    cl_df = pd.read_csv(clusters_csv_path)[["oblast", "cluster", "cluster_name"]].copy()
    assert len(cl_df) == 24, f"Expected 24 oblasts in {clusters_csv_path}, got {len(cl_df)}"
    cl_map = cl_df.set_index("oblast").to_dict(orient="index")

    oblasts = sorted(oof_df["oblast"].unique())
    assert len(oblasts) == 24, f"Expected 24 oblasts in OOF predictions, got {len(oblasts)}"

    rows = []
    for oblast in oblasts:
        sub = oof_df[oof_df["oblast"] == oblast]
        n_calm = int(len(sub))
        n_pos = int(sub["y"].sum())
        base_rate = float(sub["y"].mean())

        low_pos = n_pos < 150
        low_calm = n_calm < 1500
        low_sample_flag = bool(low_pos or low_calm)
        if low_pos and low_calm:
            sample_note = f"low_sample: n_positives={n_pos}<150 & n_calm_hours={n_calm}<1500"
        elif low_pos:
            sample_note = f"low_sample: n_positives={n_pos}<150"
        elif low_calm:
            sample_note = f"low_sample: n_calm_hours={n_calm}<1500"
        else:
            sample_note = "ok (n_positives>=150 & n_calm_hours>=1500)"

        c_info = cl_map[oblast]
        c_raw = str(c_info["cluster_name"])
        c_label = CLUSTER_LABELS_UA.get(c_raw, f"{c_raw} (активність 2022–04.2024)")

        boot_res = bootstrap_oblast_metrics(
            dates_kyiv=sub["date_kyiv"].values,
            y_true=sub["y"].values.astype(int),
            p_adapt=sub["p_adapt"].values,
            p_stat=sub["p_stat_lgb_c"].values,
            p_wf=sub["p_roll_lgb_c"].values,
            n_bootstraps=n_bootstraps,
            seed=42,
        )

        rows.append({
            "dataset": dataset_name,
            "oblast": oblast,
            "train_cluster_id": int(c_info["cluster"]),
            "train_cluster_name": c_raw,
            "train_cluster_label": c_label,
            "n_calm_hours": n_calm,
            "n_positives": n_pos,
            "base_rate": round(base_rate, 5),
            "low_sample_flag": low_sample_flag,
            "sample_note": sample_note,
            **boot_res,
            "protocol_status": "exploratory (regional descriptive breakdown, 24 oblasts x 2 sources)",
            "methodological_caveat": BASE_RATE_CAVEAT,
        })

    reg_df = pd.DataFrame(rows)
    reg_df["_ord"] = reg_df["train_cluster_name"].map(CLUSTER_ORDER)
    reg_df = reg_df.sort_values(["_ord", "base_rate", "oblast"], ascending=[True, False, True]).drop(columns=["_ord"]).reset_index(drop=True)
    reg_df.to_csv(out_csv_path, index=False)

    # Описовий підсумок за Train-only кластерами (активність 2022–04.2024)
    cluster_rows = []
    for c_raw in ["high_activity", "medium_activity", "low_activity"]:
        c_sub = reg_df[reg_df["train_cluster_name"] == c_raw]
        if len(c_sub) == 0:
            continue
        cluster_rows.append({
            "dataset": dataset_name,
            "train_cluster_label": CLUSTER_LABELS_UA[c_raw],
            "n_oblasts": int(len(c_sub)),
            "n_low_sample_oblasts": int(c_sub["low_sample_flag"].sum()),
            "mean_base_rate": round(float(c_sub["base_rate"].mean()), 4),
            "median_prauc_adapt": round(float(c_sub["prauc_adapt"].median()), 4),
            "median_prauc_static_lgb_c": round(float(c_sub["prauc_static_lgb_c"].median()), 4),
            "median_prauc_wf_roll12m_lgb_c": round(float(c_sub["prauc_wf_roll12m_lgb_c"].median()), 4),
            "mean_delta_wf_minus_static": round(float(c_sub["delta_prauc_wf_minus_static"].mean()), 4),
            "median_delta_wf_minus_static": round(float(c_sub["delta_prauc_wf_minus_static"].median()), 4),
            "mean_delta_wf_minus_adapt": round(float(c_sub["delta_prauc_wf_minus_adapt"].mean()), 4),
            "median_delta_wf_minus_adapt": round(float(c_sub["delta_prauc_wf_minus_adapt"].median()), 4),
            "n_wf_gt_static": int((c_sub["delta_prauc_wf_minus_static"] > 0).sum()),
            "n_wf_gt_adapt": int((c_sub["delta_prauc_wf_minus_adapt"] > 0).sum()),
        })
    cluster_df = pd.DataFrame(cluster_rows)

    elapsed = round(time.perf_counter() - t0, 2)
    meta = {
        "dataset": dataset_name,
        "elapsed_sec": elapsed,
        "n_oblasts": int(len(reg_df)),
        "n_low_sample_oblasts": int(reg_df["low_sample_flag"].sum()),
        "low_sample_oblasts": reg_df.loc[reg_df["low_sample_flag"], "oblast"].tolist(),
        "wf_gt_static_count": int((reg_df["delta_prauc_wf_minus_static"] > 0).sum()),
        "wf_gt_adapt_count": int((reg_df["delta_prauc_wf_minus_adapt"] > 0).sum()),
    }

    print(f"📊 [{dataset_name}] Регіональний розріз (24 області, час: {elapsed:.2f} с):")
    print(f"   WF > Static у {meta['wf_gt_static_count']}/24 областях | WF > Adaptive у {meta['wf_gt_adapt_count']}/24 областях")
    print(f"   Малі вибірки (позначені *, не виключені): {meta['low_sample_oblasts']}")
    print("\n   Зведення за Train-only кластерами (активність 2022–04.2024):")
    print(cluster_df.to_string(index=False))

    return reg_df, cluster_df, meta


def plot_regional_breakdown(off_reg: pd.DataFrame, vol_reg: pd.DataFrame, comp_dir: Path) -> Path:
    """Будує рисунок 20: описовий forest-plot ΔPR-AUC (WF − Static) та (WF − Adaptive) з 95% ДІ для всіх 24 областей."""
    fig_dir = comp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(1, 2, figsize=(20, 12.5), dpi=300, sharex=False)

    cluster_colors = {
        "high_activity": "#d62728",
        "medium_activity": "#ff7f0e",
        "low_activity": "#1f77b4",
    }

    panels = [
        (axes[0], vol_reg, "А. Volunteer (naive_fixed_D): ΔPR-AUC у межах області (Full Test 2025-06..2026-10)"),
        (axes[1], off_reg, "Б. Official: ΔPR-AUC у межах області (Full Test 2025-06..2026-09)"),
    ]

    for ax, df_src, title in panels:
        # Сортування: high -> medium -> low, всередині за базовою частотою
        df_plot = df_src.copy()
        df_plot["_ord"] = df_plot["train_cluster_name"].map(CLUSTER_ORDER)
        df_plot = df_plot.sort_values(["_ord", "base_rate", "oblast"], ascending=[True, False, True]).reset_index(drop=True)

        y_pos = np.arange(len(df_plot))[::-1]
        offset = 0.16

        for i, row in df_plot.iterrows():
            yi = y_pos[i]
            c_name = row["train_cluster_name"]
            col = cluster_colors.get(c_name, "#333333")
            is_low = bool(row["low_sample_flag"])
            m_face = "white" if is_low else col

            # 1) ΔPR-AUC (WF − Static) — кружечок + суцільна лінія
            ws_pt = row["delta_prauc_wf_minus_static"]
            ws_lo = row["delta_wf_minus_static_ci_low"]
            ws_hi = row["delta_wf_minus_static_ci_high"]
            ax.plot([ws_lo, ws_hi], [yi + offset, yi + offset], color=col, lw=2.0, alpha=0.9)
            ax.plot(
                ws_pt, yi + offset, marker="o", markersize=6.5,
                color=col, markerfacecolor=m_face, markeredgewidth=1.8, zorder=4
            )

            # 2) ΔPR-AUC (WF − Adaptive) — трикутник + пунктирна лінія
            wa_pt = row["delta_prauc_wf_minus_adapt"]
            wa_lo = row["delta_wf_minus_adapt_ci_low"]
            wa_hi = row["delta_wf_minus_adapt_ci_high"]
            ax.plot([wa_lo, wa_hi], [yi - offset, yi - offset], color="#2b5c8f", lw=1.6, ls="--", alpha=0.8)
            ax.plot(
                wa_pt, yi - offset, marker="^", markersize=6.0,
                color="#2b5c8f", markerfacecolor="white" if is_low else "#2b5c8f",
                markeredgewidth=1.6, zorder=4
            )

        # Горизонтальні розділювачі між Train-only кластерами
        prev_cluster = None
        for i, row in df_plot.iterrows():
            c_name = row["train_cluster_name"]
            if prev_cluster is not None and c_name != prev_cluster:
                sep_y = (y_pos[i - 1] + y_pos[i]) / 2.0
                ax.axhline(sep_y, color="#666666", ls=":", lw=1.2, alpha=0.7)
            prev_cluster = c_name

        ax.axvline(0.0, color="black", lw=1.5, ls="-", alpha=0.8)

        labels = []
        for _, row in df_plot.iterrows():
            star = " * [мала вибірка]" if row["low_sample_flag"] else ""
            short_cl = row["train_cluster_name"].replace("_activity", "")
            labels.append(
                f"{row['oblast']}{star} [{short_cl}, BR={row['base_rate']*100:.1f}%, n+={row['n_positives']}]"
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.8)
        ax.set_xlabel("Парна різниця ΔPR-AUC (95% Day-Block Bootstrap ДІ, N=500)", fontsize=10.5)
        ax.set_title(title, fontsize=11.8, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#d62728", marker="o", lw=2.0, label="Δ(WF − Static): high_activity (активність 2022–04.2024)"),
        Line2D([0], [0], color="#ff7f0e", marker="o", lw=2.0, label="Δ(WF − Static): medium_activity (активність 2022–04.2024)"),
        Line2D([0], [0], color="#1f77b4", marker="o", lw=2.0, label="Δ(WF − Static): low_activity (активність 2022–04.2024)"),
        Line2D([0], [0], color="#2b5c8f", marker="^", lw=1.6, ls="--", label="Δ(WF − Adaptive 14-day) у тій самій області"),
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="white", markeredgewidth=1.8, lw=0, label="* Порожній маркер: мала вибірка (n_pos<150 або n_calm<1500)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=True,
        fontsize=9.2,
    )

    fig.suptitle(
        "Рисунок 20. Регіональний описовий розріз Walk-Forward Rolling 12m проти Static та Adaptive (24 області, Exploratory)",
        fontsize=13.5, fontweight="bold", y=0.988,
    )
    fig.text(
        0.5, 0.008,
        "Методологічне застереження (Exploratory, 24 області × 2 джерела): без тверджень про стат. значущість окремих областей. "
        "Абсолютний PR-AUC залежить від базової частоти області (BR), тому порівнюються лише парні внутрішньообласні різниці ΔPR-AUC.",
        ha="center", fontsize=9.2, style="italic",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.905])
    fig_path = fig_dir / "20_walk_forward_regional_breakdown.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n🎨 Збережено графік регіонального розрізу: {fig_path.relative_to(PROJECT_ROOT)}")
    return fig_path


def main():
    t_start = time.perf_counter()
    off_dir = PROJECT_ROOT / "reports" / "official"
    vol_dir = PROJECT_ROOT / "reports" / "volunteer"
    comp_dir = PROJECT_ROOT / "reports" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    off_reg, off_cl, off_meta = analyze_regional_for_source(
        dataset_name="Official",
        oof_parquet_path=PROJECT_ROOT / "data" / "processed" / "walk_forward_oof_predictions_official.parquet",
        details_json_path=off_dir / "walk_forward_details.json",
        clusters_csv_path=off_dir / "clusters_train_only.csv",
        out_csv_path=off_dir / "walk_forward_regional_breakdown.csv",
        n_bootstraps=500,
    )

    vol_reg, vol_cl, vol_meta = analyze_regional_for_source(
        dataset_name="Volunteer (naive_fixed_D)",
        oof_parquet_path=PROJECT_ROOT / "data" / "processed" / "walk_forward_oof_predictions_volunteer_fixed_d.parquet",
        details_json_path=vol_dir / "volunteer_walk_forward_details.json",
        clusters_csv_path=vol_dir / "volunteer_clusters_train_only_fixed_d.csv",
        out_csv_path=vol_dir / "volunteer_walk_forward_regional_breakdown.csv",
        n_bootstraps=500,
    )

    # Зведена порівняльна таблиця по всіх 24 областях (Official vs Volunteer)
    keep_cols = [
        "oblast", "train_cluster_label", "n_calm_hours", "n_positives", "base_rate",
        "low_sample_flag", "sample_note",
        "prauc_adapt", "prauc_static_lgb_c", "prauc_wf_roll12m_lgb_c",
        "delta_prauc_wf_minus_static", "delta_prauc_wf_minus_static_ci_95",
        "delta_prauc_wf_minus_adapt", "delta_prauc_wf_minus_adapt_ci_95",
    ]
    comp_reg = off_reg[keep_cols].rename(
        columns={c: f"off_{c}" for c in keep_cols if c != "oblast"}
    ).merge(
        vol_reg[keep_cols].rename(
            columns={c: f"vol_{c}" for c in keep_cols if c != "oblast"}
        ),
        on="oblast",
        how="inner",
    )
    comp_csv = comp_dir / "official_vs_volunteer_walk_forward_regional.csv"
    comp_reg.to_csv(comp_csv, index=False)

    cluster_comp = pd.concat([off_cl, vol_cl], ignore_index=True)
    cluster_comp_csv = comp_dir / "official_vs_volunteer_walk_forward_regional_clusters.csv"
    cluster_comp.to_csv(cluster_comp_csv, index=False)

    plot_regional_breakdown(off_reg, vol_reg, comp_dir)

    total_sec = round(time.perf_counter() - t_start, 2)
    print(f"\n⏱️ Загальний виміряний час виконання регіонального розрізу (24 області × 2 джерела, N=500): {total_sec:.2f} с")


if __name__ == "__main__":
    main()
