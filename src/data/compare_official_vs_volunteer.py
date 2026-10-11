import json
from pathlib import Path
import numpy as np
import pandas as pd


def to_epoch_seconds(series: pd.Series) -> np.ndarray:
    return (series - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=np.float64)


def compute_inter_alert_gaps_minutes(df: pd.DataFrame) -> pd.Series:
    """
    Обчислює проміжки (у хвилинах) між завершенням попередньої тривоги (finished_at)
    та початком наступної тривоги (started_at) у тій самій області.
    """
    s = df.sort_values(["oblast", "started_at", "finished_at"]).copy()
    prev_end = s.groupby("oblast")["finished_at"].shift(1)
    gaps_min = (s["started_at"] - prev_end).dt.total_seconds() / 60.0
    return gaps_min.dropna()


def count_after_gap_merge(df: pd.DataFrame, max_gap_minutes: float) -> int:
    """
    Підраховує, скільки інтервалів залишиться, якщо злити сусідні тривоги в одній області,
    між якими пауза <= max_gap_minutes.
    """
    s = df.sort_values(["oblast", "started_at", "finished_at"]).copy()
    prev_end = s.groupby("oblast")["finished_at"].shift(1)
    gap_min = (s["started_at"] - prev_end).dt.total_seconds() / 60.0
    is_new = prev_end.isna() | (gap_min > max_gap_minutes)
    return int(is_new.sum())


def match_onsets_within_tolerance(
    src_df: pd.DataFrame, ref_df: pd.DataFrame, tol_minutes: float = 5.0
) -> float:
    """
    Частка стартів у src_df, для яких у тій самій області в ref_df є старт у межах +-tol_minutes.
    """
    if len(src_df) == 0:
        return 0.0
    tol_sec = tol_minutes * 60.0
    matched = 0
    total = 0
    ref_grouped = {
        ob: np.sort(to_epoch_seconds(grp["started_at"]))
        for ob, grp in ref_df.groupby("oblast")
    }
    for ob, grp in src_df.groupby("oblast"):
        s_arr = np.sort(to_epoch_seconds(grp["started_at"]))
        total += len(s_arr)
        r_arr = ref_grouped.get(ob)
        if r_arr is None or len(r_arr) == 0:
            continue
        idx = np.searchsorted(r_arr, s_arr)
        idx_left = np.clip(idx - 1, 0, len(r_arr) - 1)
        idx_right = np.clip(idx, 0, len(r_arr) - 1)
        diff = np.minimum(np.abs(s_arr - r_arr[idx_left]), np.abs(s_arr - r_arr[idx_right]))
        matched += int(np.sum(diff <= tol_sec))
    return float(matched / total) if total > 0 else 0.0


def share_starts_inside_ongoing_ref_interval(
    src_df: pd.DataFrame, ref_df: pd.DataFrame, min_offset_minutes: float = 5.0
) -> float:
    """
    Частка стартів у src_df, які припадають всередину вже активного інтервалу ref_df
    (тобто ref_start + min_offset_minutes < src_start <= ref_end).
    Це вимірює, як часто один довгий інтервал у ref_df охоплює декілька коротших тривог у src_df.
    """
    if len(src_df) == 0:
        return 0.0
    offset_sec = min_offset_minutes * 60.0
    inside = 0
    total = 0
    for ob, grp in src_df.groupby("oblast"):
        s_arr = np.sort(to_epoch_seconds(grp["started_at"]))
        total += len(s_arr)
        ref_ob = ref_df[ref_df["oblast"] == ob].sort_values("started_at")
        if len(ref_ob) == 0:
            continue
        r_starts = to_epoch_seconds(ref_ob["started_at"])
        r_ends = to_epoch_seconds(ref_ob["finished_at"])
        # Для кожного s знаходимо останній інтервал ref, який почався до s - offset_sec
        idx = np.searchsorted(r_starts, s_arr - offset_sec, side="right") - 1
        valid = idx >= 0
        idx_clamped = np.clip(idx, 0, len(r_ends) - 1)
        is_in = valid & (s_arr <= r_ends[idx_clamped])
        inside += int(np.sum(is_in))
    return float(inside / total) if total > 0 else 0.0


def compare_official_vs_volunteer():
    project_root = Path(__file__).resolve().parents[2]
    off_path = project_root / "data" / "processed" / "alerts_clean_oblast.parquet"
    vol_path = project_root / "data" / "processed" / "volunteer_alerts_clean_as_is.parquet"
    raw_off_path = project_root / "data" / "raw" / "official_data_uk.csv"

    off_df = pd.read_parquet(off_path)
    vol_df = pd.read_parquet(vol_path)

    raw_off = pd.read_csv(raw_off_path).drop_duplicates()
    raw_off["started_at"] = pd.to_datetime(raw_off["started_at"], utc=True)
    raw_off["finished_at"] = pd.to_datetime(raw_off["finished_at"], utc=True)

    # Виключаємо Луганську область та Крим для чесного порівняння на 24 модельованих регіонах
    off_24 = off_df[~off_df["is_occupied_extreme"]].copy()
    vol_24 = vol_df[~vol_df["is_occupied_extreme"]].copy()
    raw_off_24 = raw_off[~raw_off["oblast"].isin(["Луганська область", "Автономна Республіка Крим", "м. Севастополь"])].copy()

    common_end = off_24["started_at"].max()

    periods = [
        (
            "Pre-2025 (2022-03-15 .. 2024-12-31)",
            pd.Timestamp("2022-03-15 00:00:00", tz="UTC"),
            pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
        ),
        (
            "Jan-Jul 2025 (2025-01-01 .. 2025-07-31)",
            pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2025-07-31 23:59:59", tz="UTC"),
        ),
        (
            "Aug-Nov 2025 (2025-08-01 .. 2025-11-30)",
            pd.Timestamp("2025-08-01 00:00:00", tz="UTC"),
            pd.Timestamp("2025-11-30 23:59:59", tz="UTC"),
        ),
        (
            "Post-Dec 2025 common (2025-12-01 .. 2026-09-07)",
            pd.Timestamp("2025-12-01 00:00:00", tz="UTC"),
            common_end,
        ),
        (
            "Full Common Window (2022-03-15 .. 2026-09-07)",
            pd.Timestamp("2022-03-15 00:00:00", tz="UTC"),
            common_end,
        ),
    ]

    rows = []
    for label, p_start, p_end in periods:
        o_sub = off_24[(off_24["started_at"] >= p_start) & (off_24["started_at"] <= p_end)].copy()
        v_sub = vol_24[(vol_24["started_at"] >= p_start) & (vol_24["started_at"] <= p_end)].copy()
        ro_sub = raw_off_24[(raw_off_24["started_at"] >= p_start) & (raw_off_24["started_at"] <= p_end)].copy()

        o_gaps = compute_inter_alert_gaps_minutes(o_sub)
        v_gaps = compute_inter_alert_gaps_minutes(v_sub)

        o_cnt = len(o_sub)
        v_cnt = len(v_sub)
        ratio = v_cnt / o_cnt if o_cnt > 0 else np.nan

        v_cnt_merge_15m = count_after_gap_merge(v_sub, 15.0)
        v_cnt_merge_30m = count_after_gap_merge(v_sub, 30.0)
        o_cnt_merge_15m = count_after_gap_merge(o_sub, 15.0)
        o_cnt_merge_30m = count_after_gap_merge(o_sub, 30.0)

        rows.append({
            "period": label,
            "official_alerts": o_cnt,
            "volunteer_alerts": v_cnt,
            "vol_to_off_ratio": round(float(ratio), 3),
            "official_median_dur_min": round(float(o_sub["duration_minutes"].median()), 2),
            "volunteer_median_dur_min": round(float(v_sub["duration_minutes"].median()), 2),
            "official_mean_dur_min": round(float(o_sub["duration_minutes"].mean()), 2),
            "volunteer_mean_dur_min": round(float(v_sub["duration_minutes"].mean()), 2),
            "official_total_hours": round(float(o_sub["duration_hours"].sum()), 1),
            "volunteer_total_hours": round(float(v_sub["duration_hours"].sum()), 1),
            "official_gap_le_15m_share": round(float((o_gaps <= 15.0).mean()), 4),
            "volunteer_gap_le_15m_share": round(float((v_gaps <= 15.0).mean()), 4),
            "official_gap_le_30m_share": round(float((o_gaps <= 30.0).mean()), 4),
            "volunteer_gap_le_30m_share": round(float((v_gaps <= 30.0).mean()), 4),
            "official_gap_le_60m_share": round(float((o_gaps <= 60.0).mean()), 4),
            "volunteer_gap_le_60m_share": round(float((v_gaps <= 60.0).mean()), 4),
            "volunteer_naive_share": round(float(v_sub["naive"].mean()), 4),
            "volunteer_alerts_if_merge_gaps_le_15m": v_cnt_merge_15m,
            "volunteer_alerts_if_merge_gaps_le_30m": v_cnt_merge_30m,
            "ratio_if_both_merge_gaps_le_30m": round(float(v_cnt_merge_30m / o_cnt_merge_30m), 3),
            "off_starts_matched_in_vol_pm5m": round(match_onsets_within_tolerance(o_sub, v_sub, 5.0), 4),
            "vol_starts_matched_in_off_pm5m": round(match_onsets_within_tolerance(v_sub, o_sub, 5.0), 4),
            "vol_starts_matched_in_raw_off_any_level_pm5m": round(match_onsets_within_tolerance(v_sub, ro_sub, 5.0), 4),
            "vol_starts_inside_ongoing_off_interval_share": round(
                share_starts_inside_ongoing_ref_interval(v_sub, o_sub, 5.0), 4
            ),
            "off_starts_inside_ongoing_vol_interval_share": round(
                share_starts_inside_ongoing_ref_interval(o_sub, v_sub, 5.0), 4
            ),
        })

    res_df = pd.DataFrame(rows)
    out_csv = project_root / "reports" / "comparison" / "official_vs_volunteer_comparison.csv"
    out_json = project_root / "reports" / "comparison" / "official_vs_volunteer_comparison.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 110)
    print("📊 ПОРІВНЯННЯ ДЖЕРЕЛ: OFFICIAL VS VOLUNTEER (24 МОДЕЛЬОВАНІ РЕГІОНИ)")
    print("=" * 110)
    print(res_df[[
        "period", "official_alerts", "volunteer_alerts", "vol_to_off_ratio",
        "official_median_dur_min", "volunteer_median_dur_min",
        "off_starts_matched_in_vol_pm5m", "vol_starts_matched_in_off_pm5m",
        "vol_starts_matched_in_raw_off_any_level_pm5m",
        "vol_starts_inside_ongoing_off_interval_share",
    ]].to_string(index=False))


if __name__ == "__main__":
    compare_official_vs_volunteer()
