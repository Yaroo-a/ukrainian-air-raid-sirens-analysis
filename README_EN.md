# 🚨 Spatio-Temporal Analysis & Short-Term Forecasting of Ukrainian Air Raid Sirens (2022–2026)

[ **🇺🇦 Українська версія** ](README.md) | [ **🇬🇧 English Version** ](README_EN.md)

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Datasets-Official%20299k%20%7C%20Volunteer%20116k-red.svg)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
[![Storage](https://img.shields.io/badge/Storage-Apache%20Parquet-green.svg)](https://parquet.apache.org/)
[![ML](https://img.shields.io/badge/Models-Static%20%26%20Walk--Forward%20ML-orange.svg)](reports/comparison/official_vs_volunteer_walk_forward_comparison.csv)

A research and engineering study analyzing the spatio-temporal dynamics of air raid sirens across the 24 regions (oblasts) of Ukraine (March 2022 – October 2026) and forecasting the onset of new alerts during calm hours over the $[T, T + 1\text{h})$ horizon. Every hypothesis is cross-verified in parallel across **two independent data sources** — the official alert logs (`official_data_uk.csv`) and the independent volunteer-curated corpus (`volunteer_data_uk.csv`).

---

## ⚡ Executive Summary (Key Research Findings)

1. **Stage 1 — Robust Spatio-Temporal Structure Across Two Independent Sources (RQ1–RQ3):**
   * **Temporal Cyclicity (RQ1):** Statistically significant weekly cyclicity is confirmed on both datasets (Kruskal-Wallis $H = 25.52, \ p = 2.74 \times 10^{-4}$ on Official; $H = 17.07, \ p = 0.0090$ on Volunteer, with 3 statistically significant day pairs each after Bonferroni correction $\alpha_{\text{adj}} = 0.00238$), peaking Wednesday–Friday and reaching a minimum on Sunday, alongside a stable share of nocturnal alert onsets (`00:00–05:59 Europe/Kyiv`: **$20.2\% – 24.0\%$** across years on Official and **$21.0\% – 26.4\%$** on Volunteer).
   * **Regional Clustering (RQ2):** Isolated clustering fitted strictly on `Train <= 2024-04-30` for ML features (`low_activity`, `medium_activity`, `high_activity`: `7 / 5 / 12` split on Official and `7 / 7 / 10` on Volunteer) shows strong cross-source consistency, while full-period descriptive clustering cleanly separates 3 geographic belts (Deep Rear, Transit Center & North, Frontline/Border Zone).
   * **Spatial Propagation Corridors (RQ3):** Cross-correlations on a 2-minute discrete grid using 10-minute alert onset impulse windows (`onset_window_min = 10`, after filtering nationwide alerts $\ge 15$ oblasts) across 49 contiguous oblast pairs exhibit high cross-source agreement in joint onset steps ($\rho_{\text{Spearman}} = 0.978, \ p = 1.40 \times 10^{-33}$), joint active intervals ($\rho_{\text{Spearman}} = 0.961, \ p = 6.10 \times 10^{-28}$), and peak onset cross-correlations ($\rho_{\text{Spearman}} = 0.635, \ p = 9.74 \times 10^{-7}$), capturing consistent 8–18 minute directional lead-lag couplings along southern, central, and northern corridors.
2. **Stage 2 (Block A) — Static ML Forecasting & Diagnosing Covariate Shift:**
   * Under the Val PR-AUC selection rule (formulated after initial Test inspection; $\arg\max \text{PR-AUC}$ on Validation `2024-05-01` – `2025-05-31`, where LightGBM `early_stopping(30)` is isolated on an internal chronological tail split `Train_ES` of the last 60 days of `Train` with a 2-hour `embargo`), **`LightGBM (Set C: Spatial + Adaptive, 22 features)`** was selected among the baseline candidates (`Baselines 1–3`, `LogReg A/B/C`, `LightGBM A/B/C`) on both Official (`Val PR-AUC = 0.2967`) and Volunteer (`0.3085` on `naive_as_is` / `0.3089` on `naive_fixed_D`).
   * On the full Official Test set (`2025-06` – `2026-09`), frozen static `LightGBM (Set C)` significantly outperforms the `Adaptive 14-day Baseline` ($\Delta\text{PR-AUC} = +0.0260 \ [+0.0182, +0.0333]$, $\Delta F_2 = +0.0257 \ [+0.0198, +0.0313]$), yet trails the simpler post-hoc `Set B` configuration (13 features without historical/national aggregates: `PR-AUC = 0.3610` vs. `0.3292`).
   * **Feature Distribution Shift (Consistent With Static `Set C` Degradation):** Between `Train` (`2022–04.2024`) and `Test` (`2025-06..2026`), on Official the mean number of simultaneously active oblasts (`national_active_count`) increased **$4.73\times$** (`1.17` $\to$ `5.54`, and after the August 1, 2025 sub-regional logging transition from `3.17` to `5.95`) and `neighbors_active_count` increased **$4.04\times$** (`0.13` $\to$ `0.52`), whereas on the independent Volunteer dataset (which remained at oblast-level logging) `national_active_count` increased **$1.72\times$** (`1.26` $\to$ `2.17`), `adaptive_rate_14d` increased **$1.66\times$** (`0.084` $\to$ `0.140`), and `onsets_last_24h` increased **$1.70\times$** (`1.95` $\to$ `3.32`). This degradation of static `LightGBM (Set C)` **is consistent with** the hypothesis that static decision trees frozen on 2022–2024 data extrapolate with outdated split thresholds (no separate interventional experiment was run to isolate a single cause).
3. **Stage 2 (Block B) — Causal `naive` Sensitivity on Volunteer & Monthly Walk-Forward Retraining:**
   * Evaluating three strictly causal policies for handling the `naive` flag in Volunteer records (`naive_as_is`, `drop_naive`, and `naive_fixed_D` using median duration $D_{\text{oblast}}$ estimated strictly on `Train`; see Section 3.3) showed the same sign and statistical significance for $F_2$-Score comparisons across all three variants: static `LightGBM (Set C)` outperforms `Adaptive 14-day` on $F_2$-Score ($\Delta F_2 = +0.0056 \ [+0.0029, +0.0085]$, $+0.0079 \ [+0.0052, +0.0108]$, and $+0.0067 \ [+0.0040, +0.0096]$, respectively), while slightly trailing the adaptive baseline on $\text{PR-AUC}$ ($\Delta\text{PR-AUC} = -0.0063 \ [-0.0103, -0.0021]$, $-0.0055 \ [-0.0096, -0.0012]$, and $-0.0057 \ [-0.0095, -0.0016]$).
   * Transitioning to **monthly retraining (`LightGBM Set C [WF Rolling 12m]`, 12m Train + 2m Val with a 2-hour `embargo` buffer, where `early_stopping(30)` is fitted on an internal 30-day `Train_ES` subsplit prior to `Val_M`)** yields a statistically significant boost in gradient boosting performance across both sources — large on Official ($\Delta\text{PR-AUC}_{\text{WF − Static}} = +0.0868 \ [+0.0776, +0.0951]$ on Full Test and $+0.1190 \ [+0.0898, +0.1466]$ on the latest 60 days) and moderate but statistically significant on Volunteer ($\Delta\text{PR-AUC}_{\text{WF − Static}} = +0.0131 \ [+0.0092, +0.0164]$ on Full Test and $+0.0334 \ [+0.0243, +0.0427]$ on the latest 60 days):
     * **Official:** Pooled `PR-AUC` rises from `0.3292` (Static) to **`0.4160 [0.405, 0.426]`** ($\Delta_{\text{vs Adapt}} = +0.1127 \ [+0.1035, +0.1216]$, $\Delta_{\text{WF − Static}} = +0.0868 \ [+0.0776, +0.0951]$, winning `16/16` months), and on the latest 60 days from `0.3830` to **`0.5024 [0.475, 0.531]`** ($\Delta_{\text{vs Adapt}} = +0.1339 \ [+0.1006, +0.1655]$).
     * **Volunteer (`naive_fixed_D`):** `LightGBM (Set C) [WF Rolling 12m]` statistically significantly outperforms `Adaptive 14-day` on both Full Test (`0.3997 [0.391, 0.407]` vs. `0.3924`, $\Delta_{\text{vs Adapt}} = +0.0074 \ [+0.0027, +0.0121]$, $\Delta_{\text{WF − Static}} = +0.0131 \ [+0.0092, +0.0164]$, winning `14/17` months vs. Adaptive and `13/17` vs. Static) and the latest 60 days (`0.4913 [0.478, 0.507]` vs. `0.4810`, $\Delta_{\text{vs Adapt}} = +0.0105 \ [+0.0016, +0.0199]$, $\Delta_{\text{WF − Static}} = +0.0334 \ [+0.0243, +0.0427]$).
     * **⚠️ Mandatory Caveat on Volunteer:** On the Volunteer dataset, `LightGBM (Set C) [WF Rolling 12m]` is **not unconditionally best across all metrics** — in `PR-AUC` ranking, it slightly trails the linear `LogReg (Set C) [Static Initial Train]` (`0.3997` vs. `0.4098` on Full Test, $\Delta = -0.0101 \ [-0.0137, -0.0064]$; `0.4913` vs. `0.5003` on the latest 60 days, $\Delta = -0.0089 \ [-0.0146, -0.0029]$), outperforming `LogReg Static` only on thresholded classification $F_2$-Score ($\Delta F_2 = +0.0047 \ [+0.0027, +0.0068]$ on Full Test and $+0.0051 \ [+0.0017, +0.0089]$ on the latest 60 days) and probability calibration (`Brier = 0.1095` vs. `0.2646`).

---

## 📑 Table of Contents
1. [Data Sources, Verification & Causal Cleaning](#-1-data-sources-verification--causal-cleaning)
2. [Stage 1: Exploratory Spatio-Temporal Analytics (RQ1–RQ3)](#-2-stage-1-exploratory-spatio-temporal-analytics-rq1rq3)
3. [Stage 2 (Block A): Static ML Forecasting, Covariate Shift & `naive` Sensitivity](#-3-stage-2-block-a-static-ml-forecasting-covariate-shift--naive-sensitivity)
4. [Stage 2 (Block B): Monthly Walk-Forward Retraining](#-4-stage-2-block-b-monthly-walk-forward-retraining)
5. [Regional Walk-Forward Breakdown Across 24 Oblasts (Exploratory)](#-5-regional-walk-forward-breakdown-across-24-oblasts-exploratory)
6. [Causal Invariance Verification, Limitations & Future Work](#-6-causal-invariance-verification-limitations--future-work)
7. [Quickstart Guide](#-7-quickstart-guide)
8. [Repository Structure](#-8-repository-structure)

---

## 📦 1. Data Sources, Verification & Causal Cleaning

Both raw datasets originate from the public repository [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (verification date: `2026-10-11`, [`reports/comparison/upstream_source_verification.json`](reports/comparison/upstream_source_verification.json)) and undergo cryptographic `SHA-256` verification, deduplication, and reconstruction of continuous oblast-level intervals `[started_at, finished_at]` in [`src/data/cleaner.py`](src/data/cleaner.py).

### Table 1.1. Comparative Data Passport & Cleaning Waterfall

| Parameter / Processing Stage | Source 1: Official (`official_data_uk.csv`) | Source 2: Volunteer (`volunteer_data_uk.csv`) |
| :--- | :--- | :--- |
| **Raw File & SHA-256** | [`data/raw/official_data_uk.csv`](data/raw/official_data_uk.csv)<br>`1f50085481d229f57b624d83b1ce723e884dea409c14afb5b7040c8a2f233e4f` | [`data/raw/volunteer_data_uk.csv`](data/raw/volunteer_data_uk.csv)<br>`1ead8618b1f5eb8eda2fe4a14e5fefa773a14534aba21aae577e463189da783b` |
| **Timestamp Span (UTC)** | `2022-03-15 16:10:34` – `2026-09-07 05:28:22` | Raw: `2022-02-25 16:36:22` – `2026-10-11 04:36:59`<br>From `2022-03-15`: `2022-03-15 00:05:01` – `2026-10-11 04:36:59` |
| **Raw Row Count** | `299,275` rows (contains a duplicated upstream block) | `115,989` rows (`5,812` or `5.01%` carrying `naive == True`) |
| **Step 1: Deduplication & Start Alignment** | `-113,845` exact duplicates $\to$ **`185,430`** unique rows | `0` exact duplicates; `-1,099` records prior to `2022-03-15 00:00 UTC` removed $\to$ **`114,890`** rows in [`cleaner.py`](src/data/cleaner.py) (`clean_volunteer_data()`) |
| **Step 2: Causal Micro-Glitch Filter ($< 60\text{ s}$ within same hour)** | `-450` records (`duration < 60s` within same hour window `dt.ceil('h')` or $<0$; `3` records crossing an hour boundary are causally retained) $\to$ **`184,980`** records | `-1,358` records (`duration < 60s` within same hour window `dt.ceil('h')`; `7` records crossing an hour boundary retained) $\to$ **`113,532`** records (`5,659` with `naive == True`) |
| **Step 3: Level / `naive` Policy** | Pre-`2025-08-01`: `level == 'oblast'` only (`-14,767` sub-regional rows) $\to$ **`170,213`** records; from `2025-08-01`: union of `oblast/raion/hromada` | 3 causal policies for `naive == True`:<br>• `naive_as_is` (+30 min)<br>• `drop_naive` (drop `-5,659` rows)<br>• `naive_fixed_D` (median $D_{\text{oblast}}$ strictly from `Train <= 2024-04-30`) |
| **Step 4: Overlapping Interval Merge** | Consolidates `-85,304` nested records $\to$ **`84,909`** intervals (`84,908` across 24 oblasts + `1` in Luhansk) | • `naive_as_is`: **`113,532`** intervals (`113,530` in 24 oblasts + `2` occupied)<br>• `drop_naive`: **`107,873`** intervals (`107,873` in 24 oblasts)<br>• `naive_fixed_D`: **`113,532`** intervals (`113,530` in 24 oblasts + `2` occupied) |
| **Step 5: Hourly Feature Store (`is_active_T == 0`)** | [`data/processed/ml_features_hourly.parquet`](data/processed/ml_features_hourly.parquet):<br>**`797,427`** calm region-hours (`0` duplicates) | • `as_is`: **`871,498`** calm region-hours<br>• `drop_naive`: **`874,368`** calm region-hours<br>• `fixed_D`: **`870,931`** calm region-hours |

The full calendar grid of the Official Test window (`2025-06-01 00:00` – `2026-09-07 04:00 UTC`, `11,117` hours $\times$ `24` oblasts) comprises `266,808` region-hours; excluding `69,725` active-alert hours (`is_active_T == 1`) leaves exactly **`197,083` calm region-hours** (`266,808 − 69,725 = 197,083`).

<details>
<summary><b>🔍 Detailed Audit of the 3 Upstream Duplicate Blocks in Vadimkin (Official CSV) & Version History</b></summary>

Direct inspection of `https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv` ([`src/data/verify_upstream_source.py`](src/data/verify_upstream_source.py), report [`reports/comparison/upstream_source_verification.json`](reports/comparison/upstream_source_verification.json)) confirmed that the upstream file contains `299,275` rows (with an identical SHA-256 hash) and that the duplicated block is present directly in upstream:
* **Block 1 (rows `0..113,910`, `113,911` rows):** Spans `2022-03-15 16:10:34 UTC` – `2026-01-25 13:11:52 UTC` (`0` internal duplicates).
* **Block 2 (rows `113,911..227,831`, `113,921` rows):** Re-spans the exact same interval `2022-03-15 16:10:34 UTC` – `2026-01-25 13:11:52 UTC` and contains `113,845` exact row-for-row duplicates of Block 1 across all 6 columns (`oblast, raion, hromada, level, started_at, finished_at`) plus `76` non-overlapping rows (`0` internal duplicates).
* **Block 3 (rows `227,832..299,274`, `71,443` rows):** Spans `2026-01-23 12:42:58 UTC` – `2026-09-07 05:28:22 UTC` (`0` internal duplicates and `0` overlap with Blocks 1–2).
* **Why Metrics Changed After Fixing `cleaner.py`:** Prior to deduplicating rows `0..227,831` and aggregating post-August 2025 `raion`/`hromada` alerts, absolute alert frequencies in RQ1–RQ2 were artificially doubled while the Test base rate was depressed by the logging transition from `oblast` to `raion`. Removing the `113,845` duplicates and reconstructing oblast intervals from `raion`/`hromada` restored the true underlying process (Test base rate **$10.27\%$**, latest 60 days **$12.65\%$**).
</details>

---

## 🔬 2. Stage 1: Exploratory Spatio-Temporal Analytics (RQ1–RQ3)

All three exploratory research questions were evaluated in parallel on Official (`84,908` intervals across 24 oblasts) and Volunteer (`113,530` intervals across 24 oblasts in `naive_as_is`; `108,583` over the shared time window up to `2026-09-07`).

### 2.1. Temporal Cyclicity & Alert Durations (RQ1)
* **Weekly Rhythm (Kruskal-Wallis Test, [`src/features/cyclicity.py`](src/features/cyclicity.py)):**
  * **Official:** $H = 25.52, \ p = 2.74 \times 10^{-4}$ (3 statistically significant day pairs after Bonferroni correction $\alpha_{\text{adj}} = 0.00238$). Peak activity occurs Wednesday–Friday (median 52–53 merged oblast alerts/day, mean 53.4–55.0), reaching a minimum on Sunday (median 47, mean 46.8 alerts/day).
  * **Volunteer (`naive_as_is`):** $H = 17.07, \ p = 0.0090$ (also 3 statistically significant day pairs after Bonferroni correction), reproducing the mid-week peak and Sunday trough.
* **Diurnal Profile & Nocturnal Alerts (`00:00–05:59 Europe/Kyiv`):** After aggregating district-level alerts in 2025–2026, the nocturnal onset share remains stable between **$20.2\%$ and $24.0\%$** across years on Official (2022: $24.0\%$, 2023: $21.5\%$, 2024: $20.2\%$, 2025: $20.5\%$, 2026: $22.9\%$) and **$21.0\% – 26.4\%$** on Volunteer (2022: $23.8\%$, 2023: $21.0\%$, 2024: $21.4\%$, 2025: $26.4\%$, 2026: $22.5\%$).
* **Cross-Source Duration Comparison ([`reports/comparison/official_vs_volunteer_comparison.csv`](reports/comparison/official_vs_volunteer_comparison.csv)):** Over the shared window `2022-03-15 .. 2026-09-07`, median duration on Official is **43.15 min** (mean **102.40 min**) vs. **30.68 min** (mean **49.06 min**) on Volunteer, because merging all district intervals within an oblast after August 2025 on Official creates longer continuous oblast envelopes containing multiple shorter Volunteer waves ($64.45\%$ of post-December 2025 Volunteer starts occur inside an ongoing Official interval).

<p align="center">
  <img src="reports/official/figures/01_heatmap_24x7.png" alt="24x7 Alert Activity Heatmap" width="85%">
  <br><em>Figure 1. Heatmap of alert onset counts by hour of day (Europe/Kyiv) and day of week (Official, 2022–2026).</em>
</p>

<p align="center">
  <img src="reports/comparison/figures/17_comparison_alert_counts_and_durations.png" alt="Comparison of Alert Counts and Durations Official vs Volunteer" width="90%">
  <br><em>Figure 2. Cross-source comparison of alert counts and durations between Official and Volunteer across historical sub-periods (Figure 17 in reports/comparison/figures/).</em>
</p>

### 2.2. Regional Clustering: Descriptive (RQ2) vs. ML (Train-Only)
The repository strictly separates **descriptive full-period clustering** (for RQ2) from **isolated clustering fitted strictly on `Train <= 2024-04-30`** (fed into ML models with zero future leakage):

1. **Descriptive RQ2 Clustering ([`reports/official/regional_clusters_summary.csv`](reports/official/regional_clusters_summary.csv), full period 2022–2026, Ward linkage, Silhouette = 0.4417 at $K=3$):**
   * **Cluster 1 — Deep Rear / West (12 regions on Official / 10 on Volunteer):** Ivano-Frankivsk, Volyn, Vinnytsia, Zhytomyr, Zakarpattia, Kyiv Oblast, Lviv, Rivne, Ternopil, Khmelnytskyi, Chernivtsi, Kyiv City (avg 25.8 alerts/month, median duration 46.4 min on Official).
   * **Cluster 2 — Transit Center / North (7 regions on Official / 8 on Volunteer):** Dnipropetrovsk, Kirovohrad, Mykolaiv, Odesa, Poltava, Kherson, Cherkasy (avg 90.3 alerts/month, median duration 36.0 min on Official).
   * **Cluster 3 — Frontline / Border Zone (5 regions on Official / 6 on Volunteer):** Donetsk, Zaporizhzhia, Sumy, Kharkiv, Chernihiv (avg 128.3 alerts/month, median duration 57.3 min, share of alerts $\ge 2$h: 28.7% on Official).
2. **ML Train-Only Clusters ([`reports/official/clusters_train_only.csv`](reports/official/clusters_train_only.csv), fitted strictly on `Train <= 2024-04-30 23:59:59 UTC` via `generate_train_only_clusters()` in [`src/features/clustering.py`](src/features/clustering.py)):**
   * `cluster_1_low_activity` (`low_activity (активність 2022–04.2024)`, 7 western oblasts): Ivano-Frankivsk, Volyn, Zakarpattia, Lviv, Rivne, Ternopil, Chernivtsi (17.4–22.1 alerts/month).
   * `cluster_2_medium_activity` (`medium_activity (активність 2022–04.2024)`, 5 central/north-western regions on Official): Vinnytsia, Zhytomyr, Kyiv Oblast, Khmelnytskyi, Kyiv City (24.2–38.2 alerts/month). *(Note: on Volunteer in [`volunteer_clusters_train_only.csv`](reports/volunteer/volunteer_clusters_train_only.csv) and [`volunteer_clusters_train_only_fixed_d.csv`](reports/volunteer/volunteer_clusters_train_only_fixed_d.csv), Odesa and Cherkasy also join `medium_activity` on Train, yielding a `7 low / 7 medium / 10 high` split).*
   * `cluster_3_high_activity` (`high_activity (активність 2022–04.2024)`, 12 eastern, southern, and central oblasts on Official / 10 on Volunteer): Dnipropetrovsk, Donetsk, Zaporizhzhia, Kirovohrad, Mykolaiv, Odesa, Poltava, Sumy, Kharkiv, Kherson, Cherkasy, Chernihiv (44.0–168.8 alerts/month).

<p align="center">
  <img src="reports/official/figures/05_regional_dendrogram.png" alt="Regional Dendrogram" width="85%">
  <br><em>Figure 3. Ward hierarchical dendrogram of the 24 regions of Ukraine (RQ2, Official).</em>
</p>

### 2.3. Spatial Cross-Correlations Between Neighbors (RQ3)
* To avoid smearing short propagation delays across multi-hour alert durations, [`src/features/propagation.py`](src/features/propagation.py) filters out synchronized nationwide alerts ($\ge 15$ oblasts simultaneously) on a **2-minute discrete grid (`time_step_min = 2`)** using **10-minute alert onset impulse windows (`onset_window_min = 10`, 5 steps)** and computes cross-correlations over lags $\tau \in [-180, +180]$ min across all 49 contiguous region pairs ([`reports/official/spatial_lead_lag_summary.csv`](reports/official/spatial_lead_lag_summary.csv), [`reports/volunteer/spatial_lead_lag_summary.csv`](reports/volunteer/spatial_lead_lag_summary.csv)). Directional lead (`dominant_pair`) and delay (`lead_time_min = abs(peak_nonzero_lag_min)`) are determined by the shoulder asymmetry (`shoulder_diff = pos_shoulder - neg_shoulder` over $\pm 2..30$ min) in 100% sign consistency with `peak_nonzero_lag_min`.
* Spearman rank correlation across the 49 neighbor pairs between Official and Volunteer is **$\rho = 0.978$** ($p = 1.40 \times 10^{-33}$) for joint onset steps (`joint_onset_steps`), **$\rho = 0.961$** ($p = 6.10 \times 10^{-28}$) for joint active intervals (`joint_active_intervals`), and **$\rho = 0.635$** ($p = 9.74 \times 10^{-7}$) for peak onset cross-correlations (`max_correlation`). On Official, strongest couplings occur along southern, central, and eastern corridors: Kherson $\to$ Mykolaiv ($r = 0.494$, `lead_time_min = 8` min, `shoulder_diff = +0.0046`), Kirovohrad $\to$ Cherkasy ($r = 0.468$, `lead_time_min = 12` min, `shoulder_diff = -0.0038`), Dnipropetrovsk $\leftrightarrow$ Zaporizhzhia ($r = 0.424$, synchronous $< 2$ min, `lead_time_min = 0` min), Kyiv Oblast $\to$ Kyiv City ($r = 0.331$, `lead_time_min = 18` min, `shoulder_diff = +0.0078`), Dnipropetrovsk $\to$ Poltava ($r = 0.328$, `lead_time_min = 8` min, `shoulder_diff = -0.0036`).

<p align="center">
  <img src="reports/official/figures/06_spatial_cross_correlation_routes.png" alt="Spatial Cross-Correlation Routes" width="85%">
  <br><em>Figure 4. Top-15 adjacent region pairs by alert onset cross-correlation strength (2-min grid, Official).</em>
</p>

---

## 🤖 3. Stage 2 (Block A): Static ML Forecasting, Covariate Shift & `naive` Sensitivity

### 3.1. Problem Formulation, Feature Set Hierarchy & Val PR-AUC Selection Rule (Formulated After Initial Test Inspection)
* **Prediction Unit $(T, \text{oblast})$:** Regular hourly grid in UTC (`timestamp_utc`), conditioned on calm hours (`is_active_T == 0`).
* **Target Variable (`target_onset_next_1h`):** $y_T = 1$ if a new alert starts in the region during the half-open interval **$[T, T + 1\text{h})$**.
* **Feature Set Hierarchy (all computed strictly from data prior to $T$, `closed="left"`, on the full hourly grid before calm filtering):**
  * **Set A (Time, Burden & Train Cluster — 10 features):** `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `month` (`Europe/Kyiv`), `burden_last_6h`, `burden_last_24h` (fraction of active hours in $[T-6\text{h}, T)$ and $[T-24\text{h}, T)$), and Train-only cluster indicators (`cluster_1_low_activity`, `cluster_2_medium_activity`, `cluster_3_high_activity`).
  * **Set B (A + Spatial at $T$ — 13 features):** Set A + instantaneous status of contiguous neighbors at exact moment $T$: `neighbors_active_count`, `neighbor_any_active`, `neighbor_active_ratio`.
  * **Set C (B + Adaptive & Spatio-Temporal Memory — 22 features, Primary Configuration Selected on Val by PR-AUC Under Rule Formulated After Initial Test Inspection):** Set B + 9 historical adaptive and spatio-temporal features: `adaptive_rate_7d/14d/30d`, `national_active_count`, `onsets_last_24h`, `hours_since_last_alert` (`clip(upper=720.0)`), `neighbors_active_lag_1h/2h/3h`.
  * **Set C' (`FEATURE_SET_C_PRIME`: B + Adaptive & Neighbor Lags Only — 19 features, Post-Hoc Experiment):** Set C excluding the three most strongly shifted features (`national_active_count`, `hours_since_last_alert`, `onsets_last_24h`).
  * **Set D (`FEATURE_SET_D_POST_HOC`: Set C + 3 relative activity features `+Rel Ratios` — 25 features, Post-Hoc Exploratory in Walk-Forward):** Set C + 3 pointwise causal relative activity features in [`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py): `rel_activity_24h_vs_14d`, `adaptive_ratio_7d_vs_30d`, `adaptive_diff_14d_minus_30d`.
* **Isolation of `early_stopping` From `Validation` & Val PR-AUC Selection Rule (Formulated After Initial Test Inspection, [`reports/official/val_evaluation_details.json`](reports/official/val_evaluation_details.json)):**
  * To ensure `Validation` (`2024-05-01` – `2025-05-31`) is used exclusively for $p^* = \arg\max F_2$ threshold calibration (over the unified grid `np.linspace(0.01, 0.70, 175)`) and `Val PR-AUC` model selection, [`src/models/train_models.py`](src/models/train_models.py) runs LightGBM `early_stopping(30)` on an internal chronological tail subsplit `Train_ES` (the last 60 days of `Train` with a 2-hour `embargo_hours = 2` gap between `Train_Fit` and `Train_ES`, with zero overlap with `Validation`).
  * Selected Model on Validation: **`LightGBM (Set C: Spatial + Adaptive)`** on both Official (`Val PR-AUC = 0.2967 [0.288, 0.306]`, `Val F2 = 0.5327 [0.525, 0.539]`, $p^* = 0.1290$) and Volunteer (`Val PR-AUC = 0.3085 [0.301, 0.315]`, `Val F2 = 0.5306 [0.524, 0.537]`, $p^* = 0.1210$ in `naive_as_is` and `Val PR-AUC = 0.3089`, `Val F2 = 0.5314`, $p^* = 0.1171$ in `naive_fixed_D`).
  * Identifying static `Set B` as top-performing by Test PR-AUC and constructing `Set C'` and `Set D` were performed **after inspecting Test results and are explicitly labeled `post-hoc`**.

### Table 3.1. Static Model Benchmark on Full Official Test (`2025-06-01` – `2026-09-07`, `197,083` rows, [`reports/official/ml_models_benchmark.csv`](reports/official/ml_models_benchmark.csv))
*95% confidence intervals are computed via **Day-Block Bootstrap** resampling calendar days (`n_bootstraps = 500`). `Prec @ R=79.2%` reports Precision at the exact Recall level of the Adaptive Baseline on Full Test (`Recall = 79.16%`).*

| Model | Feature Set | Protocol Status | Threshold $p^*$ (Val) | Test PR-AUC [95% CI] | Test $F_2$ [95% CI] | Precision | Recall | Prec @ R=79.2% | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | Lagged onset in $[T-1\text{h}, T)$ | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.500 | — | 0.1034 [0.096, 0.111] | 29.1% | 8.9% | — | 0.1159 |
| **Baseline 2: Historical Lookup** | Train lookup rate (Kyiv) | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.081 | 0.2204 [0.212, 0.228] | 0.5050 [0.497, 0.513] | 22.5% | 73.2% | 20.5% | 0.0868 |
| **Baseline 3: Adaptive 14-day** | 14-day rolling rate | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.085 | 0.3031 [0.292, 0.315] | 0.5293 [0.519, 0.538] | 22.8% | 79.2% | 22.8% | 0.0818 |
| **LogReg (Set A: No Spatial)** | A (10 features) | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.518 | 0.3881 [0.377, 0.399] | 0.5329 [0.524, 0.541] | 23.5% | 77.9% | 23.0% | 0.2007 |
| **LogReg (Set B: With Spatial)** | B (13 features) | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.561 | **0.3903** [0.380, 0.400] | 0.5499 [0.541, 0.558] | 24.0% | 81.2% | 24.8% | 0.2337 |
| **LogReg (Set C: Spatial + Adaptive)** | C (22 features) | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.565 | 0.3174 [0.306, 0.329] | 0.5431 [0.535, 0.550] | 21.1% | **89.3%** | 25.0% | 0.2875 |
| **LogReg (Set C': Adaptive + Lags)** | C' (19 features) | `post-hoc sensitivity (Set C')` | 0.541 | 0.3386 [0.327, 0.350] | 0.5528 [0.544, 0.561] | 24.1% | 81.6% | 25.2% | 0.2265 |
| **LightGBM (Set A: No Spatial)** | A (10 features) | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` | 0.097 | 0.3394 [0.327, 0.351] | 0.5360 [0.527, 0.544] | **25.5%** | 74.1% | 22.8% | 0.0791 |
| **LightGBM (Set B: With Spatial)** | B (13 features) | `candidate model (Val PR-AUC selection rule formulated after initial Test inspection)` *(post-hoc best LGBM PR-AUC)* | 0.113 | 0.3610 [0.350, 0.372] | 0.5594 [0.550, 0.568] | 23.8% | 84.6% | 26.3% | **0.0767** |
| **LightGBM (Set C: Spatial + Adaptive)** | C (22 features) | **`selected on Val by PR-AUC (rule formulated after initial Test inspection)`** | 0.129 | 0.3292 [0.319, 0.339] | 0.5552 [0.547, 0.562] | 22.9% | 86.1% | 25.9% | 0.0785 |
| **LightGBM (Set C': Adaptive + Lags)** | C' (19 features) | `post-hoc sensitivity (Set C')` | 0.117 | 0.3507 [0.339, 0.362] | **0.5651** [0.556, 0.573] | 24.6% | 83.7% | **26.6%** | 0.0768 |

### Table 3.2. Static Model Benchmark on Latest 60 Days Official (`2026-07-09` – `2026-09-07`, `23,646` rows, base rate `12.65%`, [`reports/official/ml_recent_60d_benchmark.csv`](reports/official/ml_recent_60d_benchmark.csv))
*`Prec @ R=90.8%` reports Precision at the actual Recall of `Baseline 3: Adaptive 14-day Rolling` on the latest 60-day slice (`Recall = 90.84%`).*

| Model | Feature Set | Threshold $p^*$ | PR-AUC [95% CI] | $F_2$-Score [95% CI] | Precision | Recall | Prec @ R=90.8% | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | Lag $[T-1\text{h}, T)$ | 0.500 | — | 0.1621 [0.145, 0.181] | 39.2% | 14.1% | — | 0.1364 |
| **Baseline 2: Historical Lookup** | Train lookup (Kyiv) | 0.081 | 0.3257 [0.295, 0.357] | 0.5900 [0.566, 0.614] | 32.7% | 73.8% | 21.8% | 0.1038 |
| **Baseline 3: Adaptive 14-day** | 14-day rolling rate | 0.085 | 0.3682 [0.333, 0.406] | 0.6405 [0.617, 0.662] | 29.4% | 90.8% | 29.4% | 0.0911 |
| **LogReg (Set A: No Spatial)** | A (10 features) | 0.518 | **0.5058** [0.480, 0.531] | 0.6535 [0.636, 0.674] | 35.5% | 82.8% | 28.4% | 0.1693 |
| **LogReg (Set B: With Spatial)** | B (13 features) | 0.561 | 0.4788 [0.444, 0.510] | 0.6501 [0.626, 0.674] | 32.7% | 86.3% | 30.0% | 0.2064 |
| **LogReg (Set C: Spatial + Adaptive)** | C (22 features) | 0.565 | 0.3889 [0.355, 0.424] | 0.6196 [0.592, 0.643] | 25.8% | **95.5%** | 30.4% | 0.2975 |
| **LogReg (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features) | 0.541 | 0.3803 [0.343, 0.417] | 0.6494 [0.622, 0.675] | 30.9% | 89.6% | 30.1% | 0.2304 |
| **LightGBM (Set A: No Spatial)** | A (10 features) | 0.097 | 0.4508 [0.424, 0.477] | **0.6549** [0.637, 0.674] | **36.8%** | 81.4% | 27.9% | 0.0883 |
| **LightGBM (Set B: With Spatial)** *(post-hoc)* | B (13 features) | 0.113 | 0.4484 [0.415, 0.481] | 0.6446 [0.617, 0.670] | 29.9% | 90.6% | 29.6% | **0.0852** |
| **LightGBM (Set C: Spatial + Adaptive)** *(selected on Val by PR-AUC)* | C (22 features) | 0.129 | 0.3830 [0.350, 0.416] | 0.6301 [0.603, 0.653] | 27.1% | 94.1% | 29.3% | 0.0896 |
| **LightGBM (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features) | 0.117 | 0.4031 [0.369, 0.439] | 0.6518 [0.626, 0.675] | 30.6% | 90.9% | **30.6%** | 0.0865 |

<p align="center">
  <img src="reports/official/figures/10_test_pr_curves_comparison.png" alt="Test and Recent 60d PR Curves Comparison" width="95%">
  <br><em>Figure 5. Precision-Recall curves for static models across Feature Sets A, B, C, C' and baselines on Full Test (left) and Latest 60 Days (right) on Official.</em>
</p>

### 3.2. Feature Distribution Shift (Covariate Shift) on Official and Volunteer
[`src/features/analyze_feature_shift.py`](src/features/analyze_feature_shift.py) compares feature means across `Train`, `Val`, and `Test` on both sources ([`reports/comparison/official_vs_volunteer_feature_shift_comparison.csv`](reports/comparison/official_vs_volunteer_feature_shift_comparison.csv)):

| Feature | Official: Train $\to$ Val $\to$ Test | Official: Pre / Post `01.08.2025` | Official: Test / Train | Volunteer (`as_is`): Train $\to$ Val $\to$ Test | Volunteer: Pre / Post `01.08.2025` | Volunteer: Test / Train |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`national_active_count`** | `1.17` $\to$ `3.06` $\to$ **`5.54`** | `3.17` $\to$ `5.95` ($1.88\times$) | **$4.73\times$** | `1.26` $\to$ `2.02` $\to$ **`2.17`** | `1.98` $\to$ `2.20` ($1.11\times$) | **$1.72\times$** |
| **`neighbors_active_count`** | `0.13` $\to$ `0.32` $\to$ **`0.52`** | `0.34` $\to$ `0.55` ($1.62\times$) | **$4.04\times$** | `0.16` $\to$ `0.28` $\to$ **`0.32`** | `0.27` $\to$ `0.32` ($1.21\times$) | **$1.97\times$** |
| **`hours_since_last_alert`** | `32.6` $\to$ `30.1` $\to$ **`42.7`** | `22.8` $\to$ `46.1` ($2.02\times$) | **$1.31\times$** | `31.2` $\to$ `26.9` $\to$ **`30.9`** | `20.5` $\to$ `32.4` ($1.58\times$) | **$0.99\times$** |
| **`adaptive_rate_14d`** | `0.075` $\to$ `0.083` $\to$ **`0.078`** | `0.073` $\to$ `0.079` ($1.07\times$) | **$1.04\times$** | `0.084` $\to$ `0.104` $\to$ **`0.140`** | `0.101` $\to$ `0.145` ($1.44\times$) | **$1.66\times$** |
| **`onsets_last_24h`** | `1.73` $\to$ `1.89` $\to$ **`1.80`** | `1.75` $\to$ `1.81` ($1.03\times$) | **$1.04\times$** | `1.95` $\to$ `2.41` $\to$ **`3.32`** | `2.43` $\to$ `3.44` ($1.42\times$) | **$1.70\times$** |
| **`target_onset_next_1h`** | `0.081` $\to$ `0.102` $\to$ **`0.103`** | `0.096` $\to$ `0.104` ($1.08\times$) | **$1.27\times$** | `0.091` $\to$ `0.115` $\to$ **`0.153`** | `0.114` $\to$ `0.159` ($1.39\times$) | **$1.69\times$** |

<p align="center">
  <img src="reports/comparison/figures/16_comparison_feature_shift_official_vs_volunteer.png" alt="Feature Shift Comparison Official vs Volunteer" width="90%">
  <br><em>Figure 6. Cross-source comparison of feature mean shift ratios (Test / Train) between Official and Volunteer (Figure 16 in reports/comparison/figures/).</em>
</p>

### 3.3. Independent Volunteer Verification & Sensitivity to 3 Causal `naive` Policies
In the Volunteer corpus (`data/raw/volunteer_data_uk.csv`), a subset of historical rows carries the `naive == True` flag (where exact end time was unknown and imputed as `started_at + 30 min`). To verify that findings are not driven by imputation artifacts, [`src/models/run_volunteer_experiment.py`](src/models/run_volunteer_experiment.py) evaluates **three strictly causal policies** ([`reports/volunteer/volunteer_naive_sensitivity.csv`](reports/volunteer/volunteer_naive_sensitivity.csv)):

| `naive` Policy | Calm Hours (Total / Test) | Test Base Rate (`positives`) | Adaptive 14d: Test PR-AUC [95% CI] / Test $F_2$ [95% CI] | LightGBM (Set C, static): Test PR-AUC [95% CI] / Test $F_2$ [95% CI] | Paired $\Delta\text{PR-AUC}$ (`LGBM C − Adaptive`) | Paired $\Delta F_2$ (`LGBM C − Adaptive`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`(a) naive_as_is`** (`started_at + 30 min`) | `871,498` / `256,802` | `15.32%` (`39,329`) | `0.3922 [0.383, 0.401]`<br>`0.5892 [0.583, 0.595]` | `0.3855 [0.378, 0.393]`<br>`0.5949 [0.589, 0.601]` | $-0.0063 \ [-0.0103, -0.0021]$ | **`+0.0056 [+0.0029, +0.0085]`** |
| **`(b) drop_naive`** (drop `naive == True`) | `874,368` / `258,046` | `14.41%` (`37,190`) | `0.3881 [0.379, 0.397]`<br>`0.5731 [0.566, 0.579]` | `0.3822 [0.374, 0.390]`<br>`0.5811 [0.575, 0.587]` | $-0.0055 \ [-0.0096, -0.0012]$ | **`+0.0079 [+0.0052, +0.0108]`** |
| **`(c) naive_fixed_D`** (median $D_{\text{oblast}}$ strictly from `Train`) | `870,931` / `256,485` | `15.33%` (`39,329`) | `0.3924 [0.384, 0.401]`<br>`0.5897 [0.583, 0.596]` | `0.3865 [0.379, 0.394]`<br>`0.5965 [0.590, 0.602]` | $-0.0057 \ [-0.0095, -0.0016]$ | **`+0.0067 [+0.0040, +0.0096]`** |

<p align="center">
  <img src="reports/comparison/figures/18_volunteer_naive_sensitivity_comparison.png" alt="Volunteer naive Sensitivity Comparison" width="90%">
  <br><em>Figure 7. Sensitivity of PR-AUC and F2-Score on the Volunteer Test set across the three causal policies for handling the naive flag (Figure 18 in reports/comparison/figures/).</em>
</p>

<details>
<summary><b>📊 Expand Additional Static Benchmark Tables: Pre/Post 2025-08-01 Sub-Periods & Ablation Study Matrix (Official)</b></summary>

#### Table 3.3. Static Model Results Inside Test Before and After `2025-08-01` ([`reports/official/ml_test_pre_aug2025_benchmark.csv`](reports/official/ml_test_pre_aug2025_benchmark.csv), [`reports/official/ml_test_post_aug2025_benchmark.csv`](reports/official/ml_test_post_aug2025_benchmark.csv))
*Parentheses report Precision at the actual Recall of `Baseline 3: Adaptive 14-day Rolling` on that exact sub-period (`Prec@R=70.3%` pre-`01.08.2025` at `Recall = 70.27%`; `Prec@R=80.6%` post-`01.08.2025` at `Recall = 80.58%`).*

| Model | Pre-01.08.2025 (`28,929`): PR-AUC [95% CI] | Pre-01.08.2025: $F_2$ [95% CI] | Pre-01.08: Prec / Rec (Prec@R=70.3%) | Post-01.08.2025 (`168,154`): PR-AUC [95% CI] | Post-01.08.2025: $F_2$ [95% CI] | Post-01.08: Prec / Rec (Prec@R=80.6%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3: Adaptive 14-day** | 0.2735 [0.259, 0.288] | 0.4616 [0.445, 0.479] | 19.5% / 70.3% (19.5%) | 0.3068 [0.294, 0.320] | 0.5403 [0.530, 0.550] | 23.3% / 80.6% (23.3%) |
| **LogReg (Set A)** | 0.2934 [0.275, 0.313] | 0.4617 [0.443, 0.479] | 18.1% / 75.6% (19.3%) | 0.4025 [0.392, 0.413] | 0.5458 [0.537, 0.555] | 24.7% / 78.2% (23.6%) |
| **LogReg (Set B)** | 0.2877 [0.272, 0.306] | 0.4814 [0.466, 0.498] | 19.9% / 74.6% (21.3%) | **0.4041** [0.395, 0.415] | 0.5615 [0.553, 0.570] | 24.7% / 82.3% (25.4%) |
| **LogReg (Set C)** | 0.2580 [0.243, 0.276] | 0.4978 [0.483, 0.513] | 20.5% / 77.4% (22.2%) | 0.3267 [0.314, 0.339] | 0.5498 [0.541, 0.559] | 21.2% / **91.2%** (25.6%) |
| **LogReg (Set C')** *(post-hoc)* | 0.2782 [0.264, 0.294] | 0.4813 [0.467, 0.495] | 21.5% / 69.6% (21.4%) | 0.3469 [0.334, 0.359] | 0.5639 [0.554, 0.574] | 24.5% / 83.5% (25.7%) |
| **LightGBM (Set A)** | 0.2791 [0.261, 0.300] | 0.4710 [0.455, 0.488] | 20.1% / 70.8% (20.4%) | 0.3498 [0.339, 0.361] | 0.5475 [0.538, 0.557] | **26.5%** / 74.6% (23.0%) |
| **LightGBM (Set B)** *(post-hoc)* | 0.3024 [0.287, 0.319] | 0.5073 [0.492, 0.523] | 21.7% / 76.3% (23.7%) | 0.3699 [0.359, 0.381] | 0.5677 [0.558, 0.577] | 24.1% / 85.9% (26.8%) |
| **LightGBM (Set C)** *(selected on Val by PR-AUC)* | 0.2924 [0.274, 0.314] | 0.5103 [0.493, 0.529] | 23.6% / 72.0% (24.0%) | 0.3354 [0.325, 0.346] | 0.5616 [0.552, 0.570] | 22.9% / 88.3% (26.2%) |
| **LightGBM (Set C')** *(post-hoc)* | **0.3101** [0.294, 0.327] | **0.5120** [0.496, 0.529] | **23.6%** / 72.3% (**24.5%**) | 0.3568 [0.345, 0.369] | **0.5731** [0.564, 0.583] | 24.7% / 85.5% (**26.9%**) |

#### Table 3.4. Paired Bootstrap Metric Differences Across 4 Test Slices ([`reports/official/test_evaluation_details.json`](reports/official/test_evaluation_details.json), Day-Block Bootstrap $N=500$)

| Comparison (Paired Difference, Official) | Full Test: $\Delta$ PR-AUC / $\Delta F_2$ | Pre-01.08.2025: $\Delta$ PR-AUC / $\Delta F_2$ | Post-01.08.2025: $\Delta$ PR-AUC / $\Delta F_2$ | Latest 60 Days: $\Delta$ PR-AUC / $\Delta F_2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Primary (selected on Val by PR-AUC rule, formulated after initial Test inspection):** `LGBM (C) - Adaptive 14d` | **$+0.0260$** $[+0.018, +0.033]$<br>**$+0.0257$** $[+0.020, +0.031]$ | **$+0.0193$** $[+0.001, +0.040]$<br>**$+0.0490$** $[+0.030, +0.070]$ | **$+0.0287$** $[+0.021, +0.036]$<br>**$+0.0212$** $[+0.016, +0.027]$ | $+0.0148$ $[-0.003, +0.032]$<br>**$-0.0110$** $[-0.021, -0.003]$ |
| **Post-hoc:** `LGBM (C') - Adaptive 14d` | **$+0.0477$** $[+0.042, +0.054]$<br>**$+0.0357$** $[+0.031, +0.041]$ | **$+0.0367$** $[+0.025, +0.049]$<br>**$+0.0506$** $[+0.035, +0.069]$ | **$+0.0500$** $[+0.044, +0.056]$<br>**$+0.0328$** $[+0.028, +0.037]$ | **$+0.0350$** $[+0.017, +0.053]$<br>**$+0.0110$** $[+0.002, +0.021]$ |
| **Post-hoc:** `LGBM (C') - LGBM (B)` | **$-0.0104$** $[-0.018, -0.003]$<br>**$+0.0057$** $[+0.003, +0.009]$ | $+0.0076$ $[-0.003, +0.017]$<br>$+0.0047$ $[-0.002, +0.012]$ | **$-0.0130$** $[-0.021, -0.005]$<br>**$+0.0055$** $[+0.002, +0.009]$ | **$-0.0447$** $[-0.069, -0.024]$<br>$+0.0073$ $[-0.002, +0.017]$ |
| **Post-hoc:** `LGBM (C') - LGBM (C)` | **$+0.0217$** $[+0.016, +0.027]$<br>**$+0.0100$** $[+0.007, +0.013]$ | **$+0.0174$** $[+0.003, +0.033]$<br>$+0.0016$ $[-0.007, +0.009]$ | **$+0.0213$** $[+0.015, +0.027]$<br>**$+0.0116$** $[+0.008, +0.015]$ | **$+0.0201$** $[+0.010, +0.031]$<br>**$+0.0220$** $[+0.014, +0.030]$ |
| **LGBM:** `Set B - Set A` (neighbors at $T$) | **$+0.0217$** $[+0.018, +0.025]$<br>**$+0.0233$** $[+0.017, +0.030]$ | **$+0.0234$** $[+0.015, +0.033]$<br>**$+0.0369$** $[+0.024, +0.049]$ | **$+0.0202$** $[+0.015, +0.024]$<br>**$+0.0201$** $[+0.014, +0.026]$ | $-0.0022$ $[-0.018, +0.011]$<br>$-0.0111$ $[-0.032, +0.009]$ |
| **LGBM:** `Set C - Set B` | **$-0.0321$** $[-0.041, -0.024]$<br>**$-0.0043$** $[-0.009, -0.000]$ | $-0.0098$ $[-0.025, +0.007]$<br>$+0.0032$ $[-0.007, +0.014]$ | **$-0.0343$** $[-0.044, -0.025]$<br>**$-0.0060$** $[-0.010, -0.002]$ | **$-0.0648$** $[-0.092, -0.040]$<br>**$-0.0147$** $[-0.023, -0.005]$ |
</details>

---

## 🔄 4. Stage 2 (Block B): Monthly Walk-Forward Retraining

### 4.1. Causal Walk-Forward Protocol ([`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py))
To evaluate how periodic tree refitting adapts to the 2025–2026 alert regime, [`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py) implements a strictly causal monthly `Walk-Forward Retraining` protocol:
* **Refit Cadence:** Every calendar month $M$ of the Test split starting from `2025-06` (`16` test months `2025-06..2026-09` on Official; `17` test months `2025-06..2026-10` on Volunteer `naive_fixed_D`).
* **Training & Internal Validation Windows Before Month $M$:**
  * `WF Rolling 12m`: 12 calendar months of training data `Train_M` + the 2 most recent months prior to $M$ as internal validation `Val_M`. To ensure `Val_M` is used exclusively for $p^*_M = \arg\max F_2(\text{Val}_M)$ threshold calibration without early-stopping overlap, LightGBM `early_stopping(30)` is fitted on an internal chronological tail split `Train_ES` (the last 30 days of `Train_M` with a 2-hour `embargo_hours = 2` gap between `Train_Fit` and `Train_ES`).
  * `WF Expanding`: Full history from March 2022 up to `Val_M` (also using an internal 30-day `Train_ES` subsplit for `early_stopping(30)`) + 2 months `Val_M` for $p^*_M$ calibration.
* **Strict 2-Hour `embargo` Buffer:** Between `Train_Fit` and `Train_ES`, between `Train_M` and `Val_M`, and between `Val_M` and the start of test month $M$, the final 2 hours are excluded (`embargo_hours = 2`), preventing any overlap of target windows $[T, T+1\text{h})$ across split boundaries.
* **Hypothesis Status in the Walk-Forward Block (`protocol_status`):**
  * `LightGBM (Set C) [WF Rolling 12m]` — **`primary hypothesis of the WF block (specified after viewing static Test results)`**.
  * `LogReg (Set C) [WF Rolling 12m]` — `exploratory (Set C under Val PR-AUC selection rule, formulated after initial Test inspection)`.
  * `LightGBM / LogReg (Set C) [WF Expanding]` — `exploratory (expanding window)`.
  * `LightGBM / LogReg (Set D: +Rel Ratios) [WF Rolling 12m / Expanding, post-hoc]` — `post-hoc exploratory (Set D)` (testing the 3 dimensionless relative activity ratios in `Set D`, 25 features).
* **Static Reference in WF Tables (`Static Initial Train`):** Uses the frozen artifact [`models/lightgbm_adaptive.joblib`](models/lightgbm_adaptive.joblib) ($p^*=0.1290$) on Official and the frozen artifact **[`models/volunteer_lightgbm_adaptive_fixed_d.joblib`](models/volunteer_lightgbm_adaptive_fixed_d.joblib)** ($p^*=0.1171$, trained and evaluated strictly on the causal `naive_fixed_D` grid) on Volunteer.

### Table 4.1. Consolidated Walk-Forward Benchmark vs. Static Models & Baselines (All 14 Rows From [`reports/comparison/official_vs_volunteer_walk_forward_comparison.csv`](reports/comparison/official_vs_volunteer_walk_forward_comparison.csv))

| Row / Model in WF Protocol | Protocol Status (`protocol_status`) | Official Full Test (`197,083`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) | Official Latest 60 Days (`23,646`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) | Volunteer `fixed_D` Full Test (`256,485`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) | Volunteer `fixed_D` Latest 60 Days (`30,348`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) |
| :--- | :--- | :---: | :---: | :---: | :---: |


<p align="center">
  <img src="reports/comparison/figures/19_walk_forward_vs_static_comparison.png" alt="Monthly Walk-Forward vs Static Dynamics" width="95%">
  <br><em>Figure 8. Monthly PR-AUC and F2-Score trajectory for Walk-Forward Retraining (Rolling 12m), static LightGBM (Set C), and Adaptive 14-day Baseline across Official and Volunteer (Figure 19 in reports/comparison/figures/).</em>
</p>

### 4.2. Walk-Forward Interpretation, `LogReg Static vs. LGBM WF` Caveat & Non-Causal Phrasing
1. **Why Row 14 (`LogReg Static проти LGBM WF`) Is Explicitly Reported:**
   * On **Official**, monthly `LightGBM (Set C) [WF Rolling 12m]` outperforms all static and linear alternatives, beating `LogReg Static` by **$\Delta\text{PR-AUC} = +0.0986 \ [+0.0897, +0.1076]$** on Full Test and **$+0.1133 \ [+0.0871, +0.1402]$** on the latest 60 days.
   * On **Volunteer (`naive_fixed_D`)**, however, while monthly refitting significantly lifts `LightGBM (Set C) [WF Rolling 12m]` over static LightGBM ($\Delta_{\text{WF − Static}} = +0.0131 \ [+0.0092, +0.0164]$ on Full Test and $+0.0334 \ [+0.0243, +0.0427]$ on the latest 60 days) and beats `Adaptive 14-day` in `14/17` months (and `Static` in `13/17` months), **`LightGBM` cannot be called unconditionally best on Volunteer**. Static linear `LogReg (Set C) [Static Initial Train]`, which extrapolates monotonically under rising alert activity, retains a higher `PR-AUC` on both Full Test (`0.4098` vs. `0.3997`, $\Delta\text{PR-AUC} = -0.0101 \ [-0.0137, -0.0064]$) and the latest 60 days (`0.5003` vs. `0.4913`, $\Delta\text{PR-AUC} = -0.0089 \ [-0.0146, -0.0029]$). `LightGBM WF` outperforms `LogReg Static` on Volunteer only on thresholded $F_2$-Score ($\Delta F_2 = +0.0047 \ [+0.0027, +0.0068]$ on Full Test and $+0.0051 \ [+0.0017, +0.0089]$ on the latest 60 days) and probability calibration (`Brier = 0.1095` vs. `0.2646`).
2. **Non-Causal Phrasing (`is consistent with` rather than causal claims):**
   * The observed gain of `LightGBM (Set C) [WF Rolling 12m]` over frozen `LightGBM Static` on Official ($\Delta_{\text{WF − Static}} = +0.0868 \ [+0.0776, +0.0951]$ on Full Test and $+0.1190 \ [+0.0898, +0.1466]$ on the latest 60 days) and on Volunteer ($\Delta_{\text{WF − Static}} = +0.0131 \ [+0.0092, +0.0164]$ on Full Test and $+0.0334 \ [+0.0243, +0.0427]$ on the latest 60 days) **is consistent with (`узгоджується з`) the hypothesis** that periodic refitting adapts tree split thresholds to the post-June 2025 regime shift and relieves split saturation when features exceed their 2022–2024 training range. Because no separate causal intervention test was conducted to isolate a single cause, we strictly use the phrasing "is consistent with".

### 4.3. Hyperparameter Alignment Audit: Initial WF Run vs. Aligned Run (`hyperparameter_alignment_history`)
As documented in [`reports/volunteer/volunteer_walk_forward_details.json`](reports/volunteer/volunteer_walk_forward_details.json) (`methodological_notes.hyperparameter_alignment_history`) and [`notebooks/05_ml_forecasting_experiment.ipynb`](notebooks/05_ml_forecasting_experiment.ipynb), during the initial (pre-commit) run of [`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py), `fit_fold_lightgbm` used different hyperparameters (`n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, min_child_samples=120, class_weight='balanced', eval_metric='average_precision'`) than the canonical static training script [`src/models/train_models.py`](src/models/train_models.py) (`n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=50`, no `class_weight`, `eval_metric='binary_logloss'` on an internal `Train_ES` subsplit with a 2-hour `embargo`), and `Static LightGBM` was refit with those same parameters instead of loading the frozen `.joblib` artifact (while in the first WF commit `58628f8`, `Static Initial Train` on Volunteer used `volunteer_lightgbm_adaptive.joblib` trained on `naive_as_is`). After aligning `fit_fold_lightgbm` to match `train_models.py` and switching `Static Initial Train` to [`models/volunteer_lightgbm_adaptive_fixed_d.joblib`](models/volunteer_lightgbm_adaptive_fixed_d.joblib) ($p^*=0.1171$), the conclusion for `LightGBM (Set C) [WF Rolling 12m]` on Volunteer changed from statistical parity with `Adaptive 14-day` to statistically significantly outperforming it:

#### Table 4.2. Initial Run vs. Aligned Run on Volunteer (`naive_fixed_D`, [`reports/volunteer/volunteer_walk_forward_details.json`](reports/volunteer/volunteer_walk_forward_details.json))

| Evaluation Slice (`Volunteer, naive_fixed_D`) | Model (`Set C`) | Pre-Commit Run (unaligned WF hyperparams) / Commit `58628f8` (`Static` from `as_is`): Pooled PR-AUC [95% CI] & $\Delta$ vs. Adaptive | Aligned Run (`train_models.py` hyperparams with `Train_ES`; `Static` from `fixed_d.joblib`): Pooled PR-AUC [95% CI] & $\Delta$ vs. Adaptive |
| :--- | :--- | :--- | :--- |
| **Full Test** (`2025-06` — `2026-10`, `256,485` rows) | **LightGBM (Set C) [WF Rolling 12m]** | Pre-commit: `0.3935 [0.385, 0.401]`<br>$\Delta = +0.0012 \ [-0.0040, +0.0067]$ *(parity with Adaptive)* | **`0.3997 [0.391, 0.407]`**<br>**$\Delta = +0.0074 \ [+0.0027, +0.0121]$** *(beats Adaptive)* |
| **Full Test** (`2025-06` — `2026-10`, `256,485` rows) | **LightGBM (Set C) [Static Initial Train]** | Pre-commit: `0.3684 [0.360, 0.377]`, $\Delta = -0.0239 \ [-0.0285, -0.0194]$<br>Commit `58628f8` (`as_is.joblib`): `0.3881 [0.381, 0.395]`, $\Delta = -0.0040 \ [-0.0085, +0.0004]$ | **`0.3865 [0.379, 0.394]`** (`fixed_d.joblib`)<br>$\Delta = -0.0057 \ [-0.0095, -0.0016]$ |
| **Latest 60 Days** (`2026-08-12` — `2026-10-11`, `30,348` rows) | **LightGBM (Set C) [WF Rolling 12m]** | Pre-commit: `0.4933 [0.480, 0.509]`<br>$\Delta = +0.0125 \ [+0.0042, +0.0215]$ | **`0.4913 [0.478, 0.507]`**<br>**$\Delta = +0.0105 \ [+0.0016, +0.0199]$** *(beats Adaptive)* |
| **Latest 60 Days** (`2026-08-12` — `2026-10-11`, `30,348` rows) | **LightGBM (Set C) [Static Initial Train]** | Pre-commit: `0.4261 [0.415, 0.440]`, $\Delta = -0.0548 \ [-0.0641, -0.0449]$<br>Commit `58628f8` (`as_is.joblib`): `0.4519 [0.438, 0.468]`, $\Delta = -0.0290 \ [-0.0390, -0.0182]$ | **`0.4579 [0.443, 0.476]`** (`fixed_d.joblib`)<br>$\Delta = -0.0229 \ [-0.0321, -0.0134]$ |

> **Note on Intermediate Drafts:** The numbers `0.4094` and `0.4793` ($\Delta = -0.0017$) mentioned in intermediate draft notes were recalled from memory, have no confirmation in CSVs or git history, and are explicitly noted as unverified.

---

## 🗺️ 5. Regional Walk-Forward Breakdown Across 24 Oblasts (Exploratory)

[`src/models/analyze_walk_forward_regional.py`](src/models/analyze_walk_forward_regional.py) decomposes out-of-fold Walk-Forward predictions across all **24 oblasts of Ukraine** without excluding any region by sample threshold ([`reports/comparison/official_vs_volunteer_walk_forward_regional.csv`](reports/comparison/official_vs_volunteer_walk_forward_regional.csv), [`reports/comparison/official_vs_volunteer_walk_forward_regional_clusters.csv`](reports/comparison/official_vs_volunteer_walk_forward_regional_clusters.csv)):
* **Methodological Status (`exploratory`):** Because `24 oblasts × 2 sources = 48` subsets are evaluated simultaneously (multiple comparisons), this regional breakdown is strictly descriptive (`exploratory`) and makes no claims of statistical significance for individual oblasts.
* **Why Only Paired Within-Oblast Differences $\Delta\text{PR-AUC}$ Are Comparable Across Regions:** Raw within-oblast `PR-AUC` mechanically depends on the local `base_rate` (which ranges from `1.3%–1.5%` in Zakarpattia and Ivano-Frankivsk to `47%–60%` in Kharkiv, Sumy, and Donetsk). Therefore, only **paired within-oblast gains** $\Delta\text{PR-AUC (WF − Static)}$ and $\Delta\text{PR-AUC (WF − Adaptive)}$ (Day-Block Bootstrap $N=500$, `seed=42`, `date_kyiv` derived from `timestamp_utc`) are meaningful across regions.
* **Small-Sample Flag (`low_sample_flag = True` when `n_positives < 150` or `n_calm_hours < 1500`):** All 24 oblasts are retained; the flag marks **3 oblasts on Official** (Donetsk: `1,037` calm hours, `623` onsets; Dnipropetrovsk: `1,338` hours, `137` onsets; Zakarpattia: `10,852` hours, `147` onsets) and **1 oblast on Volunteer** (Zakarpattia: `11,630` hours, `147` onsets).
* **Grouping Strictly by Train-Only Clusters (`активність 2022–04.2024`):**

### Table 5.1. Regional Walk-Forward Gains Aggregated by Train-Only Clusters ([`reports/comparison/official_vs_volunteer_walk_forward_regional_clusters.csv`](reports/comparison/official_vs_volunteer_walk_forward_regional_clusters.csv))

| Dataset | Train-Only Cluster (`активність 2022–04.2024`) | Oblasts (`low_sample`) | Mean `base_rate` | Median PR-AUC: Adapt / Static / WF 12m | Mean / Median $\Delta\text{PR-AUC (WF − Static)}$ | Mean / Median $\Delta\text{PR-AUC (WF − Adapt)}$ | Oblasts with $\Delta_{\text{WF−Static}} > 0$ | Oblasts with $\Delta_{\text{WF−Adapt}} > 0$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official** | `high_activity (активність 2022–04.2024)` | 12 (2) | 29.08% | 0.3100 / 0.3471 / **0.3393** | **`+0.0443` / `+0.0186`** | **`+0.0877` / `+0.0504`** | **`11 / 12`** | **`12 / 12`** |
| **Official** | `medium_activity (активність 2022–04.2024)` | 5 (0) | 6.11% | 0.0720 / 0.1567 / **0.1812** | **`+0.0332` / `+0.0349`** | **`+0.1148` / `+0.1028`** | **`5 / 5`** | **`5 / 5`** |
| **Official** | `low_activity (активність 2022–04.2024)` | 7 (1) | 1.74% | 0.0189 / 0.1153 / **0.1281** | **`+0.0187` / `+0.0187`** | **`+0.1182` / `+0.1091`** | **`7 / 7`** | **`7 / 7`** |
| **Volunteer (`naive_fixed_D`)** | `high_activity (активність 2022–04.2024)` | 10 (0) | 28.87% | 0.3156 / 0.3300 / **0.3448** | **`+0.0189` / `+0.0155`** | **`+0.0299` / `+0.0349`** | **`7 / 10`** | **`10 / 10`** |
| **Volunteer (`naive_fixed_D`)** | `medium_activity (активність 2022–04.2024)` | 7 (0) | 9.89% | 0.1329 / 0.1847 / **0.1872** | **`+0.0074` / `+0.0033`** | **`+0.0463` / `+0.0435`** | **`5 / 7`** | **`7 / 7`** |
| **Volunteer (`naive_fixed_D`)** | `low_activity (активність 2022–04.2024)` | 7 (1) | 4.40% | 0.0224 / 0.0714 / **0.0768** | **`+0.0029` / `+0.0054`** | **`+0.0253` / `+0.0192`** | **`6 / 7`** | **`7 / 7`** |

* Across all 24 oblasts, `LightGBM (Set C) [WF Rolling 12m]` outperforms `Adaptive 14-day` in **`24/24` oblasts on Official** and **`24/24` oblasts on Volunteer (`naive_fixed_D`)**, and outperforms frozen `Static Initial Train` in **`23/24` oblasts on Official** and **`18/24` oblasts on Volunteer (`naive_fixed_D`)**. The largest gains from monthly retraining occur in frontline and central oblasts within `high_activity` and `medium_activity`.

<p align="center">
  <img src="reports/comparison/figures/20_walk_forward_regional_breakdown.png" alt="Regional Walk-Forward Breakdown Across 24 Oblasts" width="95%">
  <br><em>Figure 9. Paired within-oblast ΔPR-AUC (WF Rolling 12m − Static Initial Train) across all 24 oblasts of Ukraine grouped by Train-only clusters (2022–04.2024), with low_sample_flag (*) indicators (Figure 20 in reports/comparison/figures/).</em>
</p>

<details>
<summary><b>🗺️ Expand Complete 24-Oblast Walk-Forward Table (`reports/comparison/official_vs_volunteer_walk_forward_regional.csv`)</b></summary>

| Oblast | Train Cluster (Official) | Official: hrs / onsets (`base_rate`) | Official: PR-AUC Adapt / Static / WF 12m | Official: $\Delta_{\text{WF−Static}}$ [95% CI] | Volunteer (`fixed_D`): hrs / onsets (`base_rate`) | Volunteer: PR-AUC Adapt / Static / WF 12m | Volunteer: $\Delta_{\text{WF−Static}}$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Донецька область** | `high_activity` | 1,037 / 623 (60.1%)* | 0.5606 / 0.6021 / 0.6608 | $+0.0587 \ [+0.0204, +0.0943]$ | 9,010 / 1,808 (20.1%) | 0.2273 / 0.2241 / 0.2634 | $+0.0393 \ [+0.0303, +0.0480]$ |
| **Харківська область** | `high_activity` | 2,957 / 1,549 (52.4%) | 0.5011 / 0.5674 / 0.6962 | $+0.1288 \ [+0.1021, +0.1554]$ | 9,461 / 4,461 (47.2%) | 0.5636 / 0.5516 / 0.5975 | $+0.0459 \ [+0.0329, +0.0599]$ |
| **Сумська область** | `high_activity` | 2,714 / 1,419 (52.3%) | 0.4856 / 0.5758 / 0.6950 | $+0.1192 \ [+0.0917, +0.1454]$ | 9,496 / 3,419 (36.0%) | 0.4041 / 0.4137 / 0.4430 | $+0.0293 \ [+0.0184, +0.0412]$ |
| **Запорізька область** | `high_activity` | 4,378 / 1,664 (38.0%) | 0.3621 / 0.4303 / 0.5824 | $+0.1521 \ [+0.1252, +0.1751]$ | 9,522 / 2,567 (27.0%) | 0.2962 / 0.3052 / 0.3379 | $+0.0327 \ [+0.0220, +0.0435]$ |
| **Чернігівська область** | `high_activity` | 4,416 / 1,408 (31.9%) | 0.4124 / 0.4688 / 0.4969 | $+0.0281 \ [+0.0001, +0.0543]$ | 9,608 / 2,723 (28.3%) | 0.3351 / 0.3547 / 0.3516 | $-0.0031 \ [-0.0140, +0.0083]$ |
| **Полтавська область** | `high_activity` | 7,663 / 1,769 (23.1%) | 0.2908 / 0.3089 / 0.3279 | $+0.0190 \ [+0.0051, +0.0343]$ | 10,253 / 2,320 (22.6%) | 0.2698 / 0.2824 / 0.2840 | $+0.0016 \ [-0.0065, +0.0084]$ |
| **Миколаївська область** | `high_activity` | 8,919 / 1,733 (19.4%) | 0.3203 / 0.3526 / 0.3368 | $-0.0158 \ [-0.0352, +0.0018]$ | 10,348 / 2,622 (25.3%) | 0.3821 / 0.4031 / 0.4010 | $-0.0021 \ [-0.0135, +0.0119]$ |
| **Херсонська область** | `high_activity` | 8,498 / 1,629 (19.2%) | 0.2574 / 0.2940 / 0.3063 | $+0.0122 \ [-0.0040, +0.0280]$ | 10,399 / 2,268 (21.8%) | 0.2463 / 0.2935 / 0.2895 | $-0.0040 \ [-0.0135, +0.0057]$ |
| **Одеська область** | `high` (Off) / `med` (Vol) | 9,265 / 1,642 (17.7%) | 0.2997 / 0.3416 / 0.3419 | $+0.0003 \ [-0.0159, +0.0173]$ | 10,885 / 2,097 (19.3%) | 0.3055 / 0.3285 / 0.3234 | $-0.0051 \ [-0.0151, +0.0055]$ |
| **Кіровоградська область** | `high_activity` | 9,313 / 1,319 (14.2%) | 0.1946 / 0.2208 / 0.2291 | $+0.0084 \ [-0.0047, +0.0224]$ | 10,656 / 1,667 (15.6%) | 0.2233 / 0.2361 / 0.2365 | $+0.0004 \ [-0.0071, +0.0090]$ |
| **Черкаська область** | `high` (Off) / `med` (Vol) | 9,858 / 1,034 (10.5%) | 0.1222 / 0.1718 / 0.1740 | $+0.0022 \ [-0.0100, +0.0165]$ | 11,003 / 1,238 (11.3%) | 0.1427 / 0.1847 / 0.1872 | $+0.0026 \ [-0.0066, +0.0123]$ |
| **Дніпропетровська область** | `high_activity` | 1,338 / 137 (10.2%)* | 0.1024 / 0.0960 / 0.1141 | $+0.0181 \ [+0.0065, +0.0391]$ | 9,917 / 4,439 (44.8%) | 0.5469 / 0.5403 / 0.5895 | $+0.0492 \ [+0.0372, +0.0618]$ |
| **Київська область** | `medium_activity` | 9,011 / 1,082 (12.0%) | 0.1513 / 0.1873 / 0.2175 | $+0.0302 \ [+0.0141, +0.0515]$ | 10,964 / 1,721 (15.7%) | 0.2915 / 0.3250 / 0.3351 | $+0.0101 \ [-0.0017, +0.0204]$ |
| **м. Київ** | `medium_activity` | 10,304 / 685 (6.6%) | 0.0830 / 0.1567 / 0.1812 | $+0.0245 \ [+0.0066, +0.0424]$ | 11,288 / 950 (8.4%) | 0.1284 / 0.1383 / 0.1416 | $+0.0033 \ [-0.0064, +0.0107]$ |
| **Житомирська область** | `medium_activity` | 10,096 / 578 (5.7%) | 0.0720 / 0.1841 / 0.2231 | $+0.0390 \ [+0.0129, +0.0651]$ | 11,250 / 795 (7.1%) | 0.1329 / 0.1896 / 0.1945 | $+0.0049 \ [-0.0066, +0.0181]$ |
| **Вінницька область** | `medium_activity` | 10,523 / 409 (3.9%) | 0.0419 / 0.1098 / 0.1447 | $+0.0349 \ [+0.0168, +0.0578]$ | 11,550 / 550 (4.8%) | 0.0982 / 0.1344 / 0.1266 | $-0.0078 \ [-0.0220, +0.0039]$ |
| **Хмельницька область** | `medium_activity` | 10,769 / 248 (2.3%) | 0.0248 / 0.1435 / 0.1808 | $+0.0373 \ [+0.0090, +0.0659]$ | 11,502 / 318 (2.8%) | 0.0557 / 0.1269 / 0.1705 | $+0.0436 \ [+0.0169, +0.0681]$ |
| **Рівненська область** | `low_activity` | 10,675 / 279 (2.6%) | 0.0350 / 0.1650 / 0.1899 | $+0.0249 \ [-0.0008, +0.0519]$ | 11,599 / 391 (3.4%) | 0.0856 / 0.0958 / 0.0959 | $+0.0001 \ [-0.0157, +0.0136]$ |
| **Волинська область** | `low_activity` | 10,854 / 208 (1.9%) | 0.0235 / 0.1290 / 0.1569 | $+0.0280 \ [-0.0079, +0.0580]$ | 10,263 / 1,299 (12.7%) | 0.1724 / 0.1850 / 0.1857 | $+0.0007 \ [-0.0082, +0.0095]$ |
| **Тернопільська область** | `low_activity` | 10,895 / 184 (1.7%) | 0.0184 / 0.1331 / 0.1625 | $+0.0294 \ [-0.0072, +0.0711]$ | 11,465 / 208 (1.8%) | 0.0223 / 0.0714 / 0.0768 | $+0.0054 \ [-0.0183, +0.0274]$ |
| **Чернівецька область** | `low_activity` | 10,899 / 180 (1.7%) | 0.0189 / 0.1153 / 0.1274 | $+0.0121 \ [-0.0200, +0.0461]$ | 11,493 / 207 (1.8%) | 0.0224 / 0.0359 / 0.0438 | $+0.0079 \ [-0.0025, +0.0220]$ |
| **Львівська область** | `low_activity` | 10,917 / 165 (1.5%) | 0.0190 / 0.1094 / 0.1281 | $+0.0187 \ [-0.0203, +0.0555]$ | 11,134 / 937 (8.4%) | 0.1494 / 0.1755 / 0.1686 | $-0.0069 \ [-0.0190, +0.0039]$ |
| **Івано-Франківська область** | `low_activity` | 10,932 / 159 (1.5%) | 0.0165 / 0.1027 / 0.1029 | $+0.0002 \ [-0.0341, +0.0327]$ | 11,789 / 177 (1.5%) | 0.0173 / 0.0498 / 0.0575 | $+0.0077 \ [-0.0161, +0.0258]$ |
| **Закарпатська область** | `low_activity` | 10,852 / 147 (1.4%)* | 0.0149 / 0.0884 / 0.1058 | $+0.0174 \ [-0.0190, +0.0558]$ | 11,630 / 147 (1.3%)* | 0.0152 / 0.0283 / 0.0337 | $+0.0055 \ [-0.0110, +0.0179]$ |

*An asterisk (`*`) denotes regions with `low_sample_flag = True` (`n_positives < 150` or `n_calm_hours < 1500`).*
</details>

---

## 🛡️ 6. Causal Invariance Verification, Limitations & Future Work

### 6.1. Six-Tier Causal Invariance Automated Test ([`tests/test_leakage_invariance.py`](tests/test_leakage_invariance.py))
[`tests/test_leakage_invariance.py`](tests/test_leakage_invariance.py) verifies zero future information leakage across 6 tiers:

| Verification Tier | What Is Tested | Passing Criterion (`PASS`) |
| :--- | :--- | :--- |
| **1. Grid & Feature Set Integrity** | Zero duplicates in `alerts_clean_oblast.parquet`, `ml_features_hourly.parquet`, `volunteer_alerts_clean_*.parquet`, `volunteer_ml_features_hourly_*.parquet`, and `walk_forward_oof_predictions_*.parquet`, and exact column schemas of `Set A` (10), `Set B` (13), `Set C` (22), `Set C'` (19), and `Set D` (25 features) | `assert duplicates == 0` and exact column match |
| **2. Train-Only Clusters Truncation Invariance** | Physical truncation of `alerts_clean_oblast.parquet` and `volunteer_alerts_clean_*.parquet` at `2024-04-30 23:59:59 UTC` with censoring of `finished_at` at `train_end`, followed by re-running `generate_train_only_clusters()` | Bit-for-bit match of cluster IDs and all 6 cluster passport metrics against `clusters_train_only.csv` and `volunteer_clusters_train_only*.csv` |
| **3. End-to-End Future Truncation Invariance From Raw CSVs to Feature Store at $119$ Timestamps $T$** | Verifies raw-to-clean causal invariance of `cleaner.py` (where the $<60\text{ s}$ micro-alert filter only drops alerts ending within the same hour window `dt.ceil('h')`), plus future truncation across $19$ boundary timestamps + $100$ random timestamps $T$ (`2,399` region-hour rows on Official and `2,648` on Volunteer `naive_fixed_D`) | Exact agreement between cleaned intervals, recomputed `is_active_T == 0`, and all 25 features in Sets A, B, C, C', D vs. the saved Feature Store (`max_diff < 1e-6`) |
| **4. Target Window $[T, T + 1\text{h})$ Dependence** | Verifies `target_onset_next_1h` across all test rows on both datasets | Exact match with binary onset indicator strictly within the half-open interval $[T, T + 1\text{h})$ |
| **5. Causal `naive_fixed_D` Medians in `cleaner.py`** | Physical truncation of raw `volunteer_data_uk.csv` at `2024-04-30 23:59:59 UTC` with censoring of `finished_at` at `train_end` inside `compute_train_median_durations_volunteer()` | Exact match of $D_{\text{oblast}}$ across all 24 regions when `Val + Test` rows are removed |
| **6. Walk-Forward Split, `embargo` & All Saved OOF Columns Invariance** | Verifies the 2-hour `embargo_hours = 2` buffer between `Train_Fit`, `Train_ES`, `Val_M`, and test month $M$, weight invariance when corrupting all rows $\ge T_{\text{start}}(M) - 1\text{h}$, `SHA-256` verification, and exact reproduction of all saved OOF columns (`p_adapt`, `p_roll/exp_lgb/lr_c/d`, `b_*`, `thr_*`) when trained on data physically truncated at $T_{\text{start}}(M) - 2\text{h}$ | Zero overlap of target windows $[T, T+1\text{h})$ and exact match of all saved OOF Parquet columns against models trained on strictly truncated past data |

### 6.2. Methodological Limitations & Caveats
1. **Transparent Chronology of the Val PR-AUC Selection Rule and Post-Hoc Hypotheses:** The Val PR-AUC selection rule (formulated after initial Test inspection) designated `LightGBM (Set C)` vs. `Adaptive 14-day Rolling` as the primary comparison in the static block. Identifying static `Set B` as top-performing by PR-AUC, constructing `Set C'`, adding the relative activity ratios `Set D (+Rel Ratios)`, and formulating the `Walk-Forward Retraining` block (`primary hypothesis of the WF block (specified after viewing static Test results)`) were done after viewing static Test results.
2. **Multiple Comparisons & `base_rate` Dependence in Regional Breakdown:** Evaluating 24 oblasts across two sources (`48` slices) is strictly `exploratory`. Raw within-oblast `PR-AUC` depends mechanically on regional `base_rate`, so only paired within-oblast differences $\Delta\text{PR-AUC}$ are meaningful across regions.
3. **Sub-Regional Aggregation After 2025-08-01 on Official:** Reconstructing oblast alerts as the union of all district intervals preserves series continuity after the August 2025 logging reform, but reduces calm hours in frontline oblasts (notably Donetsk and Dnipropetrovsk on Official, which are flagged with `low_sample_flag = True`).
4. **Strictly Causal 60-Second Micro-Alert Filter:** In [`src/data/cleaner.py`](src/data/cleaner.py), records with `0 <= duration < 60 s` are removed only when both `started_at` and `finished_at` fall within the same hourly window (`started_at.dt.ceil('h') == finished_at.dt.ceil('h')`). Alerts that started less than 60 seconds before an hour boundary $T$ and are still ongoing at $T$ (`3` records on Official and `7` on Volunteer across the entire dataset) are retained in the history, guaranteeing 100% real-time causal invariance.

### 6.3. 🔭 Future Work: Meteorological Forecast Features (Open-Meteo Weather Forecasts)
In this study, we intentionally restricted features to endogenous spatio-temporal alert dynamics and relative activity ratios (`Set D`), leaving meteorological forecast integration for **Future Work** due to the following causality and historical coverage constraints:
1. **Causality and Historical Depth Constraints of NWP Archives (Open-Meteo `Historical Forecast` vs. `Previous Runs`):**
   * Using reanalysis observations (`ERA5`) over $[T, T+1\text{h})$ introduces direct future leakage, whereas using only past observations up to $T$ omits synoptic forecast trajectories for the upcoming hour.
   * **`Open-Meteo Historical Forecast API` (Single Runs)** stitches the initial hours of consecutive NWP model runs without preserving the exact forecast state available after accounting for real-world publication delay (`model availability lag`, typically 3–6 hours after the `00z/06z/12z/18z` initialization time).
   * Conversely, the causal **`Open-Meteo Previous Runs API`** has incomplete coverage for the early period of the full-scale invasion (`2022–02.2024`: most series start in March 2024 or September 2025), preventing homogeneous gap-free training over the `Train` period (`2022-03` – `2024-04`).
2. **Explicit Disclosure of Assumptions:**
   * **Expected Metric Impact Is an Assumption:** The expectation that adding causal NWP forecasts of 80–120m wind speed/gusts, low cloud cover, precipitation, and visibility could yield an incremental gain on the order of $\Delta\text{PR-AUC} \approx +0.005..+0.015$ during drone-dominated periods is an **untested assumption**, as no weather experiment was executed in this repository.
   * **Physical Arguments Regarding Weapon Types and Launch Sites Are Assumptions:** Arguments that strong headwinds/crosswinds and icing constrain *Shahed*-type UAV operations while ballistic and cruise missiles are largely insensitive to surface weather in the target oblast (or depend instead on weather at remote launch sites and over the Black/Caspian Seas) are **domain assumptions** requiring empirical verification on causal NWP archives with publication-lag controls.

---

## 🚀 7. Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Yaroo-a/ukrainian-air-raid-sirens-analysis.git
cd ukrainian-air-raid-sirens-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Full End-to-End Reproducible Run (`bash run_all.sh`)
[`run_all.sh`](run_all.sh) sequentially executes all 9 stages of the pipeline from scratch (upstream SHA-256 verification, Official & Volunteer cleaning, cross-source interval comparison, RQ1–RQ3, Feature Store generation, static training, Volunteer experiments, Walk-Forward Retraining and 24-oblast regional breakdown, execution of all 5 Jupyter Notebooks, and automated verification/leakage tests; total `run_all.sh` runtime: **`2,649 s` (~44 min 09 s), measured by the author during the last full run; not stored in reports**; JSON-recorded runtime of the individual `run_walk_forward_experiment.py` stage: **`503.78 s`** Official, **`503.96 s`** Volunteer):
```bash
bash run_all.sh
```
Or step by step (matching [`run_all.sh`](run_all.sh) 1-to-1):
```bash
# [1/9] Verify raw data sources (SHA-256 hashes and duplicate block structure)
python src/data/verify_upstream_source.py

# [2/9] Clean and merge intervals (Official + Volunteer: naive_as_is, drop_naive, naive_fixed_D)
python src/data/cleaner.py

# [3/9] Compare Official vs. Volunteer interval counts and durations
python src/data/compare_official_vs_volunteer.py

# [4/9] Stage 1 exploratory modules RQ1–RQ3 and Train-only clusters (Official)
python src/features/cyclicity.py
python src/features/clustering.py
python src/features/propagation.py

# [5/9] Generate Feature Store and analyze covariate shift (Official)
python src/features/feature_store.py
python src/features/analyze_feature_shift.py

# [6/9] Train baselines, static models (A, B, C, C'), and evaluate on Test with Day-Block Bootstrap N=500 (Official)
python src/models/baselines.py
python src/models/train_models.py
python src/models/evaluate_test.py

# [7/9] Full Volunteer experiment (3 naive policies), cross-source comparison, Walk-Forward, and 24-oblast regional breakdown
python src/models/run_volunteer_experiment.py
python src/models/compare_official_vs_volunteer_metrics.py
python src/models/run_walk_forward_experiment.py
python src/models/analyze_walk_forward_regional.py

# [8/9] Execute all 5 Jupyter Notebooks from scratch (Restart & Run All)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_data_profiling_and_cleaning.ipynb \
    notebooks/02_rq1_temporal_cyclicity.ipynb \
    notebooks/03_rq2_regional_clustering.ipynb \
    notebooks/04_rq3_spatial_propagation_routes.ipynb \
    notebooks/05_ml_forecasting_experiment.ipynb

# [9/9] Run 6-tier causal invariance test and artifact verification across all 33 figures and tables
python tests/test_leakage_invariance.py
python tests/verify_pipeline.py
```

### 3. Research Jupyter Notebooks
* [`notebooks/01_data_profiling_and_cleaning.ipynb`](notebooks/01_data_profiling_and_cleaning.ipynb) — Raw data profiling for Official and Volunteer, duplicate block audit, deduplication, and interval merging.
* [`notebooks/02_rq1_temporal_cyclicity.ipynb`](notebooks/02_rq1_temporal_cyclicity.ipynb) — Weekly/diurnal rhythms, nocturnal alert shares, and Kruskal-Wallis tests across both sources.
* [`notebooks/03_rq2_regional_clustering.ipynb`](notebooks/03_rq2_regional_clustering.ipynb) — Descriptive hierarchical clustering (RQ2) vs. isolated Train-only ML clusters.
* [`notebooks/04_rq3_spatial_propagation_routes.ipynb`](notebooks/04_rq3_spatial_propagation_routes.ipynb) — Alert onset cross-correlations on a 2-minute discrete grid and lead-lag delays across 49 neighbor pairs.
* [`notebooks/05_ml_forecasting_experiment.ipynb`](notebooks/05_ml_forecasting_experiment.ipynb) — Static ML benchmark, covariate shift analysis, `naive` sensitivity, monthly Walk-Forward Retraining, hyperparameter alignment history, and 24-oblast regional breakdown.

---

## 📁 8. Repository Structure

Below is the complete, exact tree of all files tracked in the repository (one file per line, matching `git ls-files` 1-to-1):

```
├── .gitignore
├── LICENSE
├── README.md
├── README_EN.md
├── requirements.txt
├── run_all.sh
├── data/
│   ├── raw/
│   │   ├── official_data_en.csv
│   │   ├── official_data_uk.csv
│   │   └── volunteer_data_uk.csv
│   └── processed/
│       ├── alerts_clean_oblast.parquet
│       ├── ml_features_hourly.parquet
│       ├── volunteer_alerts_clean_as_is.parquet
│       ├── volunteer_alerts_clean_drop_naive.parquet
│       ├── volunteer_alerts_clean_fixed_d.parquet
│       ├── volunteer_ml_features_hourly_as_is.parquet
│       ├── volunteer_ml_features_hourly_drop_naive.parquet
│       ├── volunteer_ml_features_hourly_fixed_d.parquet
│       ├── walk_forward_oof_predictions_official.parquet
│       └── walk_forward_oof_predictions_volunteer_fixed_d.parquet
├── models/
│   ├── lightgbm_adaptive.joblib
│   ├── lightgbm_c_prime.joblib
│   ├── lightgbm_nospatial.joblib
│   ├── lightgbm_spatial.joblib
│   ├── logreg_adaptive.joblib
│   ├── logreg_c_prime.joblib
│   ├── logreg_nospatial.joblib
│   ├── logreg_spatial.joblib
│   ├── optimal_thresholds.json
│   ├── scaler_adaptive.joblib
│   ├── scaler_c_prime.joblib
│   ├── scaler_nospatial.joblib
│   ├── scaler_spatial.joblib
│   ├── volunteer_lightgbm_adaptive.joblib
│   ├── volunteer_lightgbm_adaptive_fixed_d.joblib
│   ├── volunteer_lightgbm_c_prime.joblib
│   ├── volunteer_lightgbm_nospatial.joblib
│   ├── volunteer_lightgbm_spatial.joblib
│   ├── volunteer_optimal_thresholds.json
│   └── volunteer_optimal_thresholds_fixed_d.json
├── notebooks/
│   ├── 01_data_profiling_and_cleaning.ipynb
│   ├── 02_rq1_temporal_cyclicity.ipynb
│   ├── 03_rq2_regional_clustering.ipynb
│   ├── 04_rq3_spatial_propagation_routes.ipynb
│   └── 05_ml_forecasting_experiment.ipynb
├── reports/
│   ├── official/
│   │   ├── figures/
│   │   │   ├── 01_heatmap_24x7.png
│   │   │   ├── 02_hourly_evolution_by_year.png
│   │   │   ├── 03_duration_distribution.png
│   │   │   ├── 04_clustering_metrics_evaluation.png
│   │   │   ├── 05_regional_dendrogram.png
│   │   │   ├── 06_spatial_cross_correlation_routes.png
│   │   │   ├── 07_baselines_performance.png
│   │   │   ├── 08_ml_pr_curves_validation.png
│   │   │   ├── 09_threshold_tuning_f2.png
│   │   │   ├── 10_test_pr_curves_comparison.png
│   │   │   ├── 11_feature_importance_lightgbm.png
│   │   │   ├── 12_lift_over_baselines.png
│   │   │   └── 13_calibration_reliability_diagram.png
│   │   ├── baselines_benchmark.csv
│   │   ├── clusters_train_only.csv
│   │   ├── feature_importance_lightgbm.csv
│   │   ├── feature_shift_summary.csv
│   │   ├── feature_shift_test_subperiods.csv
│   │   ├── ml_models_benchmark.csv
│   │   ├── ml_recent_60d_benchmark.csv
│   │   ├── ml_test_post_aug2025_benchmark.csv
│   │   ├── ml_test_pre_aug2025_benchmark.csv
│   │   ├── ml_validation_benchmark.csv
│   │   ├── regional_clusters_summary.csv
│   │   ├── spatial_lead_lag_summary.csv
│   │   ├── test_evaluation_details.json
│   │   ├── val_evaluation_details.json
│   │   ├── walk_forward_details.json
│   │   ├── walk_forward_monthly_metrics.csv
│   │   ├── walk_forward_overall_benchmark.csv
│   │   ├── walk_forward_recent_60d_benchmark.csv
│   │   └── walk_forward_regional_breakdown.csv
│   ├── volunteer/
│   │   ├── figures/
│   │   │   ├── 01_heatmap_24x7.png
│   │   │   ├── 02_hourly_evolution_by_year.png
│   │   │   ├── 03_duration_distribution.png
│   │   │   ├── 04_clustering_metrics_evaluation.png
│   │   │   ├── 05_regional_dendrogram.png
│   │   │   ├── 06_spatial_cross_correlation_routes.png
│   │   │   ├── 07_baselines_performance.png
│   │   │   ├── 08_ml_pr_curves_validation.png
│   │   │   ├── 09_threshold_tuning_f2.png
│   │   │   ├── 10_test_pr_curves_comparison.png
│   │   │   ├── 11_feature_importance_lightgbm.png
│   │   │   ├── 12_lift_over_baselines.png
│   │   │   └── 13_calibration_reliability_diagram.png
│   │   ├── baselines_benchmark.csv
│   │   ├── clusters_train_only.csv
│   │   ├── regional_clusters_summary.csv
│   │   ├── spatial_lead_lag_summary.csv
│   │   ├── volunteer_clusters_train_only.csv
│   │   ├── volunteer_clusters_train_only_drop_naive.csv
│   │   ├── volunteer_clusters_train_only_fixed_d.csv
│   │   ├── volunteer_evaluation_details.json
│   │   ├── volunteer_feature_importance_lightgbm.csv
│   │   ├── volunteer_feature_shift_summary.csv
│   │   ├── volunteer_feature_shift_test_subperiods.csv
│   │   ├── volunteer_ml_models_benchmark.csv
│   │   ├── volunteer_ml_recent_60d_benchmark.csv
│   │   ├── volunteer_ml_test_post_aug2025_benchmark.csv
│   │   ├── volunteer_ml_test_pre_aug2025_benchmark.csv
│   │   ├── volunteer_ml_validation_benchmark.csv
│   │   ├── volunteer_naive_sensitivity.csv
│   │   ├── volunteer_naive_sensitivity.json
│   │   ├── volunteer_train_median_durations.csv
│   │   ├── volunteer_walk_forward_details.json
│   │   ├── volunteer_walk_forward_monthly_metrics.csv
│   │   ├── volunteer_walk_forward_overall_benchmark.csv
│   │   ├── volunteer_walk_forward_recent_60d_benchmark.csv
│   │   └── volunteer_walk_forward_regional_breakdown.csv
│   └── comparison/
│       ├── figures/
│       │   ├── 14_comparison_pr_auc_official_vs_volunteer.png
│       │   ├── 15_comparison_f2_official_vs_volunteer.png
│       │   ├── 16_comparison_feature_shift_official_vs_volunteer.png
│       │   ├── 17_comparison_alert_counts_and_durations.png
│       │   ├── 18_volunteer_naive_sensitivity_comparison.png
│       │   ├── 19_walk_forward_vs_static_comparison.png
│       │   └── 20_walk_forward_regional_breakdown.png
│       ├── official_vs_volunteer_comparison.csv
│       ├── official_vs_volunteer_comparison.json
│       ├── official_vs_volunteer_feature_shift_comparison.csv
│       ├── official_vs_volunteer_ml_metrics_comparison.csv
│       ├── official_vs_volunteer_walk_forward_comparison.csv
│       ├── official_vs_volunteer_walk_forward_regional.csv
│       ├── official_vs_volunteer_walk_forward_regional_clusters.csv
│       └── upstream_source_verification.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── compare_official_vs_volunteer.py
│   │   └── verify_upstream_source.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── analyze_feature_shift.py
│   │   ├── clustering.py
│   │   ├── cyclicity.py
│   │   ├── feature_store.py
│   │   └── propagation.py
│   ├── models/
│   │   ├── analyze_walk_forward_regional.py
│   │   ├── baselines.py
│   │   ├── compare_official_vs_volunteer_metrics.py
│   │   ├── evaluate_test.py
│   │   ├── run_volunteer_experiment.py
│   │   ├── run_walk_forward_experiment.py
│   │   └── train_models.py
│   └── visualization/
│       └── __init__.py
└── tests/
    ├── test_leakage_invariance.py
    └── verify_pipeline.py
```

---

## 📄 License
Released under the [MIT License](LICENSE). Primary data source: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset).
