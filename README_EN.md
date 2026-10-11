# 🚨 Analysis & Short-Term Forecasting of Ukrainian Air Raid Sirens (2022–2026)

[ **🇺🇦 Українська версія** ](README.md) | [ **🇬🇧 English Version** ](README_EN.md)

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Dataset-299k%20Raw%20%7C%2084.9k%20Oblast-red.svg)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
[![Storage](https://img.shields.io/badge/Storage-Apache%20Parquet-green.svg)](https://parquet.apache.org/)
[![ML](https://img.shields.io/badge/Models-LightGBM%20%7C%20LogReg%20%7C%20Adaptive-orange.svg)](reports/ml_models_benchmark.csv)

A research project analyzing the spatial-temporal patterns of air raid sirens across Ukraine (March 2022 – September 2026) and forecasting the onset of new alerts during calm periods.

The project combines **exploratory time series analytics (Stage 1: RQ1–RQ3)** with **machine learning forecasting (Stage 2: predicting alert onsets in $[T, T+1\text{h})$)**.

> **Dataset Metadata & Verification:**  
> * **Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset (`datasets/official_data_uk.csv`)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (latest upstream commit: `58ee75bc20113181b9ddf8029d4cbf3a62d8cd10` dated `2026-09-08T03:44:23Z`).  
> * **Download Date:** `2026-10-11`  
> * **Raw File SHA-256 (`data/raw/official_data_uk.csv`):** `1f50085481d229f57b624d83b1ce723e884dea409c14afb5b7040c8a2f233e4f`  
> * **Raw Row Count:** `299,275` rows (`185,430` unique records after removing `113,845` exact duplicate rows in the raw CSV; pre-deduplication breakdown by level: `oblast` — 130,235, `raion` — 132,251, `hromada` — 36,789; post-deduplication: `oblast` — 65,314, `raion` — 93,662, `hromada` — 26,454).  
> * **Actual Timestamp Span in Raw CSV:** `min(started_at) = 2022-03-15 16:10:34 UTC`, `max(started_at) = 2026-09-07 05:28:22 UTC`.  
> * **Upstream Block Duplication Verification (`Vadimkin`):** Direct download and verification of `https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv` confirmed that the upstream file contains `299,275` rows (with the exact same SHA-256 hash as the local file) and that the duplicated block is present directly in the upstream repository:
>   * **Block 1 (rows `0..113,910`, `113,911` rows):** Spans `2022-03-15 16:10:34 UTC` to `2026-01-25 13:11:52 UTC` (`0` internal duplicates).
>   * **Block 2 (rows `113,911..227,831`, `113,921` rows):** Covers the exact same time span `2022-03-15 16:10:34 UTC` to `2026-01-25 13:11:52 UTC`, containing `113,845` exact row-for-row duplicates of Block 1 across all 6 columns (`oblast, raion, hromada, level, started_at, finished_at`) plus `76` rows that are not identical to Block 1 (`0` internal duplicates within Block 2).
>   * **Block 3 (rows `227,832..299,274`, `71,443` rows):** Records spanning `2026-01-25 13:34:47 UTC` to `2026-09-07 05:28:22 UTC` (`0` internal duplicates and `0` overlap with Blocks 1–2).
> * **Data Quality Audit & Row Waterfall in `src/data/cleaner.py`:**
>   1. Raw CSV input: **`299,275`** rows (`113,845` exact duplicates).
>   2. Exact deduplication (`drop_duplicates` on `[oblast, raion, hromada, level, started_at, finished_at]`): **`-113,845`** $\rightarrow$ **`185,430`** unique rows remain.
>   3. Micro-alert glitch filtering (`duration < 60s` or negative duration): **`-453`** $\rightarrow$ **`184,977`** valid records remain.
>   4. Pre-`2025-08-01` oblast filtering (removing **`-12,188`** localized `raion`/`hromada` rows prior to August 2025, when oblast-wide alerts were comprehensively logged as `level == 'oblast'`) $\rightarrow$ **`172,789`** records (`64,875` pre-2025-08-01 + `107,914` post-2025-08-01).
>   5. Oblast-level interval reconstruction and merging of overlapping/adjacent `[started_at, finished_at]` intervals within each oblast: consolidates **`-87,881`** nested/overlapping sub-regional records $\rightarrow$ final cleaned dataset `data/processed/alerts_clean_oblast.parquet` contains **`84,908`** continuous oblast-level intervals (`64,870` pre-2025-08-01 and `20,038` post-2025-08-01; `84,907` across the 24 regular regions plus `1` in Luhansk Oblast). Zero post-cleaning duplicates are enforced via `assert == 0`.
> * **Why Metrics Changed Between the Initial and Current Versions:** In the first iteration of the pipeline, filtering `level == 'oblast'` without deduplicating the doubled upstream block (`0..227,831`) and without aggregating post-August 2025 district (`raion`/`hromada`) alerts produced `129,955` rows (`64,921` of which were exact duplicates from 2022–Jan 2026). That **artificially doubled absolute alert counts** in RQ1 and RQ2 (e.g., median daily alerts appeared as ~92–94 instead of the true 52–53, and monthly regional cluster frequencies were ~2x higher), while omitting post-August 2025 `raion` alerts created an artificial drop in the Test base rate (`2.35%` overall and `0.41%` in the last 60 days). After removing the `113,845` upstream duplicates and reconstructing oblast intervals from `raion`/`hromada`, all metrics reflect the true underlying process (Test base rate **$10.27\%$**, latest 60 days **$12.65\%$**).

---

## 📑 Table of Contents
1. [Key Findings from Exploratory Analysis (RQ1–RQ3)](#-key-findings-from-exploratory-analysis-rq1rq3)
2. [Machine Learning: Alert Onset Forecasting](#-machine-learning-alert-onset-forecasting)
3. [Result Interpretation, Covariate Shift & Post-Hoc Analysis](#-result-interpretation-covariate-shift--post-hoc-analysis)
4. [Ablation Study: Comparing Feature Sets A, B, C, and C'](#-ablation-study-comparing-feature-sets-a-b-c-and-c)
5. [Causal Invariance Verification & Methodological Limitations](#-causal-invariance-verification--methodological-limitations)
6. [Quickstart Guide](#-quickstart-guide)
7. [Repository Structure](#-repository-structure)

---

## 🔬 Key Findings from Exploratory Analysis (RQ1–RQ3)

### 1. Temporal Cyclicity (RQ1)
* **Weekly Pattern (Kruskal-Wallis Test):** $H = 25.52, \ p = 2.74 \times 10^{-4}$ (3 statistically significant day pairs after Bonferroni correction $\alpha_{\text{adj}} = 0.00238$). Peak activity occurs Wednesday–Friday (median 52–53 merged oblast alerts/day, mean 53.4–55.0), reaching a minimum on Sunday (median 47, mean 46.8 alerts/day).
* **Diurnal Profile & Nocturnal Alerts (00:00–05:59 Europe/Kyiv):** Once post-August 2025 district-level alerts are properly aggregated into oblast intervals, the share of alerts starting at night remains stable between **$20.2\%$ and $24.0\%$** across years (2022: $24.0\%$, 2023: $21.5\%$, 2024: $20.2\%$, 2025: $20.5\%$, 2026: $22.9\%$).
* **Alert Duration:** Median duration of a merged oblast interval is **43.15 min**, while the mean is **102.40 min** ($Q_{25} = 26.38$ min, $Q_{75} = 100.66$ min, $Q_{95} = 329.06$ min).

<p align="center">
  <img src="reports/figures/01_heatmap_24x7.png" alt="24x7 Alert Activity Heatmap" width="85%">
  <br><em>Figure 1. Heatmap of alert onset counts by hour of day (Europe/Kyiv) and day of week (2022–2026).</em>
</p>

<p align="center">
  <img src="reports/figures/02_hourly_evolution_by_year.png" alt="Evolution of hourly distribution by year" width="85%">
  <br><em>Figure 2. Diurnal profile of alert onsets by year (percentage share of annual alerts).</em>
</p>

---

### 2. Regional Clustering: Descriptive (RQ2) vs. ML (Train-Only)
The repository strictly separates **descriptive full-period clustering** (used for RQ2) from **Train-only clustering** (used as input features for ML models):

1. **Descriptive RQ2 Clustering (`reports/regional_clusters_summary.csv`, full period 2022–2026, Ward linkage, Silhouette = 0.4417 at $K=3$):**
   * **Cluster 1 — Deep Rear / West & North-West (12 regions):** Ivano-Frankivsk, Volyn, Vinnytsia, Zhytomyr, Zakarpattia, Kyiv Oblast, Lviv, Rivne, Ternopil, Khmelnytskyi, Chernivtsi, Kyiv City (avg 25.8 alerts/month, median duration 46.4 min).
   * **Cluster 2 — Transit Center & South (7 regions):** Dnipropetrovsk, Kirovohrad, Mykolaiv, Odesa, Poltava, Kherson, Cherkasy (avg 90.3 alerts/month, median duration 36.0 min).
   * **Cluster 3 — Frontline / Border Zone (5 regions):** Donetsk, Zaporizhzhia, Sumy, Kharkiv, Chernihiv (avg 128.3 alerts/month, median duration 57.3 min, share of alerts $\ge 2$h: 28.7%).
2. **ML Train-Only Clusters (`reports/clusters_train_only.csv`, fitted strictly on Train `<= 2024-04-30 23:59:59 UTC` via `generate_train_only_clusters()` in `src/features/clustering.py` with neutral names):**
   * `cluster_1_low_activity` (`low_activity`, 7 western oblasts): Ivano-Frankivsk, Volyn, Zakarpattia, Lviv, Rivne, Ternopil, Chernivtsi (17.4–22.1 alerts/month).
   * `cluster_2_medium_activity` (`medium_activity`, 5 central/north-western regions): Vinnytsia, Zhytomyr, Kyiv Oblast, Khmelnytskyi, Kyiv City (24.2–38.2 alerts/month).
   * `cluster_3_high_activity` (`high_activity`, 12 eastern, southern, and central oblasts): Dnipropetrovsk, Donetsk, Zaporizhzhia, Kirovohrad, Mykolaiv, Odesa, Poltava, Sumy, Kharkiv, Kherson, Cherkasy, Chernihiv (44.0–168.8 alerts/month).

<p align="center">
  <img src="reports/figures/05_regional_dendrogram.png" alt="Regional Dendrogram" width="85%">
  <br><em>Figure 3. Ward hierarchical dendrogram of the 24 regions of Ukraine (RQ2, full period).</em>
</p>

---

### 3. Spatial Cross-Correlations Between Neighbors (RQ3)
* After filtering out synchronized nationwide alerts ($\ge 15$ oblasts simultaneously) on a 10-minute discrete grid, cross-correlations with lags $\tau \in [-180, +180]$ min were computed for all 49 contiguous region pairs (`reports/spatial_lead_lag_summary.csv`).
* Strongest couplings occur along southern, eastern, and central corridors: Mykolaiv $\leftrightarrow$ Kherson ($r = 0.658$, lag 10 min), Dnipropetrovsk $\leftrightarrow$ Zaporizhzhia ($r = 0.613$), Ivano-Frankivsk $\leftrightarrow$ Lviv ($r = 0.602$, lag 10 min), Kyiv City $\leftrightarrow$ Kyiv Oblast ($r = 0.578$, lag 10 min), Chernihiv $\leftrightarrow$ Sumy ($r = 0.574$, lag 10 min).

<p align="center">
  <img src="reports/figures/06_spatial_cross_correlation_routes.png" alt="Spatial Cross-Correlation Routes" width="85%">
  <br><em>Figure 4. Top-15 adjacent region pairs by cross-correlation strength and directional lead-lag breakdown.</em>
</p>

---

## 🤖 Machine Learning: Alert Onset Forecasting

### Problem Formulation & Feature Sets (A, B, C, and Post-Hoc C')
* **Decision Timestamp $T$:** Regular hourly grid in UTC (`timestamp_utc`).
* **Calm-Period Conditioning:** `ml_features_hourly.parquet` retains strictly those observations where no alert is active in the region at moment $T$ (`is_active_T == 0`), totaling `797,428` calm region-hours across 24 regions (`0` duplicates on `(timestamp_utc, oblast)`).
* **Target Variable (`target_onset_next_1h`):** $y_T = 1$ if a new alert starts in the region during the half-open interval **$[T, T + 1\text{h})$** (implemented as `k_onset = int(s_sec // 3600)`).
* **Feature Set Hierarchy (all computed strictly from data prior to $T$, `closed="left"`, on the full hourly grid before calm filtering):**
  * **Set A (Time, Burden & Train Cluster — 10 features):** `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `month` (`Europe/Kyiv`), `burden_last_6h`, `burden_last_24h` (fraction of active hours in $[T-6\text{h}, T)$ and $[T-24\text{h}, T)$), and Train-only cluster indicators (`cluster_1_low_activity`, `cluster_2_medium_activity`, `cluster_3_high_activity`).
  * **Set B (A + Spatial at $T$ — 13 features):** Set A + instantaneous status of contiguous neighbors at exact moment $T$: `neighbors_active_count`, `neighbor_any_active`, `neighbor_active_ratio`.
  * **Set C (B + Adaptive & Spatio-Temporal Memory — 22 features, Pre-Specified Primary Configuration):** Set B + 9 historical adaptive and spatio-temporal lag features:
    * `adaptive_rate_7d`, `adaptive_rate_14d`, `adaptive_rate_30d` — rolling onset frequencies in the region over the preceding 168, 336, and 720 hours (`closed="left"`).
    * `national_active_count` — total number of oblasts across Ukraine with an active alert at moment $T$.
    * `onsets_last_24h` (Variant A) — rolling sum of hourly onsets in the region over $[T - 24\text{h}, T)$ (`closed="left"`).
    * `hours_since_last_alert` — hours elapsed from the latest known `finished_at <= T` in the region to moment $T$ (`clip(upper=720.0)`).
    * `neighbors_active_lag_1h`, `neighbors_active_lag_2h`, `neighbors_active_lag_3h` — active neighbor counts at $T-1\text{h}$, $T-2\text{h}$, $T-3\text{h}$ on the full grid.
  * **Set C' (`FEATURE_SET_C_PRIME`: B + Adaptive & Neighbor Lags Only — 19 features, Post-Hoc Experiment):** Set B + `adaptive_rate_7d/14d/30d` + `neighbors_active_lag_1h/2h/3h` (**excluding** `national_active_count`, `hours_since_last_alert`, and `onsets_last_24h`, which exhibited strong covariate shift between Train and Test).

> **⚠️ Post-Hoc Disclosure:**  
> The primary comparison was pre-specified as **Set C vs. the adaptive baseline (`Adaptive 14-day Rolling`)**; selecting **Set B** as the best model by PR-AUC (as well as constructing and evaluating **Set C'**) was done **after inspecting Test results and is post-hoc**. All hyperparameters and classification thresholds $p^* = \arg\max F_2$ across all models (including `Set C'`) were tuned exclusively on Validation without touching Test.

### Chronological Data Splits (UTC, with 2-hour Embargo Buffer)
* **Train:** `2022-03-15` – `2024-04-30` (`410,150` calm hours, `33,240` onsets, base rate **$8.10\%$**).
* **Validation:** `2024-05-01` – `2025-05-31` (`190,195` calm hours, `19,342` onsets, base rate **$10.17\%$**). All hyperparameters, early stopping, and classification thresholds $p^* = \arg\max F_2$ are tuned strictly on Validation.
* **Test (Full):** `2025-06-01` – `2026-09-07` (`197,083` calm hours, `464` calendar days, `20,250` onsets, base rate **$10.27\%$**):
  * **Test Pre-`2025-08-01` (`2025-06-01` – `2025-07-31`):** `28,929` calm hours, `62` days, `2,782` onsets (base rate **$9.62\%$**).
  * **Test Post-`2025-08-01` (`2025-08-01` – `2026-09-07`):** `168,154` calm hours, `403` days, `17,468` onsets (base rate **$10.39\%$**).
* **Latest 60 Days (Test subset):** `2026-07-09` – `2026-09-07` (`23,646` calm hours, `2,992` onsets, base rate **$12.65\%$**).

---

### Table 1. Full Test Set Benchmark (Test: June 2025 – September 2026, `reports/ml_models_benchmark.csv`)
*95% confidence intervals are computed via **Day-Block Bootstrap** resampling calendar days (`n_bootstraps = 500`). `Prec @ R=78.2%` reports each model's Precision at the exact Recall level of the Adaptive Baseline (`Recall = 78.23%`).*

| Model | Feature Set | Threshold $p^*$ (from Val) | Test PR-AUC [95% CI] | Test $F_2$ [95% CI] | Precision | Recall | Prec @ R=78.2% | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | Lagged onset in $[T-1\text{h}, T)$ | 0.500 | — | 0.1034 [0.096, 0.111] | 29.1% | 8.9% | — | 0.1159 |
| **Baseline 2: Historical Lookup** | Train lookup rate (Kyiv) | 0.085 | 0.2204 [0.212, 0.228] | 0.5033 [0.495, 0.511] | 22.7% | 72.3% | 21.1% | 0.0868 |
| **Baseline 3: Adaptive 14-day** | 14-day rolling onset rate | 0.089 | 0.3031 [0.292, 0.315] | 0.5299 [0.520, 0.538] | 23.1% | 78.2% | 23.1% | 0.0818 |
| **LogReg (Set A: No Spatial)** | A (10 features: Time + Burden + Cluster) | 0.518 | 0.3881 [0.377, 0.399] | 0.5329 [0.524, 0.541] | 23.5% | 77.9% | 23.4% | 0.2007 |
| **LogReg (Set B: With Spatial)** | B (13 features: A + Neighbors at $T$) | 0.561 | **0.3903** [0.380, 0.400] | 0.5499 [0.541, 0.558] | 24.0% | 81.2% | 25.1% | 0.2337 |
| **LogReg (Set C: Spatial + Adaptive)** | C (22 features: B + Adaptive, Nat & Lags) | 0.565 | 0.3173 [0.306, 0.328] | 0.5431 [0.535, 0.550] | 21.1% | **89.3%** | 25.3% | 0.2875 |
| **LogReg (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features: B + Adaptive & Lags) | 0.541 | 0.3386 [0.327, 0.350] | 0.5527 [0.544, 0.561] | 24.1% | 81.6% | 25.6% | 0.2265 |
| **LightGBM (Set A: No Spatial)** | A (10 features: Time + Burden + Cluster) | 0.093 | 0.3482 [0.337, 0.359] | 0.5379 [0.529, 0.546] | 25.4% | 74.6% | 23.5% | 0.0784 |
| **LightGBM (Set B: With Spatial)** *(post-hoc)* | B (13 features: A + Neighbors at $T$) | 0.109 | 0.3741 [0.363, 0.386] | 0.5610 [0.552, 0.570] | 23.5% | 85.8% | 26.8% | **0.0761** |
| **LightGBM (Set C: Spatial + Adaptive)** | C (22 features: B + Adaptive, Nat & Lags) | 0.129 | 0.3329 [0.323, 0.343] | 0.5551 [0.547, 0.562] | 23.0% | 85.8% | 26.3% | 0.0783 |
| **LightGBM (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features: B + Adaptive & Lags) | 0.129 | 0.3522 [0.342, 0.363] | **0.5683** [0.559, 0.577] | **25.7%** | 81.6% | **27.1%** | 0.0767 |

---

### Table 2. Test Sub-Period Benchmark Before and After `2025-08-01` (`reports/ml_test_pre_aug2025_benchmark.csv`, `reports/ml_test_post_aug2025_benchmark.csv`)
*Evaluated at the same frozen Validation thresholds $p^*$ separately on the direct `level == 'oblast'` logging sub-period (**Pre-2025-08-01**, 28,929 rows, base rate 9.62%, Adaptive Baseline `R = 68.58%`) and the district/community union sub-period (**Post-2025-08-01**, 168,154 rows, base rate 10.39%, Adaptive Baseline `R = 79.76%`).*

| Model | Pre-01.08.2025: PR-AUC [95% CI] | Pre-01.08.2025: $F_2$ [95% CI] | Pre-01.08: Prec / Rec (Prec@R=68.6%) | Post-01.08.2025: PR-AUC [95% CI] | Post-01.08.2025: $F_2$ [95% CI] | Post-01.08: Prec / Rec (Prec@R=79.8%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3: Adaptive 14-day** | 0.2735 [0.259, 0.288] | 0.4605 [0.444, 0.477] | 19.9% / 68.6% (17.4%) | 0.3068 [0.294, 0.320] | 0.5411 [0.531, 0.551] | 23.7% / 79.8% (23.9%) |
| **LogReg (Set A)** | 0.2934 [0.275, 0.313] | 0.4617 [0.443, 0.479] | 18.1% / 75.6% (17.3%) | 0.4025 [0.392, 0.413] | 0.5458 [0.537, 0.555] | 24.7% / 78.2% (24.7%) |
| **LogReg (Set B)** | 0.2877 [0.272, 0.306] | 0.4814 [0.466, 0.498] | 19.9% / 74.6% (19.1%) | **0.4040** [0.395, 0.415] | 0.5614 [0.553, 0.570] | 24.7% / 82.3% (26.2%) |
| **LogReg (Set C)** | 0.2579 [0.243, 0.275] | 0.4979 [0.483, 0.513] | 20.5% / 77.4% (20.3%) | 0.3266 [0.314, 0.339] | 0.5498 [0.541, 0.559] | 21.2% / **91.2%** (26.6%) |
| **LogReg (Set C')** *(post-hoc)* | 0.2783 [0.264, 0.294] | 0.4813 [0.467, 0.495] | 21.5% / 69.6% (19.2%) | 0.3470 [0.334, 0.359] | 0.5638 [0.554, 0.574] | 24.5% / 83.5% (26.7%) |
| **LightGBM (Set A)** | 0.2892 [0.270, 0.310] | 0.4712 [0.455, 0.488] | 19.8% / 71.8% (18.1%) | 0.3577 [0.348, 0.368] | 0.5497 [0.541, 0.560] | **26.6%** / 75.0% (24.7%) |
| **LightGBM (Set B)** *(post-hoc)* | 0.3105 [0.293, 0.328] | 0.5071 [0.492, 0.522] | 21.0% / 78.4% (21.0%) | 0.3837 [0.373, 0.394] | 0.5696 [0.561, 0.579] | 23.9% / 87.0% (27.8%) |
| **LightGBM (Set C)** | 0.3006 [0.284, 0.321] | 0.5044 [0.487, 0.522] | 23.3% / 71.2% (21.0%) | 0.3386 [0.327, 0.350] | 0.5623 [0.553, 0.571] | 23.0% / 88.1% (27.1%) |
| **LightGBM (Set C')** *(post-hoc)* | **0.3124** [0.295, 0.330] | **0.5106** [0.493, 0.530] | **25.1%** / 68.9% (**21.0%**) | 0.3582 [0.346, 0.370] | **0.5769** [0.568, 0.586] | 25.7% / 83.7% (**27.8%**) |

---

### Table 3. Latest 60 Days Benchmark (09.07.2026 – 07.09.2026, `reports/ml_recent_60d_benchmark.csv`)
*Subset: 23,646 calm hours, 2,992 alert onsets (base rate 12.65%). Thresholds $p^*$ are frozen from Validation. `Prec @ R=90.4%` reports Precision at the exact Recall level of the Adaptive Baseline (`Recall = 90.37%`).*

| Model | Feature Set | Threshold $p^*$ | PR-AUC [95% CI] | $F_2$-Score [95% CI] | Precision | Recall | Prec @ R=90.4% | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | Lag $[T-1\text{h}, T)$ | 0.500 | — | 0.1621 [0.145, 0.181] | 39.2% | 14.1% | — | 0.1364 |
| **Baseline 2: Historical Lookup** | Train lookup (Kyiv) | 0.085 | 0.3257 [0.295, 0.357] | 0.5886 [0.564, 0.613] | 33.1% | 73.0% | 31.7% | 0.1038 |
| **Baseline 3: Adaptive 14-day** | 14-day rolling rate | 0.089 | 0.3682 [0.333, 0.406] | 0.6430 [0.620, 0.664] | 29.9% | 90.4% | 34.1% | 0.0911 |
| **LogReg (Set A: No Spatial)** | A (10 features) | 0.518 | **0.5058** [0.480, 0.531] | 0.6535 [0.636, 0.674] | 35.5% | 82.8% | **38.5%** | 0.1693 |
| **LogReg (Set B: With Spatial)** | B (13 features) | 0.561 | 0.4788 [0.444, 0.510] | 0.6501 [0.626, 0.674] | 32.7% | 86.3% | 37.6% | 0.2064 |
| **LogReg (Set C: Spatial + Adaptive)** | C (22 features) | 0.565 | 0.3889 [0.354, 0.423] | 0.6196 [0.592, 0.643] | 25.8% | **95.5%** | 35.7% | 0.2974 |
| **LogReg (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features) | 0.541 | 0.3803 [0.343, 0.417] | 0.6494 [0.622, 0.675] | 30.9% | 89.6% | 36.6% | 0.2304 |
| **LightGBM (Set A: No Spatial)** | A (10 features) | 0.093 | 0.4501 [0.423, 0.477] | **0.6559** [0.639, 0.674] | **37.0%** | 81.3% | 38.4% | 0.0885 |
| **LightGBM (Set B: With Spatial)** *(post-hoc)* | B (13 features) | 0.109 | 0.4541 [0.424, 0.481] | 0.6442 [0.617, 0.670] | 29.9% | 90.5% | 37.1% | **0.0853** |
| **LightGBM (Set C: Spatial + Adaptive)** | C (22 features) | 0.129 | 0.3809 [0.348, 0.416] | 0.6301 [0.603, 0.654] | 27.3% | 93.5% | 34.4% | 0.0895 |
| **LightGBM (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features) | 0.129 | 0.3999 [0.367, 0.437] | 0.6527 [0.628, 0.677] | 31.2% | 89.9% | 36.0% | 0.0868 |

<p align="center">
  <img src="reports/figures/10_test_pr_curves_comparison.png" alt="Test and Recent 60d PR Curves Comparison" width="95%">
  <br><em>Figure 5. Precision-Recall curves across Feature Sets A, B, C, C' and baselines on the full Test set (left) and the latest 60 days (right).</em>
</p>

---

## 💡 Result Interpretation, Covariate Shift & Post-Hoc Analysis

1. **Pre-Specified Primary Comparison (`LightGBM Set C` vs. `Adaptive 14-day Rolling`):**
   * **On the Full Test Set (`2025-06` – `2026-09`):** `LightGBM (Set C)` statistically significantly outperforms `Adaptive 14-day Baseline`:
     * **$\Delta \text{PR-AUC} = +0.0297$**, 95% CI **$[+0.0220, +0.0373]$** (`0.3329` vs. `0.3031`, **$+9.8\%$** relative lift).
     * **$\Delta F_2 = +0.0249$**, 95% CI **$[+0.0189, +0.0304]$** (`0.5551` vs. `0.5299`, **$+4.7\%$** relative lift).
     * **Precision at equal `Recall = 78.23%`:** **26.25%** for `LightGBM (Set C)` vs. **23.14%** for the Adaptive Baseline ($+3.11$ percentage points).
   * **On the Latest 60 Days (`2026-07-09` – `2026-09-07`):** `LightGBM (Set C)` **does not outperform** the `Adaptive 14-day Baseline`:
     * In **PR-AUC**, the paired difference is **$\Delta = +0.0126$**, 95% CI **$[-0.0055, +0.0314]$** (crosses zero — not statistically significant).
     * In **$F_2$-Score** at the frozen Validation threshold $p^* = 0.129$, `Set C` trails the Adaptive Baseline: **$\Delta F_2 = -0.0133$**, 95% CI **$[-0.0223, -0.0048]$** (`0.6301` vs. `0.6430`).

2. **Verified Feature Distribution Shift Across Train, Val, and Test (`src/features/analyze_feature_shift.py`):**
   The script `src/features/analyze_feature_shift.py` computes feature means directly from `data/processed/ml_features_hourly.parquet` (`reports/feature_shift_summary.csv` and `reports/feature_shift_test_subperiods.csv`):

   | Feature | Train Mean (`2022-03` – `2024-04`) | Val Mean (`2024-05` – `2025-05`) | Test Mean (`2025-06` – `2026-09`) | Test Pre-`2025-08-01` (`28,929` hrs) | Test Post-`2025-08-01` (`168,154` hrs) | Test / Train Ratio |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
   | **`national_active_count`** | **1.17** (`1.1725`) | **3.06** (`3.0615`) | **5.55** (`5.5450`) | **3.17** (`3.1672`) | **5.95** (`5.9541`) | **$4.73\times$** |
   | **`neighbors_active_count`** | **0.13** (`0.1279`) | **0.32** (`0.3188`) | **0.52** (`0.5169`) | **0.34** (`0.3382`) | **0.55** (`0.5477`) | **$4.04\times$** |
   | **`hours_since_last_alert`** | **32.6** (`32.6421`) | **30.1** (`30.1340`) | **42.7** (`42.6688`) | **22.8** (`22.7793`) | **46.1** (`46.0906`) | **$1.31\times$** |
   | **`adaptive_rate_14d`** | **0.075** (`0.0751`) | **0.083** (`0.0825`) | **0.078** (`0.0777`) | **0.073** (`0.0733`) | **0.079** (`0.0785`) | **$1.04\times$** |

   * **Likely Cause of `Set C` Degradation Relative to `Set B`:** The likely cause of `Set C` underperforming relative to `Set B` on Test is the **strong shift in feature distributions between Train and Test** (specifically, `national_active_count` rising from `1.17` on Train to `5.55` on Test and jumping within Test from `3.17` to `5.95` after `2025-08-01`, alongside `hours_since_last_alert` shifting from `22.8` to `46.1` hours) — partly real and partly possibly due to the change in alert definition/logging in August 2025 (where an oblast is treated as active whenever any district or community within it has an active alert).

3. **Post-Hoc Experiment with `Set C'` (`Set B + adaptive_rate_7d/14d/30d + neighbors_active_lag_1h/2h/3h`):**
   Removing the three strongly shifted features (`national_active_count`, `hours_since_last_alert`, `onsets_last_24h`) in `Set C'` (19 features) yields:
   * **Compared to `Set C`:** `LightGBM (Set C')` statistically significantly improves both PR-AUC ($\Delta_{\text{C'}-\text{C}} = +0.0194 \ [+0.0143, +0.0247]$ on Full Test) and $F_2$-Score ($\Delta_{\text{C'}-\text{C}} = +0.0134 \ [+0.0102, +0.0167]$).
   * **Compared to `Adaptive 14-day Baseline`:** `LightGBM (Set C')` significantly outperforms the adaptive baseline on **Full Test** ($\Delta \text{PR-AUC} = +0.0492 \ [+0.0428, +0.0557]$, $\Delta F_2 = +0.0384 \ [+0.0331, +0.0435]$), **Pre-`2025-08-01`** ($\Delta \text{PR-AUC} = +0.0389 \ [+0.0265, +0.0510]$, $\Delta F_2 = +0.0502 \ [+0.0323, +0.0700]$), **Post-`2025-08-01`** ($\Delta \text{PR-AUC} = +0.0514 \ [+0.0445, +0.0587]$, $\Delta F_2 = +0.0358 \ [+0.0309, +0.0406]$), and on the **Latest 60 Days** in PR-AUC ($\Delta \text{PR-AUC} = +0.0314 \ [+0.0111, +0.0511]$, $\Delta F_2 = +0.0094 \ [-0.0001, +0.0189]$).
   * **Compared to `Set B`:**
     * **Before `2025-08-01` (prior to the logging transition):** `LightGBM (Set C')` achieves the highest performance across all models on both PR-AUC (`0.3124` vs. `0.3105` for `Set B`, $\Delta_{\text{C'}-\text{B}} = +0.0019 \ [-0.0076, +0.0128]$) and $F_2$ (`0.5106` vs. `0.5071` for `Set B`, $\Delta_{\text{C'}-\text{B}} = +0.0035 \ [-0.0078, +0.0134]$). Even full `LightGBM (Set C)` showed no statistically significant drop vs. `Set B` prior to August 2025 ($\Delta_{\text{C}-\text{B}} \text{PR-AUC} = -0.0096 \ [-0.0234, +0.0061]$).
     * **On Full Test and Post-`2025-08-01`:** `LightGBM (Set C')` achieves the **highest $F_2$-Score of all models** (`0.5683` on Full Test, $\Delta_{\text{C'}-\text{B}} F_2 = +0.0075 \ [+0.0043, +0.0107]$; `0.5769` Post-`2025-08-01`, $\Delta_{\text{C'}-\text{B}} F_2 = +0.0074 \ [+0.0039, +0.0105]$; `0.6527` on Latest 60 Days, $\Delta_{\text{C'}-\text{B}} F_2 = +0.0086 \ [+0.0012, +0.0161]$) and the **highest Precision at equal baseline Recall (`Prec @ R=78.2% = 27.06%`)**, while trailing compact `LightGBM (Set B)` on global PR-AUC ($\Delta_{\text{C'}-\text{B}} \text{PR-AUC} = -0.0219 \ [-0.0290, -0.0146]$ on Full Test and $-0.0254 \ [-0.0343, -0.0170]$ Post-`2025-08-01`).

4. **Predicted Probability Quantiles & Calibration:**  
   * On Test, empirical quantiles of `LightGBM (Set C)` predicted probabilities are: $Q_{10} = 1.63\%$, $Q_{25} = 3.01\%$, Median ($Q_{50}$) $= 7.88\%$, $Q_{75} = 19.81\%$, $Q_{90} = 30.39\%$, $Q_{95} = 36.37\%$, $Q_{99} = 46.96\%$.
   * **LightGBM (Set B, Set C', and Set C)** maintains strong probability calibration (**Brier Score = 0.0761**, **0.0767**, and **0.0783**), whereas Logistic Regression with `class_weight='balanced'` yields uncalibrated probabilities (`Brier = 0.2007–0.2875`).

<p align="center">
  <img src="reports/figures/13_calibration_reliability_diagram.png" alt="Calibration Reliability Diagram" width="75%">
  <br><em>Figure 6. Reliability diagram for LightGBM (Set B and Set C) on the Test set.</em>
</p>

---

## 🛰️ Ablation Study: Comparing Feature Sets A, B, C, and C'

Paired metric differences across the 4 Test slices (500-iteration Day-Block Bootstrap across calendar days, `reports/test_evaluation_details.json`):

| Comparison (Paired Difference) | Full Test: $\Delta$ PR-AUC / $\Delta F_2$ | Pre-01.08.2025: $\Delta$ PR-AUC / $\Delta F_2$ | Post-01.08.2025: $\Delta$ PR-AUC / $\Delta F_2$ | Latest 60 Days: $\Delta$ PR-AUC / $\Delta F_2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Primary (pre-specified):** `LGBM (C) - Adaptive 14d` | **$+0.0297$** $[+0.022, +0.037]$<br>**$+0.0249$** $[+0.019, +0.030]$ | **$+0.0274$** $[+0.012, +0.046]$<br>**$+0.0441$** $[+0.027, +0.062]$ | **$+0.0319$** $[+0.024, +0.040]$<br>**$+0.0212$** $[+0.016, +0.027]$ | $+0.0126$ $[-0.006, +0.031]$<br>**$-0.0133$** $[-0.022, -0.005]$ |
| **Post-hoc:** `LGBM (C') - Adaptive 14d` | **$+0.0492$** $[+0.043, +0.056]$<br>**$+0.0384$** $[+0.033, +0.043]$ | **$+0.0389$** $[+0.027, +0.051]$<br>**$+0.0502$** $[+0.032, +0.070]$ | **$+0.0514$** $[+0.045, +0.059]$<br>**$+0.0358$** $[+0.031, +0.041]$ | **$+0.0314$** $[+0.011, +0.051]$<br>$+0.0094$ $[-0.0001, +0.019]$ |
| **Post-hoc:** `LGBM (C') - LGBM (B)` | **$-0.0219$** $[-0.029, -0.015]$<br>**$+0.0075$** $[+0.004, +0.011]$ | $+0.0019$ $[-0.008, +0.013]$<br>$+0.0035$ $[-0.008, +0.013]$ | **$-0.0254$** $[-0.034, -0.017]$<br>**$+0.0074$** $[+0.004, +0.010]$ | **$-0.0533$** $[-0.074, -0.033]$<br>**$+0.0086$** $[+0.001, +0.016]$ |
| **Post-hoc:** `LGBM (C') - LGBM (C)` | **$+0.0194$** $[+0.014, +0.025]$<br>**$+0.0134$** $[+0.010, +0.017]$ | $+0.0115$ $[-0.001, +0.024]$<br>$+0.0061$ $[-0.001, +0.013]$ | **$+0.0195$** $[+0.013, +0.025]$<br>**$+0.0146$** $[+0.011, +0.018]$ | **$+0.0189$** $[+0.008, +0.030]$<br>**$+0.0228$** $[+0.016, +0.030]$ |
| **LGBM:** `Set B - Set A` (neighbors at $T$) | **$+0.0259$** $[+0.023, +0.029]$<br>**$+0.0230$** $[+0.017, +0.029]$ | **$+0.0212$** $[+0.013, +0.030]$<br>**$+0.0364$** $[+0.024, +0.049]$ | **$+0.0260$** $[+0.022, +0.029]$<br>**$+0.0198$** $[+0.014, +0.026]$ | $+0.0039$ $[-0.006, +0.014]$<br>$-0.0126$ $[-0.035, +0.007]$ |
| **LGBM:** `Set C - Set B` | **$-0.0414$** $[-0.050, -0.033]$<br>**$-0.0060$** $[-0.010, -0.003]$ | $-0.0096$ $[-0.023, +0.006]$<br>$-0.0026$ $[-0.014, +0.008]$ | **$-0.0448$** $[-0.054, -0.035]$<br>**$-0.0072$** $[-0.011, -0.004]$ | **$-0.0722$** $[-0.095, -0.050]$<br>**$-0.0142$** $[-0.021, -0.007]$ |
| **LogReg (C') - Adaptive 14d** *(post-hoc)* | **$+0.0355$** $[+0.030, +0.041]$<br>**$+0.0228$** $[+0.019, +0.026]$ | $+0.0050$ $[-0.006, +0.017]$<br>**$+0.0208$** $[+0.010, +0.032]$ | **$+0.0402$** $[+0.034, +0.047]$<br>**$+0.0227$** $[+0.019, +0.026]$ | $+0.0122$ $[-0.009, +0.033]$<br>$+0.0064$ $[-0.003, +0.015]$ |
| **LogReg (C') - LogReg (B)** *(post-hoc)* | **$-0.0518$** $[-0.060, -0.044]$<br>**$+0.0029$** $[+0.0004, +0.006]$ | $-0.0094$ $[-0.019, +0.0005]$<br>$+0.0002$ $[-0.008, +0.008]$ | **$-0.0571$** $[-0.067, -0.049]$<br>$+0.0024$ $[-0.0001, +0.005]$ | **$-0.0982$** $[-0.125, -0.076]$<br>$-0.0008$ $[-0.008, +0.006]$ |

<p align="center">
  <img src="reports/figures/11_feature_importance_lightgbm.png" alt="LightGBM Set C Feature Importance" width="85%">
  <br><em>Figure 7. Feature importance of the 22 features in LightGBM (Set C) ranked by Total Gain.</em>
</p>

---

## 🛡️ Causal Invariance Verification & Methodological Limitations

### What `tests/test_leakage_invariance.py` Verifies
The test first verifies **zero duplicates (`assert 0`)** in both `alerts_clean_oblast.parquet` and `ml_features_hourly.parquet`, validates the exact column definitions of all four feature sets (`Set A` — 10 features, `Set B` — 13 features, `Set C` — 22 features, `Set C'` — 19 features where `C' = C \ {national_active_count, hours_since_last_alert, onsets_last_24h}`), and then evaluates **$119$ timestamps $T$** ($19$ explicit boundary cases — grid start $k=0,1,2,3,6$, rolling window boundaries $k=23,24,25,167,168,335,336,719,720$, and Train/Val/Test split boundaries — plus $100$ random timestamps $T$, totaling **`2,399` region-hour rows**):
1. **Future Truncation Invariance at $T$:** All alerts with `started_at > T` are removed, and for alerts ongoing at $T$ (`finished_at > T`), `finished_at` is masked (`NaT`, treated as open at $T$). From this truncated history, the test independently recomputes the calm filter `is_active_T == 0` and all features across Sets A, B, C, and C' (`lagged_onset_1h`, `burden_last_6h`, `burden_last_24h`, `adaptive_rate_7d/14d/30d`, `national_active_count`, `onsets_last_24h`, `hours_since_last_alert`, `neighbors_active_count`, `neighbor_any_active`, `neighbor_active_ratio`, `neighbors_active_lag_1h/2h/3h`, and calendar transforms), verifying numerical agreement with `ml_features_hourly.parquet`.
2. **Target Window $[T, T + 1\text{h})$ Dependence:** Verifies that `target_onset_next_1h` matches the binary indicator computed solely from alerts with `started_at` in $[T, T + 1\text{h})$.

### ⚠️ Methodological Limitations & Caveats
1. **Post-Hoc Selection of `Set B` and `Set C'`:** The primary comparison was pre-specified as `Set C` vs. the adaptive baseline (`Adaptive 14-day Rolling`); identifying `Set B` as the top-performing model by PR-AUC and evaluating the reduced `Set C'` configuration were done after inspecting Test results and are post-hoc.
2. **Scope of `test_leakage_invariance.py` vs. `cleaner.py`:** `tests/test_leakage_invariance.py` operates on the cleaned intervals in `alerts_clean_oblast.parquet`. While it verifies the causal invariance of `feature_store.py`, **it does not inspect upstream preprocessing inside `src/data/cleaner.py`** — specifically micro-alert glitch filtering ($< 60$ s using `finished_at - started_at`, where filtering an alert that started less than 60 seconds before $T$ relies on its completion time) and overlapping interval merging.
3. **Sub-Regional Aggregation After 2025-08-01:** Reconstructing oblast alerts as the union of intervals across `oblast`, `raion`, and `hromada` preserves time-series continuity after the logging transition, but alters the distribution of `national_active_count` and `hours_since_last_alert` (as localized district alerts are treated as oblast-wide activity).
4. **Train Boundary Duration Censoring:** In `generate_train_only_clusters()`, durations of alerts crossing `2024-04-30 23:59:59 UTC` are censored at `train_end`, whereas descriptive RQ2 profiles (`regional_clusters_summary.csv`) span the full 2022–2026 period.
5. **Shared Validation Set for Early Stopping and Threshold Tuning:** The Validation split is used both for LightGBM early stopping (`stopping_rounds=30`) and for selecting the classification threshold $p^* = \arg\max F_2$.
6. **UTC Split Boundaries vs. Local Time Features:** Chronological split boundaries (`2024-04-30 23:59:59 UTC` and `2025-05-31 23:59:59 UTC`) are defined in UTC, whereas diurnal cyclical features and Historical Lookup tables operate in `Europe/Kyiv` time.

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Yaroo-a/ukrainian-air-raid-sirens-analysis.git
cd ukrainian-air-raid-sirens-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Reproducible Pipeline from Scratch
```bash
# Stage 1: Cleaning, interval aggregation, cyclicity, clustering (RQ2 + Train-only ML), and propagation
python src/data/cleaner.py
python src/features/cyclicity.py
python src/features/clustering.py
python src/features/propagation.py

# Stage 2: Feature Store generation, feature shift analysis, baselines, model training (A, B, C, C'), and Test evaluation
python src/features/feature_store.py
python src/features/analyze_feature_shift.py
python src/models/baselines.py
python src/models/train_models.py
python src/models/evaluate_test.py

# Artifact verification and causal truncation invariance tests
python tests/verify_pipeline.py
python tests/test_leakage_invariance.py
```

### 3. Interactive Jupyter Notebooks
```bash
jupyter lab
```
* `notebooks/01_data_profiling_and_cleaning.ipynb` — Raw data profiling, deduplication, and interval merging.
* `notebooks/02_rq1_temporal_cyclicity.ipynb` — Weekly/diurnal patterns and Kruskal-Wallis tests.
* `notebooks/03_rq2_regional_clustering.ipynb` — Agglomerative regional clustering.
* `notebooks/04_rq3_spatial_propagation_routes.ipynb` — Lead-lag cross-correlations between neighbors.
* `notebooks/05_ml_forecasting_experiment.ipynb` — ML experiments (A, B, C, C'), feature shift analysis, Ablation Study, and Test sub-period evaluations.

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/official_data_uk.csv               # Raw dataset (299,275 rows, SHA-256 verified)
│   └── processed/
│       ├── alerts_clean_oblast.parquet        # Merged oblast alert intervals (84,908 rows)
│       └── ml_features_hourly.parquet         # Calm-period Feature Store (797,428 rows)
├── models/                                    # Saved models (LightGBM A/B/C/C', LogReg A/B/C/C', optimal_thresholds.json)
├── notebooks/                                 # 5 executed Jupyter Notebooks
├── reports/
│   ├── figures/                               # 13 high-resolution figures (300 DPI)
│   ├── clusters_train_only.csv                # Train-only ML clusters (low/medium/high_activity)
│   ├── regional_clusters_summary.csv          # Full-period descriptive clusters (RQ2)
│   ├── spatial_lead_lag_summary.csv           # Cross-correlations for 49 neighbor pairs (RQ3)
│   ├── feature_shift_summary.csv              # Feature means across Train / Val / Test
│   ├── feature_shift_test_subperiods.csv      # Feature means inside Test pre- and post-2025-08-01
│   ├── baselines_benchmark.csv                # Validation baseline benchmarks
│   ├── ml_validation_benchmark.csv            # Validation ML benchmarks (all 11 models)
│   ├── ml_models_benchmark.csv                # Full Test benchmark (Day-Block Bootstrap N=500)
│   ├── ml_test_pre_aug2025_benchmark.csv      # Test benchmark before 2025-08-01
│   ├── ml_test_post_aug2025_benchmark.csv     # Test benchmark from 2025-08-01
│   ├── ml_recent_60d_benchmark.csv            # Latest 60-day evaluation benchmark
│   ├── test_evaluation_details.json           # Paired bootstrap differences across all 4 Test slices
│   └── feature_importance_lightgbm.csv        # LightGBM feature importance (Gain & Split)
├── src/
│   ├── data/cleaner.py                        # Deduplication and interval merging
│   ├── features/
│   │   ├── cyclicity.py                       # Temporal cyclicity analysis (RQ1)
│   │   ├── clustering.py                      # RQ2 clustering & Train-only ML cluster generation
│   │   ├── propagation.py                     # Spatial cross-correlations (RQ3)
│   │   ├── feature_store.py                   # Hourly ML Feature Store generator
│   │   └── analyze_feature_shift.py           # Feature distribution shift analysis (Train/Val/Test)
│   └── models/
│       ├── baselines.py                       # Naive, Historical Lookup, and Adaptive 14-day Rolling
│       ├── train_models.py                    # LogReg, LightGBM training (A, B, C, C') & threshold tuning
│       └── evaluate_test.py                   # Evaluation across 4 Test slices & Day-Block Bootstrap (N=500)
├── tests/
│   ├── verify_pipeline.py                     # End-to-end artifact integrity verification
│   └── test_leakage_invariance.py             # Future truncation & target window invariance test (A, B, C, C')
├── requirements.txt                           # Python dependencies
├── LICENSE                                    # MIT License
├── README.md                                  # Ukrainian documentation (Default)
└── README_EN.md                               # English documentation
```

---

## 📄 License
Released under the [MIT License](LICENSE). Primary data source: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset).
