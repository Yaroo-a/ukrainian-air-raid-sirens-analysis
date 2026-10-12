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
   * **Temporal Cyclicity (RQ1):** Statistically significant weekly cyclicity is confirmed on both datasets (Kruskal-Wallis $H = 25.52, \ p = 2.74 \times 10^{-4}$ on Official; $H = 17.05, \ p = 0.0091$ on Volunteer, with 3 statistically significant day pairs each after Bonferroni correction $\alpha_{\text{adj}} = 0.00238$), peaking Wednesday–Friday and reaching a minimum on Sunday, alongside a stable share of nocturnal alert onsets (`00:00–05:59 Europe/Kyiv`: **$20.2\% – 24.0\%$** across years on Official and **$21.0\% – 26.4\%$** on Volunteer).
   * **Regional Clustering (RQ2):** Isolated clustering fitted strictly on `Train <= 2024-04-30` for ML features (`low_activity`, `medium_activity`, `high_activity`: `7 / 5 / 12` split on Official and `7 / 7 / 10` on Volunteer) shows strong cross-source consistency, while full-period descriptive clustering cleanly separates 3 geographic belts (Deep Rear, Transit Center & North, Frontline/Border Zone).
   * **Spatial Propagation Corridors (RQ3):** Cross-correlations on a 10-minute discrete grid across 49 contiguous oblast pairs exhibit high cross-source agreement in joint active intervals ($\rho_{\text{Spearman}} = 0.965, \ p = 4.68 \times 10^{-29}$) and peak cross-correlations ($\rho_{\text{Spearman}} = 0.411, \ p = 0.0033$), capturing consistent 10–20 minute directional lead-lag couplings along southern, eastern, and central corridors.
2. **Stage 2 (Block A) — Static ML Forecasting & Diagnosing Covariate Shift:**
   * Under the formal pre-specified selection rule ($\arg\max \text{PR-AUC}$ on Validation `2024-05-01` – `2025-05-31`), **`LightGBM (Set C: Spatial + Adaptive, 22 features)`** won among all pre-specified candidates (`Baselines 1–3`, `LogReg A/B/C`, `LightGBM A/B/C`) on both Official (`Val PR-AUC = 0.3012`) and Volunteer (`0.3107` on `naive_as_is` / `0.3116` on `naive_fixed_D`).
   * On the full Official Test set (`2025-06` – `2026-09`), frozen static `LightGBM (Set C)` significantly outperforms the `Adaptive 14-day Baseline` ($\Delta\text{PR-AUC} = +0.0297 \ [+0.0220, +0.0373]$, $\Delta F_2 = +0.0249 \ [+0.0189, +0.0304]$), yet trails the simpler post-hoc `Set B` configuration (13 features without historical/national aggregates: `PR-AUC = 0.3741` vs. `0.3329`).
   * **Root Cause of Static `Set C` Degradation:** Between `Train` (`2022–04.2024`) and `Test` (`2025-06..2026`), on Official the mean number of simultaneously active oblasts (`national_active_count`) increased **$4.73\times$** (`1.17` $\to$ `5.54`, and after the August 1, 2025 sub-regional logging transition from `3.17` to `5.95`) and `neighbors_active_count` increased **$4.04\times$** (`0.13` $\to$ `0.52`), whereas on the independent Volunteer dataset (which remained at oblast-level logging) `national_active_count` increased **$1.72\times$** (`1.26` $\to$ `2.17`), `adaptive_rate_14d` increased **$1.66\times$** (`0.084` $\to$ `0.140`), and `onsets_last_24h` increased **$1.70\times$** (`1.95` $\to$ `3.32`). Static decision trees frozen on 2022–2024 data extrapolate with outdated split thresholds.
3. **Stage 2 (Block B) — Causal `naive` Sensitivity on Volunteer & Monthly Walk-Forward Retraining:**
   * Evaluating three strictly causal policies for handling the `naive` flag in Volunteer records (`naive_as_is`, `drop_naive`, and `naive_fixed_D` using median duration $D_{\text{oblast}}$ estimated strictly on `Train`) confirmed complete invariance of the static findings.
   * Transitioning to **monthly retraining (`LightGBM Set C [WF Rolling 12m]`, 12m Train + 2m Val with a 2-hour `embargo` buffer)** substantially improves gradient boosting performance:
     * **Official:** Pooled `PR-AUC` rises from `0.3329` (Static) to **`0.4203 [0.409, 0.431]`** ($\Delta_{\text{vs Adapt}} = +0.1171 \ [+0.1081, +0.1261]$, $\Delta_{\text{WF − Static}} = +0.0873 \ [+0.0789, +0.0954]$, winning `16/16` months), and on the latest 60 days from `0.3809` to **`0.5139 [0.488, 0.541]`** ($\Delta_{\text{vs Adapt}} = +0.1450 \ [+0.1177, +0.1726]$).
     * **Volunteer (`naive_fixed_D`):** `LightGBM (Set C) [WF Rolling 12m]` statistically significantly outperforms `Adaptive 14-day` on both Full Test (`0.4050 [0.397, 0.413]` vs. `0.3924`, $\Delta_{\text{vs Adapt}} = +0.0128 \ [+0.0082, +0.0168]$, $\Delta_{\text{WF − Static}} = +0.0147 \ [+0.0110, +0.0180]$, winning `15/17` months) and the latest 60 days (`0.4975 [0.486, 0.512]` vs. `0.4810`, $\Delta_{\text{vs Adapt}} = +0.0167 \ [+0.0089, +0.0252]$, $\Delta_{\text{WF − Static}} = +0.0430 \ [+0.0331, +0.0520]$).
     * **⚠️ Mandatory Caveat on Volunteer:** On the Volunteer dataset, `LightGBM (Set C) [WF Rolling 12m]` is **not unconditionally best across all metrics** — in `PR-AUC` ranking, it slightly trails the linear `LogReg (Set C) [Static Initial Train]` (`0.4050` vs. `0.4097` on Full Test, $\Delta = -0.0046 \ [-0.0081, -0.0013]$; `0.4975` vs. `0.5002` on the latest 60 days, $\Delta = -0.0027 \ [-0.0073, +0.0028]$), outperforming `LogReg Static` only on thresholded classification $F_2$-Score ($\Delta F_2 = +0.0064 \ [+0.0044, +0.0083]$) and probability calibration (`Brier = 0.1093` vs. `0.2645`).

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
| **Step 2: Micro-Glitch Filter ($< 60\text{ s}$)** | `-453` records (`duration < 60s` or $<0$) $\to$ **`184,977`** records | `-1,365` records (`duration < 60s`) $\to$ **`113,525`** records (`5,659` with `naive == True`) |
| **Step 3: Level / `naive` Policy** | Pre-`2025-08-01`: `level == 'oblast'` only (`-12,188` sub-regional rows) $\to$ **`172,789`** records; from `2025-08-01`: union of `oblast/raion/hromada` | 3 causal policies for `naive == True`:<br>• `naive_as_is` (+30 min)<br>• `drop_naive` (drop `-5,659` rows)<br>• `naive_fixed_D` (median $D_{\text{oblast}}$ strictly from `Train <= 2024-04-30`) |
| **Step 4: Overlapping Interval Merge** | Consolidates `-87,881` nested records $\to$ **`84,908`** intervals (`84,907` across 24 oblasts + `1` in Luhansk) | • `naive_as_is`: **`113,525`** intervals (`113,523` in 24 oblasts + `2` occupied)<br>• `drop_naive`: **`107,866`** intervals (`107,866` in 24 oblasts)<br>• `naive_fixed_D`: **`113,525`** intervals (`113,523` in 24 oblasts + `2` occupied) |
| **Step 5: Hourly Feature Store (`is_active_T == 0`)** | [`data/processed/ml_features_hourly.parquet`](data/processed/ml_features_hourly.parquet):<br>**`797,428`** calm region-hours (`0` duplicates) | • `as_is`: **`871,505`** calm region-hours<br>• `drop_naive`: **`874,375`** calm region-hours<br>• `fixed_D`: **`870,938`** calm region-hours |

### Why Does the Official Test Table Have `197,083` Rows While the Slice From April 2025 Has `228,371`?
This distinction comes directly from the chronological split design and the Walk-Forward validation window:
* In [`data/processed/ml_features_hourly.parquet`](data/processed/ml_features_hourly.parquet), the `split_set` column partitions the `797,428` calm region-hours into three disjoint splits separated by a 2-hour `embargo` buffer:
  * **`Train` (`split_set == "train"`, `2022-03-15` – `2024-04-30`):** `410,150` calm hours (`33,240` onsets, base rate **$8.10\%$**).
  * **`Validation` (`split_set == "val"`, `2024-05-01` – `2025-05-31`):** `190,195` calm hours (`19,342` onsets, base rate **$10.17\%$**).
  * **`Test` (`split_set == "test"`, `2025-06-01` – `2026-09-07`):** exactly **`197,083`** calm hours (`20,250` onsets, base rate **$10.27\%$**).
* When running the monthly `Walk-Forward Retraining` protocol ([`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py)), predicting the first test month `2025-06` requires a 2-month internal validation window `Val_M` (`2025-04-01` – `2025-05-31`, with a 2-hour `embargo`), which comprises **`31,288`** calm hours from `split_set == "val"`.
* Consequently:
  $$\underbrace{31,288}_{\text{April–May 2025 (internal Val}_M\text{)}} + \underbrace{197,083}_{\text{Official Test (from 2025-06-01)}} = 228,371\text{ rows.}$$
* **Which number is the true Test size in the final tables and OOF predictions?** Both the final benchmark tables (`ml_models_benchmark.csv`, `walk_forward_overall_benchmark.csv`) and the saved out-of-fold predictions file [`data/processed/walk_forward_oof_predictions_official.parquet`](data/processed/walk_forward_oof_predictions_official.parquet) contain exactly **`197,083` rows** — strictly `split_set == "test"` from `2025-06-01 00:00 UTC` (and on Volunteer `naive_fixed_D`, [`walk_forward_oof_predictions_volunteer_fixed_d.parquet`](data/processed/walk_forward_oof_predictions_volunteer_fixed_d.parquet) contains exactly **`256,490` rows** of `split_set == "test"`).

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

All three exploratory research questions were evaluated in parallel on Official (`84,907` intervals across 24 oblasts) and Volunteer (`113,523` intervals across 24 oblasts in `naive_as_is`; `108,576` over the shared time window up to `2026-09-07`).

### 2.1. Temporal Cyclicity & Alert Durations (RQ1)
* **Weekly Rhythm (Kruskal-Wallis Test, [`src/features/cyclicity.py`](src/features/cyclicity.py)):**
  * **Official:** $H = 25.52, \ p = 2.74 \times 10^{-4}$ (3 statistically significant day pairs after Bonferroni correction $\alpha_{\text{adj}} = 0.00238$). Peak activity occurs Wednesday–Friday (median 52–53 merged oblast alerts/day, mean 53.4–55.0), reaching a minimum on Sunday (median 47, mean 46.8 alerts/day).
  * **Volunteer (`naive_as_is`):** $H = 17.05, \ p = 0.0091$ (also 3 statistically significant day pairs after Bonferroni correction), reproducing the mid-week peak and Sunday trough.
* **Diurnal Profile & Nocturnal Alerts (`00:00–05:59 Europe/Kyiv`):** After aggregating district-level alerts in 2025–2026, the nocturnal onset share remains stable between **$20.2\%$ and $24.0\%$** across years on Official (2022: $24.0\%$, 2023: $21.5\%$, 2024: $20.2\%$, 2025: $20.5\%$, 2026: $22.9\%$) and **$21.0\% – 26.4\%$** on Volunteer (2022: $23.8\%$, 2023: $21.0\%$, 2024: $21.4\%$, 2025: $26.4\%$, 2026: $22.5\%$).
* **Cross-Source Duration Comparison ([`reports/comparison/official_vs_volunteer_comparison.csv`](reports/comparison/official_vs_volunteer_comparison.csv)):** Over the shared window `2022-03-15 .. 2026-09-07`, median duration on Official is **43.15 min** (mean **102.40 min**) vs. **30.68 min** (mean **49.06 min**) on Volunteer, because merging all district intervals within an oblast after August 2025 on Official creates longer continuous oblast envelopes containing multiple shorter Volunteer waves ($64.5\%$ of post-December 2025 Volunteer starts occur inside an ongoing Official interval).

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
* After filtering out synchronized nationwide alerts ($\ge 15$ oblasts simultaneously) on a 10-minute discrete grid, cross-correlations with lags $\tau \in [-180, +180]$ min were computed for all 49 contiguous region pairs ([`reports/official/spatial_lead_lag_summary.csv`](reports/official/spatial_lead_lag_summary.csv), [`reports/volunteer/spatial_lead_lag_summary.csv`](reports/volunteer/spatial_lead_lag_summary.csv)).
* Spearman rank correlation between the counts of joint active 10-minute intervals (`joint_active_intervals`) across the 49 neighbor pairs on Official and Volunteer is **$\rho = 0.965$** ($p = 4.68 \times 10^{-29}$), and between peak cross-correlations (`max_correlation`) is **$\rho = 0.411$** ($p = 0.0033$). On Official, strongest couplings occur along southern, eastern, and central corridors: Mykolaiv $\to$ Kherson ($r = 0.658$, lag 10 min), Dnipropetrovsk $\leftrightarrow$ Zaporizhzhia ($r = 0.613$, lag 0 min), Ivano-Frankivsk $\to$ Lviv ($r = 0.602$, lag 10 min), Kyiv City $\to$ Kyiv Oblast ($r = 0.578$, lag 10 min), Chernihiv $\to$ Sumy ($r = 0.574$, lag 10 min).

<p align="center">
  <img src="reports/official/figures/06_spatial_cross_correlation_routes.png" alt="Spatial Cross-Correlation Routes" width="85%">
  <br><em>Figure 4. Top-15 adjacent region pairs by cross-correlation strength and directional lead-lag breakdown (Official).</em>
</p>

---

## 🤖 3. Stage 2 (Block A): Static ML Forecasting, Covariate Shift & `naive` Sensitivity

### 3.1. Problem Formulation, Feature Set Hierarchy & Formal Validation Selection Rule
* **Prediction Unit $(T, \text{oblast})$:** Regular hourly grid in UTC (`timestamp_utc`), conditioned on calm hours (`is_active_T == 0`).
* **Target Variable (`target_onset_next_1h`):** $y_T = 1$ if a new alert starts in the region during the half-open interval **$[T, T + 1\text{h})$**.
* **Feature Set Hierarchy (all computed strictly from data prior to $T$, `closed="left"`, on the full hourly grid before calm filtering):**
  * **Set A (Time, Burden & Train Cluster — 10 features):** `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `month` (`Europe/Kyiv`), `burden_last_6h`, `burden_last_24h` (fraction of active hours in $[T-6\text{h}, T)$ and $[T-24\text{h}, T)$), and Train-only cluster indicators (`cluster_1_low_activity`, `cluster_2_medium_activity`, `cluster_3_high_activity`).
  * **Set B (A + Spatial at $T$ — 13 features):** Set A + instantaneous status of contiguous neighbors at exact moment $T$: `neighbors_active_count`, `neighbor_any_active`, `neighbor_active_ratio`.
  * **Set C (B + Adaptive & Spatio-Temporal Memory — 22 features, Pre-Specified Primary Configuration):** Set B + 9 historical adaptive and spatio-temporal features: `adaptive_rate_7d/14d/30d`, `national_active_count`, `onsets_last_24h`, `hours_since_last_alert` (`clip(upper=720.0)`), `neighbors_active_lag_1h/2h/3h`.
  * **Set C' (`FEATURE_SET_C_PRIME`: B + Adaptive & Neighbor Lags Only — 19 features, Post-Hoc Experiment):** Set C excluding the three most strongly shifted features (`national_active_count`, `hours_since_last_alert`, `onsets_last_24h`).
  * **Set D (`FEATURE_SET_D_POST_HOC`: Set C + 3 relative activity features `+Rel Ratios` — 25 features, Post-Hoc Exploratory in Walk-Forward):** Set C + 3 pointwise causal relative activity features in [`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py): `rel_activity_24h_vs_14d`, `adaptive_ratio_7d_vs_30d`, `adaptive_diff_14d_minus_30d`.
* **Formal Pre-Specified Validation Selection Rule ([`reports/official/val_evaluation_details.json`](reports/official/val_evaluation_details.json)):**
  * Pre-specified selection rule: $\arg\max \text{PR-AUC}$ on Validation (`2024-05-01` – `2025-05-31`) across pre-specified candidates (`Baselines 1–3`, `LogReg A/B/C`, `LightGBM A/B/C`).
  * Validation Winner: **`LightGBM (Set C: Spatial + Adaptive)`** on both Official (`Val PR-AUC = 0.3012 [0.289, 0.314]`, `Val F2 = 0.4910 [0.479, 0.502]`) and Volunteer (`Val PR-AUC = 0.3107 [0.299, 0.323]` in `naive_as_is` and `0.3116 [0.300, 0.324]` in `naive_fixed_D`).
  * Identifying static `Set B` as top-performing by Test PR-AUC and constructing `Set C'` and `Set D` were performed **after inspecting Test results and are explicitly labeled `post-hoc`**.

### Table 3.1. Static Model Benchmark on Full Official Test (`2025-06-01` – `2026-09-07`, `197,083` rows, [`reports/official/ml_models_benchmark.csv`](reports/official/ml_models_benchmark.csv))
*95% confidence intervals are computed via **Day-Block Bootstrap** resampling calendar days (`n_bootstraps = 500`). `Prec @ R=78.2%` reports Precision at the exact Recall level of the Adaptive Baseline (`Recall = 78.23%`).*

| Model | Feature Set | Protocol Status | Threshold $p^*$ (Val) | Test PR-AUC [95% CI] | Test $F_2$ [95% CI] | Precision | Recall | Prec @ R=78.2% | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | Lagged onset in $[T-1\text{h}, T)$ | `pre-specified candidate` | 0.500 | — | 0.1034 [0.096, 0.111] | 29.1% | 8.9% | — | 0.1159 |
| **Baseline 2: Historical Lookup** | Train lookup rate (Kyiv) | `pre-specified candidate` | 0.085 | 0.2204 [0.212, 0.228] | 0.5033 [0.495, 0.511] | 22.7% | 72.3% | 21.1% | 0.0868 |
| **Baseline 3: Adaptive 14-day** | 14-day rolling onset rate | `pre-specified candidate` | 0.089 | 0.3031 [0.292, 0.315] | 0.5299 [0.520, 0.538] | 23.1% | 78.2% | 23.1% | 0.0818 |
| **LogReg (Set A: No Spatial)** | A (10 features) | `pre-specified candidate` | 0.518 | 0.3881 [0.377, 0.399] | 0.5329 [0.524, 0.541] | 23.5% | 77.9% | 23.4% | 0.2007 |
| **LogReg (Set B: With Spatial)** | B (13 features) | `pre-specified candidate` | 0.561 | **0.3903** [0.380, 0.400] | 0.5499 [0.541, 0.558] | 24.0% | 81.2% | 25.1% | 0.2337 |
| **LogReg (Set C: Spatial + Adaptive)** | C (22 features) | `pre-specified candidate` | 0.565 | 0.3173 [0.306, 0.328] | 0.5431 [0.535, 0.550] | 21.1% | **89.3%** | 25.3% | 0.2875 |
| **LogReg (Set C': Adaptive + Lags)** | C' (19 features) | `post-hoc sensitivity` | 0.541 | 0.3386 [0.327, 0.350] | 0.5527 [0.544, 0.561] | 24.1% | 81.6% | 25.6% | 0.2265 |
| **LightGBM (Set A: No Spatial)** | A (10 features) | `pre-specified candidate` | 0.093 | 0.3482 [0.337, 0.359] | 0.5379 [0.529, 0.546] | 25.4% | 74.6% | 23.5% | 0.0784 |
| **LightGBM (Set B: With Spatial)** | B (13 features) | `pre-specified candidate` *(post-hoc best PR-AUC)* | 0.109 | 0.3741 [0.363, 0.386] | 0.5610 [0.552, 0.570] | 23.5% | 85.8% | 26.8% | **0.0761** |
| **LightGBM (Set C: Spatial + Adaptive)** | C (22 features) | **`pre-specified winner (Val PR-AUC)`** | 0.129 | 0.3329 [0.323, 0.343] | 0.5551 [0.547, 0.562] | 23.0% | 85.8% | 26.3% | 0.0783 |
| **LightGBM (Set C': Adaptive + Lags)** | C' (19 features) | `post-hoc sensitivity` | 0.129 | 0.3522 [0.342, 0.363] | **0.5683** [0.559, 0.577] | **25.7%** | 81.6% | **27.1%** | 0.0767 |

### Table 3.2. Static Model Benchmark on Latest 60 Days Official (`2026-07-09` – `2026-09-07`, `23,646` rows, base rate `12.65%`, [`reports/official/ml_recent_60d_benchmark.csv`](reports/official/ml_recent_60d_benchmark.csv))

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
| **LightGBM (Set C: Spatial + Adaptive)** *(Val winner)* | C (22 features) | 0.129 | 0.3809 [0.348, 0.416] | 0.6301 [0.603, 0.654] | 27.3% | 93.5% | 34.4% | 0.0895 |
| **LightGBM (Set C': Adaptive + Lags)** *(post-hoc)* | C' (19 features) | 0.129 | 0.3999 [0.367, 0.437] | 0.6527 [0.628, 0.677] | 31.2% | 89.9% | 36.0% | 0.0868 |

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
| **`(a) naive_as_is`** (`started_at + 30 min`) | `871,505` / `256,807` | `15.31%` (`39,328`) | `0.3922 [0.383, 0.401]`<br>`0.5900 [0.584, 0.596]` | `0.3877 [0.381, 0.395]`<br>`0.5951 [0.589, 0.601]` | $-0.0042 \ [-0.0087, +0.0002]$ | **`+0.0050 [+0.0025, +0.0079]`** |
| **`(b) drop_naive`** (drop `naive == True`) | `874,375` / `258,051` | `14.41%` (`37,189`) | `0.3881 [0.379, 0.397]`<br>`0.5728 [0.565, 0.579]` | `0.3843 [0.377, 0.392]`<br>`0.5821 [0.576, 0.588]` | $-0.0034 \ [-0.0083, +0.0014]$ | **`+0.0092 [+0.0066, +0.0119]`** |
| **`(c) naive_fixed_D`** (median $D_{\text{oblast}}$ strictly from `Train`) | `870,938` / `256,490` | `15.33%` (`39,328`) | `0.3924 [0.384, 0.401]`<br>`0.5905 [0.584, 0.596]` | `0.3902 [0.383, 0.397]`<br>`0.5960 [0.590, 0.602]` | $-0.0019 \ [-0.0065, +0.0024]$ | **`+0.0055 [+0.0030, +0.0082]`** |

<p align="center">
  <img src="reports/comparison/figures/18_volunteer_naive_sensitivity_comparison.png" alt="Volunteer naive Sensitivity Comparison" width="90%">
  <br><em>Figure 7. Sensitivity of PR-AUC and F2-Score on the Volunteer Test set across the three causal policies for handling the naive flag (Figure 18 in reports/comparison/figures/).</em>
</p>

<details>
<summary><b>📊 Expand Additional Static Benchmark Tables: Pre/Post 2025-08-01 Sub-Periods & Ablation Study Matrix (Official)</b></summary>

#### Table 3.3. Static Model Results Inside Test Before and After `2025-08-01` ([`reports/official/ml_test_pre_aug2025_benchmark.csv`](reports/official/ml_test_pre_aug2025_benchmark.csv), [`reports/official/ml_test_post_aug2025_benchmark.csv`](reports/official/ml_test_post_aug2025_benchmark.csv))

| Model | Pre-01.08.2025 (`28,929`): PR-AUC [95% CI] | Pre-01.08.2025: $F_2$ [95% CI] | Pre-01.08: Prec / Rec (Prec@R=68.6%) | Post-01.08.2025 (`168,154`): PR-AUC [95% CI] | Post-01.08.2025: $F_2$ [95% CI] | Post-01.08: Prec / Rec (Prec@R=79.8%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3: Adaptive 14-day** | 0.2735 [0.259, 0.288] | 0.4605 [0.444, 0.477] | 19.9% / 68.6% (17.4%) | 0.3068 [0.294, 0.320] | 0.5411 [0.531, 0.551] | 23.7% / 79.8% (23.9%) |
| **LogReg (Set A)** | 0.2934 [0.275, 0.313] | 0.4617 [0.443, 0.479] | 18.1% / 75.6% (17.3%) | 0.4025 [0.392, 0.413] | 0.5458 [0.537, 0.555] | 24.7% / 78.2% (24.7%) |
| **LogReg (Set B)** | 0.2877 [0.272, 0.306] | 0.4814 [0.466, 0.498] | 19.9% / 74.6% (19.1%) | **0.4040** [0.395, 0.415] | 0.5614 [0.553, 0.570] | 24.7% / 82.3% (26.2%) |
| **LogReg (Set C)** | 0.2579 [0.243, 0.275] | 0.4979 [0.483, 0.513] | 20.5% / 77.4% (20.3%) | 0.3266 [0.314, 0.339] | 0.5498 [0.541, 0.559] | 21.2% / **91.2%** (26.6%) |
| **LogReg (Set C')** *(post-hoc)* | 0.2783 [0.264, 0.294] | 0.4813 [0.467, 0.495] | 21.5% / 69.6% (19.2%) | 0.3470 [0.334, 0.359] | 0.5638 [0.554, 0.574] | 24.5% / 83.5% (26.7%) |
| **LightGBM (Set A)** | 0.2892 [0.270, 0.310] | 0.4712 [0.455, 0.488] | 19.8% / 71.8% (18.1%) | 0.3577 [0.348, 0.368] | 0.5497 [0.541, 0.560] | **26.6%** / 75.0% (24.7%) |
| **LightGBM (Set B)** *(post-hoc)* | 0.3105 [0.293, 0.328] | 0.5071 [0.492, 0.522] | 21.0% / 78.4% (21.0%) | 0.3837 [0.373, 0.394] | 0.5696 [0.561, 0.579] | 23.9% / 87.0% (27.8%) |
| **LightGBM (Set C)** *(Val winner)* | 0.3006 [0.284, 0.321] | 0.5044 [0.487, 0.522] | 23.3% / 71.2% (21.0%) | 0.3386 [0.327, 0.350] | 0.5623 [0.553, 0.571] | 23.0% / 88.1% (27.1%) |
| **LightGBM (Set C')** *(post-hoc)* | **0.3124** [0.295, 0.330] | **0.5106** [0.493, 0.530] | **25.1%** / 68.9% (**21.0%**) | 0.3582 [0.346, 0.370] | **0.5769** [0.568, 0.586] | 25.7% / 83.7% (**27.8%**) |

#### Table 3.4. Paired Bootstrap Metric Differences Across 4 Test Slices ([`reports/official/test_evaluation_details.json`](reports/official/test_evaluation_details.json), Day-Block Bootstrap $N=500$)

| Comparison (Paired Difference, Official) | Full Test: $\Delta$ PR-AUC / $\Delta F_2$ | Pre-01.08.2025: $\Delta$ PR-AUC / $\Delta F_2$ | Post-01.08.2025: $\Delta$ PR-AUC / $\Delta F_2$ | Latest 60 Days: $\Delta$ PR-AUC / $\Delta F_2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Primary (pre-specified):** `LGBM (C) - Adaptive 14d` | **$+0.0297$** $[+0.022, +0.037]$<br>**$+0.0249$** $[+0.019, +0.030]$ | **$+0.0274$** $[+0.012, +0.046]$<br>**$+0.0441$** $[+0.027, +0.062]$ | **$+0.0319$** $[+0.024, +0.040]$<br>**$+0.0212$** $[+0.016, +0.027]$ | $+0.0126$ $[-0.006, +0.031]$<br>**$-0.0133$** $[-0.022, -0.005]$ |
| **Post-hoc:** `LGBM (C') - Adaptive 14d` | **$+0.0492$** $[+0.043, +0.056]$<br>**$+0.0384$** $[+0.033, +0.043]$ | **$+0.0389$** $[+0.027, +0.051]$<br>**$+0.0502$** $[+0.032, +0.070]$ | **$+0.0514$** $[+0.045, +0.059]$<br>**$+0.0358$** $[+0.031, +0.041]$ | **$+0.0314$** $[+0.011, +0.051]$<br>$+0.0094$ $[-0.0001, +0.019]$ |
| **Post-hoc:** `LGBM (C') - LGBM (B)` | **$-0.0219$** $[-0.029, -0.015]$<br>**$+0.0075$** $[+0.004, +0.011]$ | $+0.0019$ $[-0.008, +0.013]$<br>$+0.0035$ $[-0.008, +0.013]$ | **$-0.0254$** $[-0.034, -0.017]$<br>**$+0.0074$** $[+0.004, +0.010]$ | **$-0.0533$** $[-0.074, -0.033]$<br>**$+0.0086$** $[+0.001, +0.016]$ |
| **Post-hoc:** `LGBM (C') - LGBM (C)` | **$+0.0194$** $[+0.014, +0.025]$<br>**$+0.0134$** $[+0.010, +0.017]$ | $+0.0115$ $[-0.001, +0.024]$<br>$+0.0061$ $[-0.001, +0.013]$ | **$+0.0195$** $[+0.013, +0.025]$<br>**$+0.0146$** $[+0.011, +0.018]$ | **$+0.0189$** $[+0.008, +0.030]$<br>**$+0.0228$** $[+0.016, +0.030]$ |
| **LGBM:** `Set B - Set A` (neighbors at $T$) | **$+0.0259$** $[+0.023, +0.029]$<br>**$+0.0230$** $[+0.017, +0.029]$ | **$+0.0212$** $[+0.013, +0.030]$<br>**$+0.0364$** $[+0.024, +0.049]$ | **$+0.0260$** $[+0.022, +0.029]$<br>**$+0.0198$** $[+0.014, +0.026]$ | $+0.0039$ $[-0.006, +0.014]$<br>$-0.0126$ $[-0.035, +0.007]$ |
| **LGBM:** `Set C - Set B` | **$-0.0414$** $[-0.050, -0.033]$<br>**$-0.0060$** $[-0.010, -0.003]$ | $-0.0096$ $[-0.023, +0.006]$<br>$-0.0026$ $[-0.014, +0.008]$ | **$-0.0448$** $[-0.054, -0.035]$<br>**$-0.0072$** $[-0.011, -0.004]$ | **$-0.0722$** $[-0.095, -0.050]$<br>**$-0.0142$** $[-0.021, -0.007]$ |
</details>

---

## 🔄 4. Stage 2 (Block B): Monthly Walk-Forward Retraining

### 4.1. Causal Walk-Forward Protocol ([`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py))
To evaluate how periodic tree refitting adapts to the 2025–2026 alert regime, [`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py) implements a strictly causal monthly `Walk-Forward Retraining` protocol:
* **Refit Cadence:** Every calendar month $M$ of the Test split starting from `2025-06` (`16` test months `2025-06..2026-09` on Official; `17` test months `2025-06..2026-10` on Volunteer `naive_fixed_D`).
* **Training & Internal Validation Windows Before Month $M$:**
  * `WF Rolling 12m`: 12 calendar months of training data `Train_M` + the 2 most recent months prior to $M$ as internal validation `Val_M` (used for LightGBM `early_stopping(30)` and selecting $p^*_M = \arg\max F_2(\text{Val}_M)$).
  * `WF Expanding`: Full history from March 2022 up to `Val_M` + 2 months `Val_M`.
* **Strict 2-Hour `embargo` Buffer:** Between `Train_M` and `Val_M`, and between `Val_M` and the start of test month $M$, the final 2 hours are excluded (`embargo_hours = 2`), preventing any overlap of target windows $[T, T+1\text{h})$ across split boundaries.
* **Hypothesis Status in the Walk-Forward Block (`protocol_status`):**
  * `LightGBM (Set C) [WF Rolling 12m]` — **`primary hypothesis of the WF block (specified after viewing static Test results)`**.
  * `LogReg (Set C) [WF Rolling 12m]` — `exploratory (pre-specified Set C)`.
  * `LightGBM / LogReg (Set C) [WF Expanding]` — `exploratory (expanding window)`.
  * `LightGBM / LogReg (Set D: +Rel Ratios) [WF Rolling 12m / Expanding, post-hoc]` — `post-hoc exploratory (Set D)` (testing the 3 dimensionless relative activity ratios in `Set D`, 25 features).
* **Static Reference in WF Tables (`Static Initial Train`):** Uses the frozen artifact [`models/lightgbm_adaptive.joblib`](models/lightgbm_adaptive.joblib) ($p^*=0.1290$) on Official and the frozen artifact **[`models/volunteer_lightgbm_adaptive_fixed_d.joblib`](models/volunteer_lightgbm_adaptive_fixed_d.joblib)** ($p^*=0.1131$, trained and evaluated strictly on the causal `naive_fixed_D` grid) on Volunteer.

### Table 4.1. Consolidated Walk-Forward Benchmark vs. Static Models & Baselines (All 14 Rows From [`reports/comparison/official_vs_volunteer_walk_forward_comparison.csv`](reports/comparison/official_vs_volunteer_walk_forward_comparison.csv))

| Row / Model in WF Protocol | Protocol Status (`protocol_status`) | Official Full Test (`197,083`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) | Official Latest 60 Days (`23,646`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) | Volunteer `fixed_D` Full Test (`256,490`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) | Volunteer `fixed_D` Latest 60 Days (`30,347`): Pooled PR-AUC [95% CI] / $F_2$ [95% CI] ($\Delta$ PR-AUC vs Adapt) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Reference: Predict-All ($p^* = 0.0$)** | `reference (base-rate F2 floor)` | —<br>0.3640 [0.354, 0.373] | —<br>0.4201 [0.395, 0.442] | —<br>0.4752 [0.467, 0.484] | —<br>0.5885 [0.575, 0.603] |
| **2. Baseline 3: Adaptive 14-day (Static Val $p^*$)** | `reference baseline (frozen Val p*)` | 0.3031 [0.292, 0.315]<br>0.5299 [0.520, 0.538]<br>($+0.0000$) | 0.3682 [0.333, 0.406]<br>0.6430 [0.620, 0.664]<br>($+0.0000$) | 0.3924 [0.384, 0.401]<br>0.5905 [0.584, 0.596]<br>($+0.0000$) | 0.4810 [0.469, 0.496]<br>0.6664 [0.657, 0.676]<br>($+0.0000$) |
| **3. Baseline 3: Adaptive 14-day (WF $p^*_M$)** | `PRIMARY FAIR BASELINE (WF p*_M)` | 0.3031 [0.292, 0.315]<br>0.5275 [0.518, 0.536]<br>($+0.0000$) | 0.3682 [0.333, 0.406]<br>0.6399 [0.615, 0.665]<br>($+0.0000$) | 0.3924 [0.384, 0.401]<br>0.5901 [0.584, 0.596]<br>($+0.0000$) | 0.4810 [0.469, 0.496]<br>0.6656 [0.656, 0.675]<br>($+0.0000$) |
| **4. LightGBM (Set C) [Static Initial Train]** | `reference static model (Val winner, frozen .joblib)` | 0.3329 [0.323, 0.343]<br>0.5551 [0.547, 0.562]<br>($+0.0297 \ [+0.0220, +0.0373]$) | 0.3809 [0.348, 0.416]<br>0.6301 [0.603, 0.654]<br>($+0.0126 \ [-0.0055, +0.0314]$) | 0.3902 [0.383, 0.397]<br>0.5960 [0.590, 0.602]<br>($-0.0019 \ [-0.0065, +0.0024]$) | 0.4546 [0.440, 0.472]<br>0.6674 [0.658, 0.677]<br>($-0.0262 \ [-0.0360, -0.0156]$) |
| **5. LogReg (Set C) [Static Initial Train]** | `reference static model` | 0.3173 [0.306, 0.328]<br>0.5431 [0.535, 0.550]<br>($+0.0140 \ [+0.0089, +0.0199]$) | 0.3889 [0.354, 0.423]<br>0.6196 [0.592, 0.643]<br>($+0.0205 \ [+0.0052, +0.0349]$) | **0.4097** [0.401, 0.418]<br>0.5924 [0.586, 0.598]<br>(**$+0.0174 \ [+0.0150, +0.0200]$**) | **0.5002** [0.487, 0.515]<br>0.6635 [0.655, 0.673]<br>(**$+0.0194 \ [+0.0133, +0.0257]$**) |
| **6. LightGBM (Set C) [WF Rolling 12m]** | `primary hypothesis of the WF block (specified after viewing static Test results)` | **0.4203** [0.409, 0.431]<br>**0.5757** [0.566, 0.584]<br>(**$+0.1171 \ [+0.1081, +0.1261]$**) | 0.5139 [0.488, 0.541]<br>0.6597 [0.636, 0.684]<br>(**$+0.1450 \ [+0.1177, +0.1726]$**) | 0.4050 [0.397, 0.413]<br>0.5989 [0.594, 0.605]<br>(**$+0.0128 \ [+0.0082, +0.0168]$**) | 0.4975 [0.486, 0.512]<br>0.6685 [0.659, 0.678]<br>(**$+0.0167 \ [+0.0089, +0.0252]$**) |
| **7. LogReg (Set C) [WF Rolling 12m]** | `exploratory (pre-specified Set C)` | 0.3964 [0.386, 0.407]<br>0.5636 [0.555, 0.572]<br>($+0.0932 \ [+0.0849, +0.1017]$) | 0.4719 [0.446, 0.499]<br>**0.6647** [0.641, 0.688]<br>($+0.1039 \ [+0.0798, +0.1293]$) | 0.4057 [0.397, 0.414]<br>0.5952 [0.589, 0.601]<br>($+0.0134 \ [+0.0105, +0.0163]$) | 0.4980 [0.486, 0.513]<br>0.6688 [0.660, 0.679]<br>($+0.0171 \ [+0.0108, +0.0230]$) |
| **8. LightGBM (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]** | `post-hoc exploratory (Set D)` | 0.4190 [0.409, 0.429]<br>0.5753 [0.567, 0.583]<br>($+0.1158 \ [+0.1066, +0.1248]$) | 0.5082 [0.482, 0.533]<br>0.6593 [0.635, 0.682]<br>($+0.1394 \ [+0.1100, +0.1695]$) | 0.4053 [0.397, 0.413]<br>0.5984 [0.593, 0.604]<br>($+0.0129 \ [+0.0085, +0.0172]$) | 0.4979 [0.486, 0.512]<br>0.6677 [0.658, 0.678]<br>($+0.0171 \ [+0.0092, +0.0255]$) |
| **9. LogReg (Set D: +Rel Ratios) [WF Rolling 12m, post-hoc]** | `post-hoc exploratory (Set D)` | 0.3964 [0.386, 0.407]<br>0.5631 [0.554, 0.572]<br>($+0.0932 \ [+0.0848, +0.1018]$) | 0.4718 [0.447, 0.499]<br>0.6638 [0.640, 0.687]<br>($+0.1038 \ [+0.0797, +0.1291]$) | 0.4060 [0.398, 0.415]<br>0.5951 [0.589, 0.601]<br>($+0.0137 \ [+0.0107, +0.0165]$) | 0.4981 [0.486, 0.513]<br>0.6705 [0.662, 0.680]<br>($+0.0171 \ [+0.0109, +0.0230]$) |
| **10. LightGBM (Set C) [WF Expanding]** | `exploratory (expanding window)` | 0.4227 [0.412, 0.433]<br>0.5737 [0.565, 0.581]<br>($+0.1195 \ [+0.1108, +0.1282]$) | **0.5165** [0.492, 0.542]<br>0.6598 [0.637, 0.681]<br>(**$+0.1477 \ [+0.1228, +0.1738]$**) | 0.4077 [0.400, 0.415]<br>0.5990 [0.593, 0.605]<br>($+0.0155 \ [+0.0117, +0.0195]$) | 0.4979 [0.485, 0.513]<br>0.6688 [0.659, 0.678]<br>($+0.0171 \ [+0.0092, +0.0252]$) |
| **11. LogReg (Set C) [WF Expanding]** | `exploratory (expanding window)` | 0.3673 [0.356, 0.378]<br>0.5603 [0.551, 0.568]<br>($+0.0641 \ [+0.0575, +0.0709]$) | 0.4291 [0.398, 0.460]<br>0.6612 [0.638, 0.683]<br>($+0.0608 \ [+0.0429, +0.0795]$) | 0.4100 [0.402, 0.418]<br>0.5940 [0.588, 0.600]<br>($+0.0177 \ [+0.0150, +0.0203]$) | 0.4987 [0.486, 0.514]<br>0.6656 [0.657, 0.675]<br>($+0.0178 \ [+0.0116, +0.0242]$) |
| **12. LightGBM (Set D: +Rel Ratios) [WF Expanding, post-hoc]** | `post-hoc exploratory (Set D, expanding)` | **0.4235** [0.413, 0.434]<br>0.5717 [0.563, 0.579]<br>(**$+0.1203 \ [+0.1113, +0.1294]$**) | 0.5145 [0.490, 0.540]<br>0.6582 [0.636, 0.680]<br>($+0.1457 \ [+0.1213, +0.1711]$) | 0.4081 [0.400, 0.416]<br>**0.5997** [0.594, 0.606]<br>($+0.0159 \ [+0.0121, +0.0198]$) | 0.4988 [0.486, 0.514]<br>**0.6705** [0.662, 0.680]<br>($+0.0179 \ [+0.0098, +0.0259]$) |
| **13. LogReg (Set D: +Rel Ratios) [WF Expanding, post-hoc]** | `post-hoc exploratory (Set D, expanding)` | 0.3680 [0.357, 0.379]<br>0.5600 [0.551, 0.568]<br>($+0.0648 \ [+0.0579, +0.0718]$) | 0.4311 [0.400, 0.462]<br>0.6597 [0.638, 0.682]<br>($+0.0628 \ [+0.0446, +0.0820]$) | **0.4100** [0.402, 0.418]<br>0.5941 [0.588, 0.600]<br>(**$+0.0177 \ [+0.0150, +0.0203]$**) | 0.4988 [0.486, 0.514]<br>0.6670 [0.658, 0.676]<br>($+0.0179 \ [+0.0117, +0.0243]$) |
| **14. LogReg Static проти LGBM WF (`Set C: LGBM_WF − LogReg_Static`)** | `direct paired comparison (LGBM WF Rolling 12m − LogReg Static)` | **$\Delta\text{PR-AUC} = +0.1031 \ [+0.0944, +0.1120]$**<br>$\Delta F_2 = +0.0325 \ [+0.0276, +0.0371]$<br>$\Delta\text{Brier} = -0.2140$ | **$\Delta\text{PR-AUC} = +0.1245 \ [+0.1030, +0.1479]$**<br>$\Delta F_2 = +0.0407 \ [+0.0305, +0.0521]$<br>$\Delta\text{Brier} = -0.2164$ | **$\Delta\text{PR-AUC} = -0.0046 \ [-0.0081, -0.0013]$**<br>$\Delta F_2 = +0.0064 \ [+0.0044, +0.0083]$<br>$\Delta\text{Brier} = -0.1552$ | $\Delta\text{PR-AUC} = -0.0027 \ [-0.0073, +0.0028]$<br>$\Delta F_2 = +0.0049 \ [+0.0006, +0.0097]$<br>$\Delta\text{Brier} = -0.1709$ |

<p align="center">
  <img src="reports/comparison/figures/19_walk_forward_vs_static_comparison.png" alt="Monthly Walk-Forward vs Static Dynamics" width="95%">
  <br><em>Figure 8. Monthly PR-AUC and F2-Score trajectory for Walk-Forward Retraining (Rolling 12m), static LightGBM (Set C), and Adaptive 14-day Baseline across Official and Volunteer (Figure 19 in reports/comparison/figures/).</em>
</p>

### 4.2. Walk-Forward Interpretation, `LogReg Static vs. LGBM WF` Caveat & Non-Causal Phrasing
1. **Why Row 14 (`LogReg Static проти LGBM WF`) Is Explicitly Reported:**
   * On **Official**, monthly `LightGBM (Set C) [WF Rolling 12m]` outperforms all static and linear alternatives, beating `LogReg Static` by **$\Delta\text{PR-AUC} = +0.1031 \ [+0.0944, +0.1120]$** on Full Test and **$+0.1245 \ [+0.1030, +0.1479]$** on the latest 60 days.
   * On **Volunteer (`naive_fixed_D`)**, however, while monthly refitting significantly lifts `LightGBM (Set C) [WF Rolling 12m]` over static LightGBM ($\Delta_{\text{WF − Static}} = +0.0147 \ [+0.0110, +0.0180]$ on Full Test and $+0.0430 \ [+0.0331, +0.0520]$ on the latest 60 days) and beats `Adaptive 14-day` in `15/17` months, **`LightGBM` cannot be called unconditionally best on Volunteer**. Static linear `LogReg (Set C) [Static Initial Train]`, which extrapolates monotonically under rising alert activity, retains a higher `PR-AUC` on Full Test (`0.4097` vs. `0.4050`, $\Delta\text{PR-AUC} = -0.0046 \ [-0.0081, -0.0013]$) and is statistically indistinguishable on the latest 60 days (`0.5002` vs. `0.4975`, $\Delta\text{PR-AUC} = -0.0027 \ [-0.0073, +0.0028]$). `LightGBM WF` outperforms `LogReg Static` on Volunteer only on thresholded $F_2$-Score ($\Delta F_2 = +0.0064 \ [+0.0044, +0.0083]$) and probability calibration (`Brier = 0.1093` vs. `0.2645`).
2. **Non-Causal Phrasing (`is consistent with` rather than causal claims):**
   * The observed gain of `LightGBM (Set C) [WF Rolling 12m]` over frozen `LightGBM Static` on Official ($\Delta_{\text{WF − Static}} = +0.0873 \ [+0.0789, +0.0954]$ on Full Test and $+0.1324 \ [+0.1142, +0.1510]$ on the latest 60 days) and on Volunteer ($\Delta_{\text{WF − Static}} = +0.0147 \ [+0.0110, +0.0180]$ on Full Test and $+0.0430 \ [+0.0331, +0.0520]$ on the latest 60 days) **is consistent with (`узгоджується з`) the hypothesis** that periodic refitting adapts tree split thresholds to the post-June 2025 regime shift and relieves split saturation when features exceed their 2022–2024 training range. Because no separate causal intervention test was conducted to isolate a single cause, we strictly use the phrasing "is consistent with".

### 4.3. Hyperparameter Alignment Audit: Initial WF Run vs. Aligned Run (`hyperparameter_alignment_history`)
As documented in [`reports/volunteer/volunteer_walk_forward_details.json`](reports/volunteer/volunteer_walk_forward_details.json) (`methodological_notes.hyperparameter_alignment_history`) and [`notebooks/05_ml_forecasting_experiment.ipynb`](notebooks/05_ml_forecasting_experiment.ipynb), during the initial (pre-commit) run of [`src/models/run_walk_forward_experiment.py`](src/models/run_walk_forward_experiment.py), `fit_fold_lightgbm` used different hyperparameters (`n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, min_child_samples=120, class_weight='balanced', eval_metric='average_precision'`) than the canonical static training script [`src/models/train_models.py`](src/models/train_models.py) (`n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=50`, no `class_weight`, `eval_metric='binary_logloss'`), and `Static LightGBM` was refit with those same parameters instead of loading the frozen `.joblib` artifact (while in the first WF commit `58628f8`, `Static Initial Train` on Volunteer used `volunteer_lightgbm_adaptive.joblib` trained on `naive_as_is`). After aligning `fit_fold_lightgbm` to match `train_models.py` and switching `Static Initial Train` to [`models/volunteer_lightgbm_adaptive_fixed_d.joblib`](models/volunteer_lightgbm_adaptive_fixed_d.joblib) ($p^*=0.1131$), the conclusion for `LightGBM (Set C) [WF Rolling 12m]` on Volunteer changed from statistical parity with `Adaptive 14-day` to statistically significantly outperforming it:

#### Table 4.2. Initial Run vs. Aligned Run on Volunteer (`naive_fixed_D`, [`reports/volunteer/volunteer_walk_forward_details.json`](reports/volunteer/volunteer_walk_forward_details.json))

| Evaluation Slice (`Volunteer, naive_fixed_D`) | Model (`Set C`) | Pre-Commit Run (unaligned WF hyperparams) / Commit `58628f8` (`Static` from `as_is`): Pooled PR-AUC [95% CI] & $\Delta$ vs. Adaptive | Aligned Run (`train_models.py` hyperparams; `Static` from `fixed_d.joblib`): Pooled PR-AUC [95% CI] & $\Delta$ vs. Adaptive |
| :--- | :--- | :--- | :--- |
| **Full Test** (`2025-06` — `2026-10`, `256,490` rows) | **LightGBM (Set C) [WF Rolling 12m]** | Pre-commit: `0.3935 [0.385, 0.401]`<br>$\Delta = +0.0012 \ [-0.0040, +0.0067]$ *(parity with Adaptive)* | **`0.4050 [0.397, 0.413]`**<br>**$\Delta = +0.0128 \ [+0.0082, +0.0168]$** *(beats Adaptive)* |
| **Full Test** (`2025-06` — `2026-10`, `256,490` rows) | **LightGBM (Set C) [Static Initial Train]** | Pre-commit: `0.3684 [0.360, 0.377]`, $\Delta = -0.0239 \ [-0.0285, -0.0194]$<br>Commit `58628f8` (`as_is.joblib`): `0.3881 [0.381, 0.395]`, $\Delta = -0.0040 \ [-0.0085, +0.0004]$ | **`0.3902 [0.383, 0.397]`** (`fixed_d.joblib`)<br>$\Delta = -0.0019 \ [-0.0065, +0.0024]$ |
| **Latest 60 Days** (`2026-08-12` — `2026-10-11`, `30,347` rows) | **LightGBM (Set C) [WF Rolling 12m]** | Pre-commit: `0.4933 [0.480, 0.509]`<br>$\Delta = +0.0125 \ [+0.0042, +0.0215]$ | **`0.4975 [0.486, 0.512]`**<br>**$\Delta = +0.0167 \ [+0.0089, +0.0252]$** *(beats Adaptive)* |
| **Latest 60 Days** (`2026-08-12` — `2026-10-11`, `30,347` rows) | **LightGBM (Set C) [Static Initial Train]** | Pre-commit: `0.4261 [0.415, 0.440]`, $\Delta = -0.0548 \ [-0.0641, -0.0449]$<br>Commit `58628f8` (`as_is.joblib`): `0.4519 [0.438, 0.468]`, $\Delta = -0.0290 \ [-0.0390, -0.0182]$ | **`0.4546 [0.440, 0.472]`** (`fixed_d.joblib`)<br>$\Delta = -0.0262 \ [-0.0360, -0.0156]$ |

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
| **Official** | `high_activity (активність 2022–04.2024)` | 12 (2) | 29.08% | 0.3100 / 0.3460 / **0.3373** | **`+0.0494` / `+0.0234`** | **`+0.0908` / `+0.0542`** | **`10 / 12`** | **`12 / 12`** |
| **Official** | `medium_activity (активність 2022–04.2024)` | 5 (0) | 6.11% | 0.0720 / 0.1592 / **0.1887** | **`+0.0265` / `+0.0297`** | **`+0.1167` / `+0.1057`** | **`5 / 5`** | **`5 / 5`** |
| **Official** | `low_activity (активність 2022–04.2024)` | 7 (1) | 1.74% | 0.0189 / 0.1321 / **0.1255** | **`+0.0020` / `+0.0033`** | **`+0.1137` / `+0.1066`** | **`4 / 7`** | **`7 / 7`** |
| **Volunteer (`naive_fixed_D`)** | `high_activity (активність 2022–04.2024)` | 10 (0) | 28.87% | 0.3156 / 0.3358 / **0.3474** | **`+0.0162` / `+0.0111`** | **`+0.0325` / `+0.0383`** | **`7 / 10`** | **`10 / 10`** |
| **Volunteer (`naive_fixed_D`)** | `medium_activity (активність 2022–04.2024)` | 7 (0) | 9.89% | 0.1329 / 0.1907 / **0.1932** | **`+0.0097` / `+0.0036`** | **`+0.0525` / `+0.0465`** | **`4 / 7`** | **`7 / 7`** |
| **Volunteer (`naive_fixed_D`)** | `low_activity (активність 2022–04.2024)` | 7 (1) | 4.40% | 0.0224 / 0.0690 / **0.0931** | **`+0.0061` / `+0.0034`** | **`+0.0292` / `+0.0228`** | **`4 / 7`** | **`7 / 7`** |

* Across all 24 oblasts, `LightGBM (Set C) [WF Rolling 12m]` outperforms `Adaptive 14-day` in **`24/24` oblasts on Official** and **`24/24` oblasts on Volunteer (`naive_fixed_D`)**, and outperforms frozen `Static Initial Train` in **`19/24` oblasts on Official** and **`15/24` oblasts on Volunteer (`naive_fixed_D`)**. The largest gains from monthly retraining occur in frontline and central oblasts within `high_activity` and `medium_activity`.

<p align="center">
  <img src="reports/comparison/figures/20_walk_forward_regional_breakdown.png" alt="Regional Walk-Forward Breakdown Across 24 Oblasts" width="95%">
  <br><em>Figure 9. Paired within-oblast ΔPR-AUC (WF Rolling 12m − Static Initial Train) across all 24 oblasts of Ukraine grouped by Train-only clusters (2022–04.2024), with low_sample_flag (*) indicators (Figure 20 in reports/comparison/figures/).</em>
</p>

<details>
<summary><b>🗺️ Expand Complete 24-Oblast Walk-Forward Table (`reports/comparison/official_vs_volunteer_walk_forward_regional.csv`)</b></summary>

| Oblast | Train Cluster (Official) | Official: hrs / onsets (`base_rate`) | Official: PR-AUC Adapt / Static / WF 12m | Official: $\Delta_{\text{WF−Static}}$ [95% CI] | Volunteer (`fixed_D`): hrs / onsets (`base_rate`) | Volunteer: PR-AUC Adapt / Static / WF 12m | Volunteer: $\Delta_{\text{WF−Static}}$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Донецька область** | `high_activity` | 1,037 / 623 (60.1%)* | 0.5606 / 0.5964 / 0.6582 | $+0.0618 \ [+0.0209, +0.1043]$ | 9,010 / 1,808 (20.1%) | 0.2273 / 0.2235 / 0.2640 | $+0.0405 \ [+0.0316, +0.0497]$ |
| **Харківська область** | `high_activity` | 2,957 / 1,549 (52.4%) | 0.5011 / 0.5612 / 0.7003 | $+0.1390 \ [+0.1106, +0.1647]$ | 9,463 / 4,460 (47.1%) | 0.5634 / 0.5736 / 0.6068 | $+0.0332 \ [+0.0220, +0.0449]$ |
| **Сумська область** | `high_activity` | 2,714 / 1,419 (52.3%) | 0.4856 / 0.5765 / 0.7025 | $+0.1260 \ [+0.0992, +0.1524]$ | 9,496 / 3,419 (36.0%) | 0.4041 / 0.4269 / 0.4442 | $+0.0173 \ [+0.0082, +0.0273]$ |
| **Запорізька область** | `high_activity` | 4,378 / 1,664 (38.0%) | 0.3621 / 0.4357 / 0.5901 | $+0.1544 \ [+0.1275, +0.1768]$ | 9,522 / 2,567 (27.0%) | 0.2962 / 0.3124 / 0.3421 | $+0.0297 \ [+0.0190, +0.0410]$ |
| **Чернігівська область** | `high_activity` | 4,416 / 1,408 (31.9%) | 0.4124 / 0.4663 / 0.5112 | $+0.0448 \ [+0.0173, +0.0728]$ | 9,608 / 2,723 (28.3%) | 0.3351 / 0.3592 / 0.3527 | $-0.0065 \ [-0.0161, +0.0040]$ |
| **Полтавська область** | `high_activity` | 7,663 / 1,769 (23.1%) | 0.2908 / 0.3035 / 0.3258 | $+0.0223 \ [+0.0092, +0.0364]$ | 10,253 / 2,320 (22.6%) | 0.2698 / 0.2866 / 0.2843 | $-0.0023 \ [-0.0093, +0.0044]$ |
| **Миколаївська область** | `high_activity` | 8,919 / 1,733 (19.4%) | 0.3203 / 0.3498 / 0.3418 | $-0.0080 \ [-0.0236, +0.0060]$ | 10,348 / 2,622 (25.3%) | 0.3821 / 0.3982 / 0.4032 | $+0.0049 \ [-0.0061, +0.0150]$ |
| **Херсонська область** | `high_activity` | 8,498 / 1,629 (19.2%) | 0.2574 / 0.2913 / 0.3093 | $+0.0180 \ [+0.0017, +0.0334]$ | 10,399 / 2,268 (21.8%) | 0.2463 / 0.2931 / 0.2927 | $-0.0004 \ [-0.0091, +0.0096]$ |
| **Одеська область** | `high` (Off) / `med` (Vol) | 9,265 / 1,642 (17.7%) | 0.2997 / 0.3423 / 0.3329 | $-0.0094 \ [-0.0247, +0.0054]$ | 10,886 / 2,097 (19.3%) | 0.3055 / 0.3196 / 0.3232 | $+0.0036 \ [-0.0053, +0.0124]$ |
| **Кіровоградська область** | `high_activity` | 9,313 / 1,319 (14.2%) | 0.1946 / 0.2201 / 0.2287 | $+0.0086 \ [-0.0059, +0.0224]$ | 10,656 / 1,667 (15.6%) | 0.2233 / 0.2362 / 0.2378 | $+0.0016 \ [-0.0057, +0.0097]$ |
| **Черкаська область** | `high` (Off) / `med` (Vol) | 9,858 / 1,034 (10.5%) | 0.1222 / 0.1678 / 0.1785 | $+0.0107 \ [+0.0001, +0.0228]$ | 11,003 / 1,238 (11.3%) | 0.1427 / 0.1907 / 0.1899 | $-0.0007 \ [-0.0089, +0.0092]$ |
| **Дніпропетровська область** | `high_activity` | 1,338 / 137 (10.2%)* | 0.1024 / 0.0947 / 0.1191 | $+0.0244 \ [+0.0081, +0.0610]$ | 9,918 / 4,439 (44.8%) | 0.5469 / 0.5477 / 0.5917 | $+0.0440 \ [+0.0339, +0.0561]$ |
| **Київська область** | `medium_activity` | 9,011 / 1,082 (12.0%) | 0.1513 / 0.1938 / 0.2235 | $+0.0297 \ [+0.0118, +0.0509]$ | 10,965 / 1,721 (15.7%) | 0.2916 / 0.3274 / 0.3381 | $+0.0107 \ [-0.0002, +0.0207]$ |
| **м. Київ** | `medium_activity` | 10,304 / 685 (6.6%) | 0.0830 / 0.1582 / 0.1887 | $+0.0306 \ [+0.0107, +0.0486]$ | 11,288 / 950 (8.4%) | 0.1284 / 0.1424 / 0.1418 | $-0.0005 \ [-0.0097, +0.0071]$ |
| **Житомирська область** | `medium_activity` | 10,096 / 578 (5.7%) | 0.0720 / 0.1952 / 0.2184 | $+0.0232 \ [+0.0015, +0.0438]$ | 11,250 / 795 (7.1%) | 0.1329 / 0.1969 / 0.2005 | $+0.0036 \ [-0.0081, +0.0150]$ |
| **Вінницька область** | `medium_activity` | 10,523 / 409 (3.9%) | 0.0419 / 0.1177 / 0.1476 | $+0.0300 \ [+0.0142, +0.0494]$ | 11,550 / 550 (4.8%) | 0.0982 / 0.1409 / 0.1355 | $-0.0055 \ [-0.0189, +0.0076]$ |
| **Хмельницька область** | `medium_activity` | 10,769 / 248 (2.3%) | 0.0248 / 0.1592 / 0.1783 | $+0.0191 \ [-0.0120, +0.0448]$ | 11,502 / 318 (2.8%) | 0.0557 / 0.1367 / 0.1932 | $+0.0564 \ [+0.0335, +0.0802]$ |
| **Рівненська область** | `low_activity` | 10,675 / 279 (2.6%) | 0.0350 / 0.1668 / 0.1870 | $+0.0203 \ [-0.0042, +0.0454]$ | 11,599 / 391 (3.4%) | 0.0856 / 0.0951 / 0.0984 | $+0.0034 \ [-0.0108, +0.0168]$ |
| **Волинська область** | `low_activity` | 10,854 / 208 (1.9%) | 0.0235 / 0.1362 / 0.1505 | $+0.0144 \ [-0.0177, +0.0449]$ | 10,263 / 1,299 (12.7%) | 0.1724 / 0.1862 / 0.1855 | $-0.0006 \ [-0.0108, +0.0082]$ |
| **Тернопільська область** | `low_activity` | 10,895 / 184 (1.7%) | 0.0184 / 0.1543 / 0.1575 | $+0.0033 \ [-0.0338, +0.0424]$ | 11,465 / 208 (1.8%) | 0.0223 / 0.0690 / 0.0931 | $+0.0241 \ [+0.0022, +0.0461]$ |
| **Чернівецька область** | `low_activity` | 10,899 / 180 (1.7%) | 0.0189 / 0.1286 / 0.1255 | $-0.0031 \ [-0.0356, +0.0272]$ | 11,493 / 207 (1.8%) | 0.0224 / 0.0362 / 0.0493 | $+0.0131 \ [+0.0006, +0.0315]$ |
| **Львівська область** | `low_activity` | 10,917 / 165 (1.5%) | 0.0190 / 0.1321 / 0.1190 | $-0.0131 \ [-0.0536, +0.0232]$ | 11,134 / 937 (8.4%) | 0.1494 / 0.1729 / 0.1722 | $-0.0007 \ [-0.0135, +0.0114]$ |
| **Івано-Франківська область** | `low_activity` | 10,932 / 159 (1.5%) | 0.0165 / 0.1158 / 0.1044 | $-0.0114 \ [-0.0440, +0.0186]$ | 11,789 / 177 (1.5%) | 0.0173 / 0.0537 / 0.0606 | $+0.0070 \ [-0.0125, +0.0255]$ |
| **Закарпатська область** | `low_activity` | 10,852 / 147 (1.4%)* | 0.0149 / 0.0946 / 0.0981 | $+0.0035 \ [-0.0299, +0.0417]$ | 11,630 / 147 (1.3%)* | 0.0152 / 0.0339 / 0.0300 | $-0.0039 \ [-0.0148, +0.0127]$ |

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
| **3. Future Truncation Invariance of Feature Store at $119$ Timestamps $T$** | Across $19$ boundary timestamps (grid start, rolling window boundaries $6\text{h}/24\text{h}/7\text{d}/14\text{d}/30\text{d}$, Train/Val/Test split boundaries) + $100$ random timestamps $T$ (`2,399` region-hour rows on Official and `2,465` on Volunteer `naive_fixed_D`), all alerts with `started_at > T` are deleted and `finished_at > T` is masked (`NaT`) | Exact agreement between recomputed `is_active_T == 0` and all features in Sets A, B, C, C', D vs. the saved Feature Store (`max_diff < 1e-6`) |
| **4. Target Window $[T, T + 1\text{h})$ Dependence** | Verifies `target_onset_next_1h` across all test rows on both datasets | Exact match with binary onset indicator strictly within the half-open interval $[T, T + 1\text{h})$ |
| **5. Causal `naive_fixed_D` Medians in `cleaner.py`** | Physical truncation of raw `volunteer_data_uk.csv` at `2024-04-30 23:59:59 UTC` with censoring of `finished_at` at `train_end` inside `compute_train_median_durations_volunteer()` | Exact match of $D_{\text{oblast}}$ across all 24 regions when `Val + Test` rows are removed |
| **6. Walk-Forward Split, `embargo` & Saved OOF Invariance** | Verifies the 2-hour `embargo_hours = 2` buffer between `Train_M`, `Val_M`, and test month $M$, weight invariance when corrupting all rows $\ge T_{\text{start}}(M) - 1\text{h}$, `SHA-256` verification, and exact reproduction of saved `p_roll_lgb_c` when trained on data physically truncated at $T_{\text{start}}(M) - 2\text{h}$ | Zero overlap of target windows $[T, T+1\text{h})$ and exact match of saved OOF Parquets against models trained on strictly truncated past data |

### 6.2. Methodological Limitations & Caveats
1. **Strict Separation of Pre-Specified vs. Post-Hoc Hypotheses:** The primary pre-specified comparison in the static block was `LightGBM (Set C)` vs. `Adaptive 14-day Rolling`. Identifying static `Set B` as top-performing by PR-AUC, constructing `Set C'`, adding the relative activity ratios `Set D (+Rel Ratios)`, and formulating the `Walk-Forward Retraining` block (`primary hypothesis of the WF block (specified after viewing static Test results)`) were done after viewing static Test results.
2. **Multiple Comparisons & `base_rate` Dependence in Regional Breakdown:** Evaluating 24 oblasts across two sources (`48` slices) is strictly `exploratory`. Raw within-oblast `PR-AUC` depends mechanically on regional `base_rate`, so only paired within-oblast differences $\Delta\text{PR-AUC}$ are meaningful across regions.
3. **Sub-Regional Aggregation After 2025-08-01 on Official:** Reconstructing oblast alerts as the union of all district intervals preserves series continuity after the August 2025 logging reform, but reduces calm hours in frontline oblasts (notably Donetsk and Dnipropetrovsk on Official, which are flagged with `low_sample_flag = True`).
4. **Boundary Effect of the 60-Second Micro-Alert Filter:** For an alert that started less than 60 seconds before $T$ and is still ongoing at $T$, retrospective filtering relies on whether its final duration reaches 60 seconds (affecting $< 0.05\%$ of grid hours).

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
[`run_all.sh`](run_all.sh) sequentially executes all 9 stages of the pipeline from scratch (upstream SHA-256 verification, Official & Volunteer cleaning, cross-source interval comparison, RQ1–RQ3, Feature Store generation, static training, Volunteer experiments, Walk-Forward Retraining and 24-oblast regional breakdown, execution of all 5 Jupyter Notebooks, and automated verification/leakage tests):
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
* [`notebooks/04_rq3_spatial_propagation_routes.ipynb`](notebooks/04_rq3_spatial_propagation_routes.ipynb) — Cross-correlations and lead-lag delays across 49 neighbor pairs.
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
